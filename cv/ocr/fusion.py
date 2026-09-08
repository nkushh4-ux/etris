from collections import defaultdict, deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isfinite
from typing import Any

from cv.ocr.india.formats import classify_india_plate
from cv.ocr.india.models import IndiaPlateFormat, PrefixStatus
from cv.ocr.models import (
    CharacterDiagnostic,
    EngineContribution,
    FusedPlateResult,
    FusionStatus,
    OCREvidence,
)
from cv.ocr.normalization import (
    DEFAULT_SEPARATORS,
    levenshtein_distance,
    normalize_plate_text,
    normalized_edit_distance,
)


@dataclass(frozen=True, slots=True)
class EnginePolicy:
    eligible_for_independent_accept: bool = True


@dataclass(frozen=True, slots=True)
class OCRFusionConfig:
    max_evidence_per_track: int = 12
    min_unique_frames_for_accept: int = 2
    max_cluster_edit_distance: int = 2
    max_cluster_normalized_edit_distance: float = 0.20
    min_cluster_weight_share: float = 0.70
    min_mean_position_confidence: float = 0.80
    min_position_confidence: float = 0.65
    min_character_support_frames: int = 2
    min_final_confidence: float = 0.82
    weight_ocr_confidence: float = 0.60
    weight_plate_quality: float = 0.40
    confidence_weight_cluster_share: float = 0.40
    confidence_weight_position: float = 0.40
    confidence_weight_evidence: float = 0.20
    separators: str = DEFAULT_SEPARATORS
    engine_policies: Mapping[str, EnginePolicy] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.max_evidence_per_track < 1:
            raise ValueError("max_evidence_per_track must be at least 1")
        if self.min_unique_frames_for_accept < 1:
            raise ValueError("min_unique_frames_for_accept must be at least 1")
        if self.min_character_support_frames < 1:
            raise ValueError("min_character_support_frames must be at least 1")
        if self.max_evidence_per_track < max(
            self.min_unique_frames_for_accept,
            self.min_character_support_frames,
        ):
            raise ValueError("max evidence must cover configured support minima")
        if self.max_cluster_edit_distance < 0:
            raise ValueError("max_cluster_edit_distance must be non-negative")
        probabilities = (
            self.max_cluster_normalized_edit_distance,
            self.min_cluster_weight_share,
            self.min_mean_position_confidence,
            self.min_position_confidence,
            self.min_final_confidence,
        )
        if any(not 0.0 <= value <= 1.0 for value in probabilities):
            raise ValueError("fusion probabilities must be within [0, 1]")
        evidence_weights = (
            self.weight_ocr_confidence,
            self.weight_plate_quality,
        )
        confidence_weights = (
            self.confidence_weight_cluster_share,
            self.confidence_weight_position,
            self.confidence_weight_evidence,
        )
        for weights in (evidence_weights, confidence_weights):
            if any(value < 0.0 for value in weights):
                raise ValueError("weights must be non-negative")
            if abs(sum(weights) - 1.0) > 1e-6:
                raise ValueError("weights must sum to approximately 1")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "OCRFusionConfig":
        data = dict(values)
        data.pop("enabled", None)
        policies = {
            name: EnginePolicy(**policy)
            for name, policy in data.pop("engine_policies", {}).items()
        }
        return cls(engine_policies=policies, **data)


@dataclass(frozen=True, slots=True)
class _WeightedEvidence:
    evidence: OCREvidence
    text: str
    weight: float


@dataclass(slots=True)
class _TrackAccumulator:
    evidence: list[_WeightedEvidence] = field(default_factory=list)
    fingerprints: deque[tuple] = field(default_factory=deque)
    rejected_ids: deque[str] = field(default_factory=deque)
    total_seen: int = 0
    duplicate_count: int = 0
    invalid_count: int = 0
    evicted_count: int = 0


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _compatible(first: str, second: str, config: OCRFusionConfig) -> bool:
    return (
        levenshtein_distance(first, second) <= config.max_cluster_edit_distance
        and normalized_edit_distance(first, second)
        <= config.max_cluster_normalized_edit_distance
    )


def _align(reference: str, observed: str) -> tuple[str, str]:
    """Globally align two strings with deterministic diagonal-first traceback."""

    rows, columns = len(reference) + 1, len(observed) + 1
    costs = [[0] * columns for _ in range(rows)]
    for row in range(rows):
        costs[row][0] = row
    for column in range(columns):
        costs[0][column] = column
    for row in range(1, rows):
        for column in range(1, columns):
            costs[row][column] = min(
                costs[row - 1][column - 1]
                + (reference[row - 1] != observed[column - 1]),
                costs[row - 1][column] + 1,
                costs[row][column - 1] + 1,
            )
    left, right = [], []
    row, column = len(reference), len(observed)
    while row or column:
        if row and column and costs[row][column] == (
            costs[row - 1][column - 1]
            + (reference[row - 1] != observed[column - 1])
        ):
            left.append(reference[row - 1])
            right.append(observed[column - 1])
            row -= 1
            column -= 1
        elif row and costs[row][column] == costs[row - 1][column] + 1:
            left.append(reference[row - 1])
            right.append("")
            row -= 1
        else:
            left.append("")
            right.append(observed[column - 1])
            column -= 1
    return "\0".join(reversed(left)), "\0".join(reversed(right))


def _effective_weights(
    evidence: list[_WeightedEvidence],
) -> dict[int, float]:
    """Cap each physical frame at its strongest normalized evidence weight.

    Evidence from engines on the same frame shares that frame-level budget in
    proportion to its individual weight. This preserves engine votes and
    provenance without treating engines as independent physical observations.
    """

    by_frame: dict[int, list[_WeightedEvidence]] = defaultdict(list)
    for item in evidence:
        by_frame[item.evidence.frame_index].append(item)
    effective: dict[int, float] = {}
    for items in by_frame.values():
        frame_budget = max(item.weight for item in items)
        raw_total = sum(item.weight for item in items)
        for item in items:
            effective[id(item)] = (
                frame_budget * item.weight / raw_total
                if raw_total else 0.0
            )
    return effective


def _medoid(
    cluster: list[_WeightedEvidence],
    weights: Mapping[int, float],
) -> str:
    texts = sorted({item.text for item in cluster})
    return min(
        texts,
        key=lambda candidate: (
            sum(
                weights[id(item)] * levenshtein_distance(candidate, item.text)
                for item in cluster
            ),
            candidate,
        ),
    )


def _consensus(
    cluster: list[_WeightedEvidence],
    weights: Mapping[int, float],
) -> tuple[str, tuple[CharacterDiagnostic, ...]]:
    reference = _medoid(cluster, weights)
    total_weight = sum(weights[id(item)] for item in cluster)
    base_votes = [defaultdict(float) for _ in reference]
    base_frames = [defaultdict(set) for _ in reference]
    insert_votes: dict[tuple[int, int], dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    insert_frames: dict[tuple[int, int], dict[str, set[int]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for item in cluster:
        vote_weight = weights[id(item)]
        aligned_reference, aligned_observed = _align(reference, item.text)
        base_index = 0
        insert_index = defaultdict(int)
        for left, right in zip(
            aligned_reference.split("\0"),
            aligned_observed.split("\0"),
        ):
            if left == "":
                slot = (base_index, insert_index[base_index])
                insert_index[base_index] += 1
                insert_votes[slot][right] += vote_weight
                insert_frames[slot][right].add(item.evidence.frame_index)
            else:
                base_votes[base_index][right] += vote_weight
                base_frames[base_index][right].add(item.evidence.frame_index)
                base_index += 1

    output: list[str] = []
    diagnostics: list[CharacterDiagnostic] = []

    def append_vote(votes, frames) -> None:
        gap_weight = total_weight - sum(votes.values())
        choices = [(gap_weight, "")] + [
            (weight, character) for character, weight in votes.items()
        ]
        support, character = min(
            choices,
            key=lambda choice: (-choice[0], choice[1]),
        )
        if not character:
            return
        output.append(character)
        diagnostics.append(CharacterDiagnostic(
            position=len(output) - 1,
            character=character,
            weighted_support=support,
            total_compatible_weight=total_weight,
            position_confidence=support / total_weight if total_weight else 0.0,
            supporting_frame_count=len(frames[character]),
        ))

    for base_index in range(len(reference) + 1):
        for slot in sorted(key for key in insert_votes if key[0] == base_index):
            append_vote(insert_votes[slot], insert_frames[slot])
        if base_index < len(reference):
            append_vote(base_votes[base_index], base_frames[base_index])
    return "".join(output), tuple(diagnostics)


class MultiFrameOCRFuser:
    """Bounded OCR evidence fusion isolated by camera and vehicle track."""

    def __init__(self, config: OCRFusionConfig | None = None, india_decoder=None) -> None:
        self.config = config or OCRFusionConfig()
        self.india_decoder = india_decoder
        self._tracks: dict[tuple[str, int], _TrackAccumulator] = {}

    def _weight(self, evidence: OCREvidence) -> float | None:
        if not (
            isfinite(evidence.ocr_confidence)
            and isfinite(evidence.plate_quality_score)
        ):
            return None
        return (
            self.config.weight_ocr_confidence
            * _clamp(evidence.ocr_confidence)
            + self.config.weight_plate_quality
            * _clamp(evidence.plate_quality_score)
        )

    def add_evidence(self, evidence: OCREvidence) -> bool:
        key = (evidence.camera_id, evidence.track_id)
        state = self._tracks.setdefault(key, _TrackAccumulator())
        state.total_seen += 1
        text = normalize_plate_text(
            evidence.normalized_text or evidence.raw_text,
            self.config.separators,
        )
        weight = self._weight(evidence)
        fingerprint = (
            evidence.camera_id,
            evidence.track_id,
            evidence.frame_index,
            evidence.engine,
            text,
        )
        if fingerprint in state.fingerprints:
            state.duplicate_count += 1
            self._remember_rejection(state, evidence.evidence_id)
            return False
        state.fingerprints.append(fingerprint)
        while len(state.fingerprints) > self.config.max_evidence_per_track * 4:
            state.fingerprints.popleft()
        if not text or weight is None:
            state.invalid_count += 1
            self._remember_rejection(state, evidence.evidence_id)
            return False
        candidate = _WeightedEvidence(evidence, text, weight)
        retained = self._retain(state.evidence + [candidate])
        accepted = any(item is candidate for item in retained)
        if not accepted:
            state.evicted_count += 1
            self._remember_rejection(state, evidence.evidence_id)
        else:
            evicted = [item for item in state.evidence if item not in retained]
            state.evicted_count += len(evicted)
            for item in evicted:
                self._remember_rejection(state, item.evidence.evidence_id)
        state.evidence = retained
        return accepted

    def _remember_rejection(self, state: _TrackAccumulator, evidence_id: str) -> None:
        state.rejected_ids.append(evidence_id)
        while len(state.rejected_ids) > self.config.max_evidence_per_track:
            state.rejected_ids.popleft()

    def _retain(self, items: list[_WeightedEvidence]) -> list[_WeightedEvidence]:
        rank = lambda item: (
            -item.weight,
            item.evidence.frame_index,
            item.evidence.engine,
            item.text,
            item.evidence.evidence_id,
        )
        ordered = sorted(items, key=rank)
        selected: list[_WeightedEvidence] = []
        frames = set()
        for item in ordered:
            if item.evidence.frame_index not in frames:
                selected.append(item)
                frames.add(item.evidence.frame_index)
                if len(selected) == self.config.max_evidence_per_track:
                    return sorted(selected, key=rank)
        for item in ordered:
            if item not in selected:
                selected.append(item)
                if len(selected) == self.config.max_evidence_per_track:
                    break
        return sorted(selected, key=rank)

    def get(self, camera_id: str, track_id: int) -> FusedPlateResult:
        return self._fuse(camera_id, track_id, self._tracks.get((camera_id, track_id)))

    def fuse(self, camera_id: str, track_id: int) -> FusedPlateResult:
        return self.get(camera_id, track_id)

    def finalize_track(self, camera_id: str, track_id: int) -> FusedPlateResult:
        state = self._tracks.pop((camera_id, track_id), None)
        return self._fuse(camera_id, track_id, state)

    def clear(self) -> None:
        self._tracks.clear()

    def _fuse(
        self,
        camera_id: str,
        track_id: int,
        state: _TrackAccumulator | None,
    ) -> FusedPlateResult:
        if state is None or not state.evidence:
            return self._empty_result(camera_id, track_id, state)
        evidence = state.evidence
        total_effective_weights = _effective_weights(evidence)
        total_weight = sum(total_effective_weights.values())
        cluster_candidates = []
        for center in sorted({item.text for item in evidence}):
            cluster = [
                item for item in evidence
                if _compatible(center, item.text, self.config)
            ]
            cluster_effective_weights = {
                id(item): total_effective_weights[id(item)] for item in cluster
            }
            cluster_weight = sum(cluster_effective_weights.values())
            frames = {item.evidence.frame_index for item in cluster}
            cluster_candidates.append((
                cluster_weight,
                len(frames),
                cluster_weight / len(frames),
                center,
                cluster,
            ))
        _, _, _, _, cluster = min(
            cluster_candidates,
            key=lambda item: (-item[0], -item[1], -item[2], item[3]),
        )
        cluster_effective_weights = {
            id(item): total_effective_weights[id(item)] for item in cluster
        }
        cluster_weight = sum(cluster_effective_weights.values())
        cluster_share = cluster_weight / total_weight if total_weight else 0.0
        text, characters = _consensus(cluster, cluster_effective_weights)
        position_confidences = [item.position_confidence for item in characters]
        mean_position = (
            sum(position_confidences) / len(position_confidences)
            if position_confidences else 0.0
        )
        minimum_position = min(position_confidences, default=0.0)
        all_frames = {item.evidence.frame_index for item in evidence}
        supporting_frames = {item.evidence.frame_index for item in cluster}
        mean_evidence_weight = (
            cluster_weight / len(supporting_frames)
            if supporting_frames else 0.0
        )
        confidence = _clamp(
            self.config.confidence_weight_cluster_share * cluster_share
            + self.config.confidence_weight_position * mean_position
            + self.config.confidence_weight_evidence * mean_evidence_weight
        )
        reasons = []
        if len(all_frames) < self.config.min_unique_frames_for_accept:
            reasons.append("insufficient_unique_frames")
        if not text or not characters:
            reasons.append("no_coherent_candidate")
        if cluster_share < self.config.min_cluster_weight_share:
            reasons.append("cluster_weight_share_below_threshold")
        if mean_position < self.config.min_mean_position_confidence:
            reasons.append("mean_position_confidence_below_threshold")
        if minimum_position < self.config.min_position_confidence:
            reasons.append("position_confidence_below_threshold")
        if any(
            item.supporting_frame_count < self.config.min_character_support_frames
            for item in characters
        ):
            reasons.append("character_frame_support_below_threshold")
        if confidence < self.config.min_final_confidence:
            reasons.append("final_confidence_below_threshold")
        policies = {
            item.evidence.engine: self.config.engine_policies.get(
                item.evidence.engine, EnginePolicy()
            )
            for item in cluster
        }
        if not any(policy.eligible_for_independent_accept for policy in policies.values()):
            reasons.append("no_independently_eligible_engine")
        if len(all_frames) < self.config.min_unique_frames_for_accept or not text:
            status = FusionStatus.UNKNOWN
            plate_text = None
        elif reasons:
            status = FusionStatus.LOW_CONFIDENCE
            plate_text = text
        else:
            status = FusionStatus.ACCEPTED
            plate_text = text
        constraint_diagnostic = None
        if self.india_decoder is not None and text:
            try:
                constraint_diagnostic = self.india_decoder.decode(text, (
                    (item.text, cluster_effective_weights[id(item)], item.evidence.frame_index)
                    for item in cluster
                ))
                text = constraint_diagnostic.constrained_text
                plate_text = text if status is not FusionStatus.UNKNOWN else None
                reasons.extend(constraint_diagnostic.reason_codes)
                if (constraint_diagnostic.prefix_status.value == "INVALID"
                        and status is FusionStatus.ACCEPTED):
                    status = FusionStatus.LOW_CONFIDENCE
                    reasons.append("invalid_state_prefix_constraint_downgrade")
            except Exception as exc:
                reasons.append(f"india_constraint_failure:{type(exc).__name__}")
        cluster_ids = {id(item) for item in cluster}
        conflicting = [
            item.evidence.evidence_id for item in evidence
            if id(item) not in cluster_ids
        ]
        contributions = []
        for engine in sorted({item.evidence.engine for item in cluster}):
            items = [item for item in cluster if item.evidence.engine == engine]
            policy = self.config.engine_policies.get(engine, EnginePolicy())
            contributions.append(EngineContribution(
                engine=engine,
                evidence_count=len(items),
                unique_frame_count=len({item.evidence.frame_index for item in items}),
                total_weight=sum(
                    cluster_effective_weights[id(item)] for item in items
                ),
                eligible_for_independent_accept=policy.eligible_for_independent_accept,
            ))
        if state.duplicate_count:
            reasons.append("duplicate_evidence_ignored")
        if state.invalid_count:
            reasons.append("invalid_evidence_ignored")
        if state.evicted_count:
            reasons.append("evidence_evicted_by_bound")
        # Canonical last-line guard: decoder errors or future rescoring changes
        # cannot allow an invalid state-series prefix to escape as ACCEPTED.
        constraints_enabled = bool(
            self.india_decoder is not None
            and getattr(getattr(self.india_decoder, "config", None), "enabled", False)
        )
        if constraints_enabled and text:
            final_validation = classify_india_plate(text)
            if (final_validation.format is IndiaPlateFormat.STANDARD_STATE
                    and (final_validation.prefix_status is PrefixStatus.INVALID
                         or not final_validation.is_structurally_valid)
                    and status is FusionStatus.ACCEPTED):
                status = FusionStatus.LOW_CONFIDENCE
                plate_text = text
                if final_validation.prefix_status is PrefixStatus.INVALID:
                    reasons.extend(("INVALID_STATE_PREFIX_NO_SUPPORTED_ALTERNATIVE",
                                    "invalid_state_prefix_constraint_downgrade"))
                if not final_validation.is_structurally_valid:
                    reasons.append("invalid_standard_structure_constraint_downgrade")
        return FusedPlateResult(
            status=status,
            plate_text=plate_text,
            confidence=confidence,
            camera_id=camera_id,
            track_id=track_id,
            total_evidence_count=state.total_seen,
            usable_evidence_count=len(evidence),
            unique_frame_count=len(all_frames),
            supporting_frame_count=len(supporting_frames),
            dominant_cluster_weight=cluster_weight,
            dominant_cluster_weight_share=cluster_share,
            mean_position_confidence=mean_position,
            minimum_position_confidence=minimum_position,
            supporting_evidence_ids=tuple(sorted(
                item.evidence.evidence_id for item in cluster
            )),
            rejected_evidence_ids=tuple(sorted(set(
                conflicting + list(state.rejected_ids)
            ))),
            character_diagnostics=characters,
            engine_contributions=tuple(contributions),
            reason_codes=tuple(sorted(set(reasons))),
            constraint_diagnostic=constraint_diagnostic,
        )

    def _empty_result(
        self,
        camera_id: str,
        track_id: int,
        state: _TrackAccumulator | None,
    ) -> FusedPlateResult:
        reasons = ["no_usable_evidence"]
        if state and state.invalid_count:
            reasons.append("invalid_evidence_ignored")
        return FusedPlateResult(
            status=FusionStatus.UNKNOWN,
            plate_text=None,
            confidence=0.0,
            camera_id=camera_id,
            track_id=track_id,
            total_evidence_count=state.total_seen if state else 0,
            usable_evidence_count=0,
            unique_frame_count=0,
            supporting_frame_count=0,
            dominant_cluster_weight=0.0,
            dominant_cluster_weight_share=0.0,
            mean_position_confidence=0.0,
            minimum_position_confidence=0.0,
            supporting_evidence_ids=(),
            rejected_evidence_ids=tuple(state.rejected_ids) if state else (),
            character_diagnostics=(),
            engine_contributions=(),
            reason_codes=tuple(reasons),
        )
