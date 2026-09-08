from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from cv.ocr.india.formats import classify_india_plate
from cv.ocr.india.models import (
    IndiaConstraintDiagnostic,
    IndiaPlateFormat,
    PrefixStatus,
)


@dataclass(frozen=True, slots=True)
class IndiaConstraintConfig:
    enabled: bool = True
    beam_width: int = 16
    max_character_changes: int = 2
    min_changed_character_support_frames: int = 2
    min_changed_character_support: float = .20
    invalid_standard_prefix_penalty: float = .35
    current_prefix_bonus: float = .10
    legacy_prefix_bonus: float = .04
    bh_format_bonus: float = .10
    min_constraint_score_margin: float = .04

    def __post_init__(self):
        if self.beam_width < 1 or self.max_character_changes < 0:
            raise ValueError("beam_width must be positive and max_character_changes non-negative")
        if self.min_changed_character_support_frames < 1:
            raise ValueError("changed-character frame support must be positive")
        if not 0 <= self.min_changed_character_support <= 1:
            raise ValueError("changed-character weighted support must be within [0, 1]")
        if self.min_constraint_score_margin < 0:
            raise ValueError("constraint score margin must be non-negative")

    @classmethod
    def from_mapping(cls, values: Mapping):
        return cls(**dict(values))


class IndiaPlateConstraintDecoder:
    def __init__(self, config: IndiaConstraintConfig | None = None):
        self.config = config or IndiaConstraintConfig()

    def decode(self, raw: str, observations: Iterable[tuple[str, float, int]]) -> IndiaConstraintDiagnostic:
        validation = classify_india_plate(raw)
        base_reason = []
        if not self.config.enabled:
            return self._diagnostic(raw, raw, validation, 0, 0, (), ("CONSTRAINTS_DISABLED",))
        rows = [(text, weight, frame) for text, weight, frame in observations if len(text) == len(raw)]
        total = sum(weight for _, weight, _ in rows)
        options = []
        for position, original in enumerate(raw):
            support, frames = {}, {}
            for text, weight, frame in rows:
                character = text[position]
                support[character] = support.get(character, 0.0) + weight
                frames.setdefault(character, set()).add(frame)
            choices = [original]
            for character in sorted(support):
                if character != original and len(frames[character]) >= self.config.min_changed_character_support_frames \
                        and total and support[character] / total >= self.config.min_changed_character_support:
                    choices.append(character)
            options.append((choices, support))

        def evidence_score(candidate):
            return sum(options[i][1].get(char, 0.0) / total for i, char in enumerate(candidate)) / len(candidate) if total else 0.0
        def prior(candidate):
            check = classify_india_plate(candidate)
            if check.format is IndiaPlateFormat.STANDARD_STATE:
                if check.prefix_status is PrefixStatus.VALID_CURRENT: return self.config.current_prefix_bonus
                if check.prefix_status is PrefixStatus.VALID_LEGACY: return self.config.legacy_prefix_bonus
                if check.prefix_status is PrefixStatus.INVALID: return -self.config.invalid_standard_prefix_penalty
            if check.format is IndiaPlateFormat.BH_SERIES: return self.config.bh_format_bonus
            return 0.0

        beam = [("", ())]
        for position, (choices, _support) in enumerate(options):
            expanded = []
            for prefix, changed in beam:
                for character in choices:
                    next_changed = changed + ((position,) if character != raw[position] else ())
                    if len(next_changed) <= self.config.max_character_changes:
                        expanded.append((prefix + character, next_changed))
            expanded.sort(key=lambda item: (
                -sum(options[i][1].get(char, 0.0) for i, char in enumerate(item[0])),
                len(item[1]), item[0]))
            beam = expanded[:self.config.beam_width]
        candidates = sorted(((evidence_score(candidate) + prior(candidate), candidate, changed)
                             for candidate, changed in beam),
                            key=lambda item: (-item[0], len(item[2]), item[1]))
        raw_score = evidence_score(raw) + prior(raw)
        best_score, best, changed = candidates[0] if candidates else (raw_score, raw, ())
        margin = best_score - raw_score
        if not changed or margin < self.config.min_constraint_score_margin:
            best_score, best, changed, margin = raw_score, raw, (), 0.0
        final_validation = classify_india_plate(best)
        if changed:
            base_reason.append("VALID_STATE_PREFIX_SUPPORTED_BY_MULTIFRAME_EVIDENCE")
        elif validation.format is IndiaPlateFormat.STANDARD_STATE and validation.prefix_status is PrefixStatus.INVALID:
            base_reason.append("INVALID_STATE_PREFIX_NO_SUPPORTED_ALTERNATIVE")
        else:
            base_reason.append("CONSTRAINT_EVALUATED_NO_CHANGE")
        return self._diagnostic(raw, best, final_validation, best_score, raw_score, changed, tuple(base_reason), margin)

    @staticmethod
    def _diagnostic(raw, final, validation, candidate_score, raw_score, changed, reasons, margin=0.0):
        return IndiaConstraintDiagnostic(raw, final, validation.format, validation.prefix,
            validation.prefix_status, bool(changed), tuple(changed), candidate_score, raw_score,
            margin, tuple(reasons))
