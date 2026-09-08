from datetime import UTC, datetime

import pytest

from cv.ocr.fusion import EnginePolicy, MultiFrameOCRFuser, OCRFusionConfig
from cv.ocr.models import FusionStatus, OCREvidence
from cv.ocr.normalization import normalize_plate_text


def evidence(
    text: str,
    frame: int,
    *,
    confidence: float = 0.9,
    quality: float = 0.9,
    engine: str = "primary",
    camera: str = "CAM-A",
    track: int = 1,
    evidence_id: str | None = None,
) -> OCREvidence:
    return OCREvidence(
        camera_id=camera,
        track_id=track,
        frame_index=frame,
        timestamp=datetime.now(UTC),
        raw_text=text,
        normalized_text=normalize_plate_text(text),
        ocr_confidence=confidence,
        plate_quality_score=quality,
        engine=engine,
        engine_version="1",
        evidence_id=evidence_id or f"{camera}-{track}-{frame}-{engine}-{text}",
    )


def fuse(*items: OCREvidence, config: OCRFusionConfig | None = None):
    fuser = MultiFrameOCRFuser(config)
    for item in items:
        fuser.add_evidence(item)
    return fuser.get(items[0].camera_id, items[0].track_id)


def test_perfect_exact_consensus_is_accepted():
    result = fuse(*(evidence("KA02MM9091", frame) for frame in (1, 2, 3)))
    assert result.status is FusionStatus.ACCEPTED
    assert result.plate_text == "KA02MM9091"
    assert result.supporting_frame_count == 3


def test_single_character_disagreement_uses_weighted_majority():
    result = fuse(
        evidence("KA02MM9091", 1),
        evidence("KA02MM9091", 2),
        evidence("KA02MM9094", 3),
    )
    assert result.status is FusionStatus.ACCEPTED
    assert result.plate_text == "KA02MM9091"


def test_h_n_confusion_uses_observed_weighted_consensus():
    result = fuse(
        evidence("KA02HN1826", 1),
        evidence("KA02MN1826", 2),
        evidence("KA02HN1826", 3),
    )
    assert result.status is FusionStatus.ACCEPTED
    assert result.plate_text == "KA02HN1826"


def test_conflicting_equal_clusters_are_not_accepted():
    result = fuse(
        evidence("KA02MM9091", 1),
        evidence("KA02MM9091", 2),
        evidence("DL01AB1234", 3),
        evidence("DL01AB1234", 4),
    )
    assert result.status is FusionStatus.LOW_CONFIDENCE
    assert result.dominant_cluster_weight_share == pytest.approx(0.5)


def test_one_frame_is_unknown():
    result = fuse(evidence("KA02MM9091", 100))
    assert result.status is FusionStatus.UNKNOWN
    assert result.plate_text is None


def test_multiple_engines_on_one_frame_count_as_one_frame():
    result = fuse(
        evidence("KA02MM9091", 100, engine="svtrv2"),
        evidence("KA02MM9091", 100, engine="paddle"),
    )
    assert result.unique_frame_count == 1
    assert result.supporting_frame_count == 1
    assert result.status is FusionStatus.UNKNOWN


def test_three_engines_on_one_frame_do_not_outvote_two_frames():
    result = fuse(
        evidence("KA02MM9091", 100, engine="one"),
        evidence("KA02MM9091", 100, engine="two"),
        evidence("KA02MM9091", 100, engine="three"),
        evidence("KA02MM9094", 101, engine="one"),
        evidence("KA02MM9094", 102, engine="one"),
    )
    assert result.plate_text == "KA02MM9094"
    assert result.character_diagnostics[-1].supporting_frame_count == 2


def test_agreeing_engines_share_one_physical_frame_budget():
    result = fuse(
        evidence("KA02MM9091", 100, engine="one"),
        evidence("KA02MM9091", 100, engine="two"),
        evidence("KA02MM9091", 100, engine="three"),
    )
    assert result.supporting_frame_count == 1
    assert result.dominant_cluster_weight == pytest.approx(0.9)


def test_same_frame_engine_provenance_is_preserved_after_aggregation():
    items = [
        evidence("KA02MM9091", 100, engine="one", evidence_id="e1"),
        evidence("KA02MM9091", 100, engine="two", evidence_id="e2"),
        evidence("KA02MM9091", 100, engine="three", evidence_id="e3"),
    ]
    result = fuse(*items)
    assert result.supporting_evidence_ids == ("e1", "e2", "e3")
    assert {item.engine for item in result.engine_contributions} == {
        "one", "two", "three"
    }
    assert sum(
        item.total_weight for item in result.engine_contributions
    ) == pytest.approx(0.9)


def test_per_frame_aggregation_is_deterministic():
    items = [
        evidence("KA02MM9091", 100, engine="one"),
        evidence("KA02MM9094", 100, engine="two"),
        evidence("KA02MM9094", 101, engine="one"),
        evidence("KA02MM9094", 102, engine="one"),
    ]
    assert fuse(*items) == fuse(*reversed(items))


def test_weak_matching_evidence_fails_final_confidence():
    result = fuse(
        evidence("KA02MM9091", 1, confidence=0.01, quality=0.01),
        evidence("KA02MM9091", 2, confidence=0.01, quality=0.01),
    )
    assert result.status is FusionStatus.LOW_CONFIDENCE
    assert "final_confidence_below_threshold" in result.reason_codes


def test_no_speculative_character_substitution():
    result = fuse(
        evidence("KAO2MM909I", 1),
        evidence("KAO2MM909I", 2),
    )
    assert result.status is FusionStatus.ACCEPTED
    assert result.plate_text == "KAO2MM909I"
    assert result.plate_text != "KA02MM9091"


def test_duplicate_observation_does_not_double_support():
    fuser = MultiFrameOCRFuser()
    first = evidence("KA02MM9091", 1, evidence_id="first")
    duplicate = evidence("KA02MM9091", 1, evidence_id="duplicate")
    assert fuser.add_evidence(first)
    assert not fuser.add_evidence(duplicate)
    result = fuser.get("CAM-A", 1)
    assert result.usable_evidence_count == 1
    assert result.unique_frame_count == 1
    assert "duplicate_evidence_ignored" in result.reason_codes


def test_result_is_deterministic_for_input_order():
    items = [
        evidence("KA02MM9091", 1),
        evidence("KA02MM9094", 3),
        evidence("KA02MM9091", 2),
    ]
    forward = fuse(*items)
    reverse = fuse(*reversed(items))
    assert forward == reverse


def test_camera_and_track_state_are_isolated():
    fuser = MultiFrameOCRFuser()
    for camera, track, text in (
        ("A", 1, "AA11AA1111"),
        ("A", 2, "BB22BB2222"),
        ("B", 1, "CC33CC3333"),
    ):
        fuser.add_evidence(evidence(text, 1, camera=camera, track=track))
        fuser.add_evidence(evidence(text, 2, camera=camera, track=track))
    assert fuser.get("A", 1).plate_text == "AA11AA1111"
    assert fuser.get("A", 2).plate_text == "BB22BB2222"
    assert fuser.get("B", 1).plate_text == "CC33CC3333"


def test_finalize_returns_result_and_releases_state():
    fuser = MultiFrameOCRFuser()
    fuser.add_evidence(evidence("KA02MM9091", 1))
    fuser.add_evidence(evidence("KA02MM9091", 2))
    assert fuser.finalize_track("CAM-A", 1).status is FusionStatus.ACCEPTED
    assert fuser.get("CAM-A", 1).status is FusionStatus.UNKNOWN
    assert ("CAM-A", 1) not in fuser._tracks


def test_state_is_bounded_and_preserves_frame_diversity():
    config = OCRFusionConfig(max_evidence_per_track=3)
    fuser = MultiFrameOCRFuser(config)
    for frame in range(10):
        fuser.add_evidence(evidence("KA02MM9091", frame))
    state = fuser._tracks[("CAM-A", 1)]
    assert len(state.evidence) == 3
    assert len({item.evidence.frame_index for item in state.evidence}) == 3
    assert "evidence_evicted_by_bound" in fuser.get("CAM-A", 1).reason_codes


@pytest.mark.parametrize("text", ["", " - / "])
def test_empty_normalized_reads_cannot_produce_identity(text):
    result = fuse(evidence(text, 1), evidence(text, 2))
    assert result.status is FusionStatus.UNKNOWN
    assert result.plate_text is None


def test_restricted_secondary_engine_cannot_establish_acceptance():
    config = OCRFusionConfig(engine_policies={
        "secondary": EnginePolicy(eligible_for_independent_accept=False)
    })
    result = fuse(
        evidence("KA02MM9091", 1, engine="secondary"),
        evidence("KA02MM9091", 2, engine="secondary"),
        config=config,
    )
    assert result.status is FusionStatus.LOW_CONFIDENCE
    assert "no_independently_eligible_engine" in result.reason_codes


def test_insertions_and_deletions_align_without_inventing_characters():
    result = fuse(
        evidence("KA02MM9091", 1),
        evidence("KAO2MM9091", 2),
        evidence("KA02MM9091", 3),
    )
    assert result.plate_text == "KA02MM9091"
    assert all(
        diagnostic.character in {item.raw_text[diagnostic.position] for item in (
            evidence("KA02MM9091", 1), evidence("KA02MM9091", 3)
        )}
        for diagnostic in result.character_diagnostics
    )


def test_configuration_validation_and_mapping():
    with pytest.raises(ValueError):
        OCRFusionConfig(weight_ocr_confidence=0.8, weight_plate_quality=0.4)
    with pytest.raises(ValueError):
        OCRFusionConfig(max_evidence_per_track=1, min_unique_frames_for_accept=2)
    config = OCRFusionConfig.from_mapping({
        "enabled": True,
        "engine_policies": {
            "secondary": {"eligible_for_independent_accept": False}
        },
    })
    assert not config.engine_policies["secondary"].eligible_for_independent_accept
