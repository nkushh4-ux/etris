from datetime import UTC, datetime

from cv.ocr.fusion import MultiFrameOCRFuser, OCRFusionConfig
from cv.ocr.india import (
    IndiaConstraintConfig,
    IndiaPlateConstraintDecoder,
    IndiaPlateFormat,
    PrefixStatus,
    classify_india_plate,
)
from cv.ocr.models import FusionStatus, OCREvidence


def decoder(**overrides):
    values = {**IndiaConstraintConfig().__dict__, **overrides} if hasattr(IndiaConstraintConfig(), "__dict__") else overrides
    return IndiaPlateConstraintDecoder(IndiaConstraintConfig(**values)) if overrides else IndiaPlateConstraintDecoder()


def observations(*texts):
    return [(text, 1.0, index) for index, text in enumerate(texts)]


def test_supported_valid_prefix_resolves_invalid_consensus():
    result = decoder().decode("XA02MH7256", observations("XA02MH7256", "KA02MH7256", "KA02MH7256"))
    assert result.constrained_text == "KA02MH7256"
    assert result.changed_positions == (0,)
    assert result.constraint_applied


def test_invalid_prefix_is_structurally_standard_not_unknown():
    validation = classify_india_plate("XA02MH7256")
    assert validation.format is IndiaPlateFormat.STANDARD_STATE
    assert validation.prefix == "XA"
    assert validation.prefix_status is PrefixStatus.INVALID
    assert validation.is_structurally_valid is True
    assert validation.is_prefix_valid is False


def test_standard_like_ocr_confusion_cannot_bypass_invalid_prefix_guard():
    validation = classify_india_plate("XAO2MH7256")
    assert validation.format is IndiaPlateFormat.STANDARD_STATE
    assert validation.prefix_status is PrefixStatus.INVALID
    assert validation.is_structurally_valid is False
    fuser = MultiFrameOCRFuser(OCRFusionConfig(), decoder())
    fuser.add_evidence(evidence("XAO2MH7256", 1)); fuser.add_evidence(evidence("XAO2MH7256", 2))
    result = fuser.finalize_track("CAM", 1)
    assert result.plate_text == "XAO2MH7256"
    assert result.status is FusionStatus.LOW_CONFIDENCE


def test_weak_single_frame_alternative_does_not_override_visual_evidence():
    result = decoder().decode("XA02MH7256", [
        ("XA02MH7256", 1.0, 1), ("XA02MH7256", 1.0, 2), ("KA02MH7256", .1, 3)])
    assert result.constrained_text == "XA02MH7256"


def test_decoder_never_introduces_unobserved_character():
    result = decoder().decode("XA02MH7256", observations("XA02MH7256", "XA02MH7256"))
    assert result.constrained_text == "XA02MH7256" and result.changed_positions == ()


def test_common_current_prefixes_are_valid_and_unchanged():
    for text in ("KA02MN1826", "KA02MM9091", "DL01AB1234", "MH12XY9087"):
        validation = classify_india_plate(text)
        result = decoder().decode(text, observations(text, text))
        assert validation.format is IndiaPlateFormat.STANDARD_STATE
        assert validation.prefix_status is PrefixStatus.VALID_CURRENT
        assert result.constrained_text == text


def test_bh_series_does_not_receive_state_prefix_validation():
    validation = classify_india_plate("22BH1234AA")
    assert validation.format is IndiaPlateFormat.BH_SERIES
    assert validation.prefix_status is PrefixStatus.NOT_APPLICABLE


def test_legacy_prefix_is_valid_without_modernization():
    validation = classify_india_plate("OR02AB1234")
    result = decoder().decode("OR02AB1234", observations("OR02AB1234", "OR02AB1234"))
    assert validation.prefix_status is PrefixStatus.VALID_LEGACY
    assert result.constrained_text == "OR02AB1234"


def evidence(text, frame, confidence=.95):
    return OCREvidence("CAM", 1, frame, text, text, confidence, .9, "svtr", "1",
                       f"e{frame}", datetime.now(UTC))


def test_invalid_prefix_without_supported_alternative_cannot_be_accepted():
    fuser = MultiFrameOCRFuser(OCRFusionConfig(), decoder())
    fuser.add_evidence(evidence("XA02MH7256", 1))
    fuser.add_evidence(evidence("XA02MH7256", 2))
    result = fuser.finalize_track("CAM", 1)
    assert result.plate_text == "XA02MH7256"
    assert result.status is FusionStatus.LOW_CONFIDENCE
    assert "invalid_state_prefix_constraint_downgrade" in result.reason_codes


def test_fusion_can_select_only_a_supported_constrained_character():
    fuser = MultiFrameOCRFuser(OCRFusionConfig(), decoder())
    for frame in (1, 2): fuser.add_evidence(evidence("XA02MH7256", frame, .99))
    for frame in (3, 4): fuser.add_evidence(evidence("KA02MH7256", frame, .20))
    result = fuser.finalize_track("CAM", 1)
    assert result.plate_text == "KA02MH7256"
    assert result.status is FusionStatus.ACCEPTED, result.reason_codes
    assert result.constraint_diagnostic.raw_consensus_text == "XA02MH7256"
    assert result.constraint_diagnostic.changed_positions == (0,)


def test_constraint_failure_preserves_raw_fusion_result():
    class Broken:
        def decode(self, *_): raise RuntimeError("decoder unavailable")
    fuser = MultiFrameOCRFuser(OCRFusionConfig(), Broken())
    fuser.add_evidence(evidence("KA02MN1826", 1)); fuser.add_evidence(evidence("KA02MN1826", 2))
    result = fuser.finalize_track("CAM", 1)
    assert result.plate_text == "KA02MN1826"
    assert any(reason.startswith("india_constraint_failure:") for reason in result.reason_codes)


def test_valid_syntax_cannot_upgrade_bad_ocr_evidence():
    fuser = MultiFrameOCRFuser(OCRFusionConfig(), decoder())
    fuser.add_evidence(evidence("KA02MN1826", 1, .05))
    result = fuser.finalize_track("CAM", 1)
    assert result.status is FusionStatus.UNKNOWN


def test_every_invalid_standard_prefix_is_blocked_by_final_invariant():
    for text in ("XA02MH7256", "ZZ01AB1234", "QQ9A123"):
        fuser = MultiFrameOCRFuser(OCRFusionConfig(), decoder())
        fuser.add_evidence(evidence(text, 1)); fuser.add_evidence(evidence(text, 2))
        result = fuser.finalize_track("CAM", 1)
        validation = classify_india_plate(result.plate_text)
        assert validation.format is IndiaPlateFormat.STANDARD_STATE
        assert validation.prefix_status is PrefixStatus.INVALID
        assert result.status is not FusionStatus.ACCEPTED
