from abc import ABC, abstractmethod

from cv.ocr.models import OCREvidence, OCRResult, OCRResultStatus
from cv.plate.models import PlateCandidate


class OCREngine(ABC):
    """Engine-agnostic interface for recognizing retained plate crops."""

    @abstractmethod
    def recognize(self, candidate: PlateCandidate) -> OCRResult:
        raise NotImplementedError


def result_to_evidence(result: OCRResult) -> OCREvidence | None:
    """Convert a successful, non-empty OCR result to lightweight evidence."""

    if result.status is not OCRResultStatus.SUCCESS or not result.normalized_text:
        return None
    return OCREvidence(
        camera_id=result.camera_id,
        track_id=result.track_id,
        frame_index=result.frame_index,
        timestamp=result.timestamp,
        raw_text=result.raw_text,
        normalized_text=result.normalized_text,
        ocr_confidence=result.confidence,
        plate_quality_score=result.plate_quality_score,
        engine=result.engine,
        engine_version=result.engine_version,
        evidence_id=result.evidence_id,
    )
