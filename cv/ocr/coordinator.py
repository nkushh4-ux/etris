from collections.abc import Iterable

from cv.ocr.engine import OCREngine, result_to_evidence
from cv.ocr.fusion import MultiFrameOCRFuser
from cv.ocr.models import FusedPlateResult, OCRResult
from cv.plate.models import PlateCandidate


class TrackOCRCoordinator:
    """Connect finalized Stage 3 crops to recognition and Stage 5 fusion."""

    def __init__(self, engine: OCREngine, fuser: MultiFrameOCRFuser) -> None:
        self.engine = engine
        self.fuser = fuser

    def add_candidate(self, candidate: PlateCandidate) -> OCRResult:
        result = self.engine.recognize(candidate)
        evidence = result_to_evidence(result)
        if evidence is not None:
            self.fuser.add_evidence(evidence)
        return result

    def add_candidates(self, candidates: Iterable[PlateCandidate]) -> tuple[OCRResult, ...]:
        return tuple(self.add_candidate(candidate) for candidate in candidates)

    def finalize_track(self, camera_id: str, track_id: int) -> FusedPlateResult:
        return self.fuser.finalize_track(camera_id, track_id)
