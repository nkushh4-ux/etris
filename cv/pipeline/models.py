from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from cv.ocr.models import FusedPlateResult, OCRResult


class TrackTerminationReason(str, Enum):
    TRACK_EXPIRED = "track_expired"
    END_OF_STREAM = "end_of_stream"


@dataclass(frozen=True, slots=True)
class CompletedTrackANPR:
    """Bounded terminal output for one camera-local vehicle track."""

    camera_id: str
    track_id: int
    termination_reason: TrackTerminationReason
    termination_frame_index: int
    timestamp: datetime
    selected_candidate_count: int
    ocr_result_count: int
    ocr_results: tuple[OCRResult, ...]
    fused_plate_result: FusedPlateResult
    finalization_errors: tuple[str, ...] = ()
