from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any


class FusionStatus(str, Enum):
    ACCEPTED = "accepted"
    LOW_CONFIDENCE = "low_confidence"
    UNKNOWN = "unknown"


class OCRResultStatus(str, Enum):
    SUCCESS = "success"
    REJECTED = "rejected"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class OCRResult:
    """One engine observation; this is evidence, not a final identity."""

    camera_id: str
    track_id: int
    frame_index: int
    raw_text: str
    normalized_text: str
    confidence: float
    engine: str
    engine_version: str
    evidence_id: str
    plate_quality_score: float
    detector_confidence: float
    bbox: tuple[float, float, float, float]
    timestamp: datetime | None = None
    status: OCRResultStatus = OCRResultStatus.SUCCESS
    rejected_reason: str | None = None


@dataclass(frozen=True, slots=True)
class OCREvidence:
    camera_id: str
    track_id: int
    frame_index: int
    raw_text: str
    normalized_text: str
    ocr_confidence: float
    plate_quality_score: float
    engine: str
    engine_version: str
    evidence_id: str
    timestamp: datetime | None = None


@dataclass(frozen=True, slots=True)
class CharacterDiagnostic:
    position: int
    character: str
    weighted_support: float
    total_compatible_weight: float
    position_confidence: float
    supporting_frame_count: int


@dataclass(frozen=True, slots=True)
class EngineContribution:
    engine: str
    evidence_count: int
    unique_frame_count: int
    total_weight: float
    eligible_for_independent_accept: bool


@dataclass(frozen=True, slots=True)
class FusedPlateResult:
    status: FusionStatus
    plate_text: str | None
    confidence: float
    camera_id: str
    track_id: int
    total_evidence_count: int
    usable_evidence_count: int
    unique_frame_count: int
    supporting_frame_count: int
    dominant_cluster_weight: float
    dominant_cluster_weight_share: float
    mean_position_confidence: float
    minimum_position_confidence: float
    supporting_evidence_ids: tuple[str, ...]
    rejected_evidence_ids: tuple[str, ...]
    character_diagnostics: tuple[CharacterDiagnostic, ...]
    engine_contributions: tuple[EngineContribution, ...]
    reason_codes: tuple[str, ...]
    constraint_diagnostic: Any | None = None
