from dataclasses import dataclass
from enum import Enum


class IndiaPlateFormat(str, Enum):
    STANDARD_STATE = "STANDARD_STATE"
    BH_SERIES = "BH_SERIES"
    SPECIAL = "SPECIAL"
    UNKNOWN = "UNKNOWN"


class PrefixStatus(str, Enum):
    VALID_CURRENT = "VALID_CURRENT"
    VALID_LEGACY = "VALID_LEGACY"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    INVALID = "INVALID"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class IndiaPlateValidation:
    input_text: str
    format: IndiaPlateFormat
    prefix: str | None
    prefix_status: PrefixStatus
    is_structurally_valid: bool
    is_prefix_valid: bool | None
    reason_codes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class IndiaConstraintDiagnostic:
    raw_consensus_text: str
    constrained_text: str
    format: IndiaPlateFormat
    prefix: str | None
    prefix_status: PrefixStatus
    constraint_applied: bool
    changed_positions: tuple[int, ...]
    candidate_score: float
    raw_score: float
    score_margin: float
    reason_codes: tuple[str, ...]
