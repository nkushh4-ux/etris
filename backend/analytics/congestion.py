from dataclasses import dataclass

from backend.analytics.models import CongestionLevel


@dataclass(frozen=True, slots=True)
class CongestionThresholds:
    minimum_samples: int = 3
    moderate_ratio: float = 1.25
    congested_ratio: float = 1.6
    severe_ratio: float = 2.2

    def __post_init__(self) -> None:
        if self.minimum_samples < 1:
            raise ValueError("minimum_samples must be positive")
        if not 1 <= self.moderate_ratio < self.congested_ratio < self.severe_ratio:
            raise ValueError("congestion ratios must be strictly increasing")


def classify_congestion(sample_count: int, ratio: float | None,
                        thresholds: CongestionThresholds) -> CongestionLevel:
    if sample_count < thresholds.minimum_samples or ratio is None:
        return CongestionLevel.INSUFFICIENT_DATA
    if ratio < thresholds.moderate_ratio: return CongestionLevel.FREE_FLOW
    if ratio < thresholds.congested_ratio: return CongestionLevel.MODERATE
    if ratio < thresholds.severe_ratio: return CongestionLevel.CONGESTED
    return CongestionLevel.SEVERE
