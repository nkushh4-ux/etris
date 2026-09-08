from cv.plate.models import PlateCandidate


def _rank(candidate: PlateCandidate) -> tuple[float, float, int]:
    return (
        -candidate.quality_score,
        -candidate.detector_confidence,
        candidate.frame_index,
    )


class PlateBestFrameSelector:
    """Maintain a bounded, deterministic top-K for each camera track."""

    def __init__(self, top_k: int = 3, min_frame_gap: int = 2) -> None:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if min_frame_gap < 0:
            raise ValueError("min_frame_gap must be non-negative")
        self.top_k = top_k
        self.min_frame_gap = min_frame_gap
        self._candidates: dict[
            tuple[str, int], dict[str, list[PlateCandidate]]
        ] = {}

    def _selected(self, key: tuple[str, int]) -> list[PlateCandidate]:
        tiers = self._candidates.get(key, {})
        primary = tiers.get("PRIMARY", [])[:self.top_k]
        remaining = self.top_k - len(primary)
        return primary + tiers.get("FALLBACK", [])[:remaining]

    def offer(self, candidate: PlateCandidate) -> bool:
        key = (candidate.camera_id, candidate.track_id)
        tiers = self._candidates.setdefault(
            key, {"PRIMARY": [], "FALLBACK": []}
        )
        tier = candidate.selection_tier
        if tier not in tiers:
            raise ValueError(f"Unknown plate candidate selection tier: {tier}")

        # A fallback never competes with a nearby primary. A newly arriving
        # primary supersedes nearby fallbacks regardless of quality.
        primary = tiers["PRIMARY"]
        fallback = tiers["FALLBACK"]
        if tier == "FALLBACK" and any(
            abs(item.frame_index - candidate.frame_index) <= self.min_frame_gap
            for item in primary
        ):
            return False
        if tier == "PRIMARY":
            fallback[:] = [
                item for item in fallback
                if abs(item.frame_index - candidate.frame_index) > self.min_frame_gap
            ]

        retained = tiers[tier]
        nearby = [
            item for item in retained
            if abs(item.frame_index - candidate.frame_index) <= self.min_frame_gap
        ]
        if nearby and any(_rank(item) <= _rank(candidate) for item in nearby):
            return False
        if nearby:
            nearby_ids = {id(item) for item in nearby}
            retained[:] = [item for item in retained if id(item) not in nearby_ids]

        retained.append(candidate)
        retained.sort(key=_rank)
        was_retained = any(item is candidate for item in self._selected(key))
        del retained[self.top_k:]
        return was_retained and any(item is candidate for item in self._selected(key))

    def get(self, camera_id: str, track_id: int) -> tuple[PlateCandidate, ...]:
        return tuple(self._selected((camera_id, track_id)))

    def finalize(self, camera_id: str, track_id: int) -> tuple[PlateCandidate, ...]:
        """Return final candidates and release all selector state for the track."""

        key = (camera_id, track_id)
        selected = tuple(self._selected(key))
        self._candidates.pop(key, None)
        return selected
