from __future__ import annotations

import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from statistics import mean, median


class StageTimingProfiler:
    """Lightweight wall-clock timings for the live pipeline."""

    def __init__(self) -> None:
        self._values: dict[str, list[float]] = defaultdict(list)
        self._lock = threading.Lock()

    @contextmanager
    def measure(self, stage: str):
        started = time.perf_counter()
        try:
            yield
        finally:
            self.add(stage, (time.perf_counter() - started) * 1000)

    def add(self, stage: str, milliseconds: float) -> None:
        with self._lock:
            self._values[stage].append(milliseconds)

    def report(self) -> dict[str, dict[str, float | int]]:
        with self._lock:
            values = {key: tuple(items) for key, items in self._values.items()}
        total = sum(sum(items) for items in values.values()) or 1.0
        result = {}
        for stage in sorted(values):
            items = sorted(values[stage])
            index = min(len(items) - 1, max(0, int(.95 * len(items)) - 1))
            result[stage] = {
                "count": len(items), "mean_ms": mean(items), "median_ms": median(items),
                "p95_ms": items[index], "percent": 100 * sum(items) / total,
            }
        return result
