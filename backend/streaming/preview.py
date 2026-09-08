from __future__ import annotations

import queue
import threading
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

from cv.ocr.models import OCRResultStatus


@dataclass(frozen=True, slots=True)
class LivePlatePreview:
    camera_id: str
    track_id: int
    frame_id: int
    raw_text: str
    normalized_text: str
    ocr_confidence: float
    plate_quality: float
    timestamp: datetime
    observation_count: int
    status: str

    def __post_init__(self) -> None:
        if self.status not in {"PROVISIONAL", "LIVE_CONSENSUS"}:
            raise ValueError("preview status must remain presentation-only")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["timestamp"] = self.timestamp.isoformat()
        return payload


class LivePreviewManager:
    """Presentation-only OCR history, isolated from canonical fusion."""

    def __init__(self, min_frame_gap: int = 5) -> None:
        if min_frame_gap < 1:
            raise ValueError("min_frame_gap must be positive")
        self.min_frame_gap = min_frame_gap
        self._last_attempt: dict[tuple[str, int], int] = {}
        self._identities: dict[tuple[str, int], Counter[str]] = {}
        self._current: dict[tuple[str, int], LivePlatePreview] = {}
        self._lock = threading.RLock()

    def may_attempt(self, camera_id: str, track_id: int, frame_id: int) -> bool:
        with self._lock:
            key = (camera_id, track_id)
            previous = self._last_attempt.get(key)
            if previous is not None and frame_id - previous < self.min_frame_gap:
                return False
            self._last_attempt[key] = frame_id
            return True

    def observe(self, candidate, result) -> LivePlatePreview | None:
        if result.status is not OCRResultStatus.SUCCESS or not result.normalized_text:
            return None
        with self._lock:
            key = (candidate.camera_id, candidate.track_id)
            counts = self._identities.setdefault(key, Counter())
            counts[result.normalized_text] += 1
            count = counts[result.normalized_text]
            preview = LivePlatePreview(
                candidate.camera_id, candidate.track_id, candidate.frame_index,
                result.raw_text, result.normalized_text, result.confidence,
                candidate.quality_score, candidate.timestamp, count,
                "LIVE_CONSENSUS" if count >= 2 else "PROVISIONAL",
            )
            self._current[key] = preview
            return preview

    def get(self, camera_id: str, track_id: int) -> LivePlatePreview | None:
        with self._lock: return self._current.get((camera_id, track_id))

    def latest(self) -> LivePlatePreview | None:
        with self._lock:
            return max(self._current.values(), key=lambda item: (item.frame_id, item.track_id), default=None)

    def expire_except(self, camera_id: str, active_track_ids: set[int]) -> None:
        with self._lock:
            for key in tuple(self._current):
                if key[0] == camera_id and key[1] not in active_track_ids:
                    self._current.pop(key, None)
                    self._identities.pop(key, None)
                    self._last_attempt.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._last_attempt.clear(); self._identities.clear(); self._current.clear()


class PreviewingBestFrameSelector:
    """Observe retained candidates without altering selector or fuser state."""

    def __init__(self, delegate, engine, manager: LivePreviewManager,
                 on_preview: Callable[[LivePlatePreview, float], None], source_fps: float) -> None:
        self.delegate, self.engine, self.manager = delegate, engine, manager
        self.on_preview, self.source_fps = on_preview, source_fps

    def offer(self, candidate) -> bool:
        retained = self.delegate.offer(candidate)
        if retained and self.manager.may_attempt(candidate.camera_id, candidate.track_id, candidate.frame_index):
            started = __import__("time").perf_counter()
            result = self.engine.recognize(candidate)
            latency = __import__("time").perf_counter() - started
            preview = self.manager.observe(candidate, result)
            if preview is not None:
                self.on_preview(preview, latency)
        return retained

    def get(self, camera_id: str, track_id: int):
        return self.delegate.get(camera_id, track_id)

    def finalize(self, camera_id: str, track_id: int):
        return self.delegate.finalize(camera_id, track_id)


class AsyncPreviewingBestFrameSelector(PreviewingBestFrameSelector):
    """Drop-safe presentation OCR worker; canonical selector state stays synchronous."""

    def __init__(self, *args, queue_size: int = 2, profiler=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.profiler = profiler
        self._jobs: queue.Queue = queue.Queue(maxsize=queue_size)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="etris-preview-ocr", daemon=True)
        self._thread.start()

    def offer(self, candidate) -> bool:
        retained = self.delegate.offer(candidate)
        if retained and self.manager.may_attempt(candidate.camera_id, candidate.track_id, candidate.frame_index):
            try:
                self._jobs.put_nowait(candidate)
            except queue.Full:
                try:
                    self._jobs.get_nowait(); self._jobs.task_done()
                except queue.Empty:
                    pass
                try:
                    self._jobs.put_nowait(candidate)
                except queue.Full:
                    pass
        return retained

    @property
    def pending_jobs(self) -> int:
        return self._jobs.qsize()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                candidate = self._jobs.get(timeout=.1)
            except queue.Empty:
                continue
            started = time.perf_counter()
            try:
                result = self.engine.recognize(candidate)
                preview = self.manager.observe(candidate, result)
                if preview is not None:
                    self.on_preview(preview, time.perf_counter() - started)
            finally:
                if self.profiler is not None:
                    self.profiler.add("preview_ocr", (time.perf_counter() - started) * 1000)
                self._jobs.task_done()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)
