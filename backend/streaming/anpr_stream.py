from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2

from backend.config.loader import load_model_config
from backend.streaming.preview import (
    AsyncPreviewingBestFrameSelector,
    LivePlatePreview,
    LivePreviewManager,
    PreviewingBestFrameSelector,
)
from cv.common.frame import FramePacket
from cv.ocr.models import FusionStatus, OCRResultStatus


@dataclass(frozen=True, slots=True)
class ANPREvent:
    plate: str
    status: str
    confidence: float
    camera_id: str
    track_id: int
    timestamp: str
    frame_id: int
    video_time_s: float | None = None

    def to_dict(self) -> dict[str, Any]: return asdict(self)


@dataclass(frozen=True, slots=True)
class ANPRStreamStatus:
    running: bool
    frame: int
    fps: float
    source: str
    camera_id: str
    pipeline_status: str
    error: str | None = None
    playback_state: str = "PLAYING"
    current_frame: int = -1
    total_frames: int = 0
    current_time_s: float = 0.0
    duration_s: float = 0.0
    source_fps: float = 0.0
    playback_fps: float = 0.0
    inference_fps: float = 0.0
    latest_analyzed_frame: int = -1
    mode: str = "LIVE_ANALYSIS"
    backend_online: bool = True
    processing_rate: str = "AUTO"

    def to_dict(self) -> dict[str, Any]: return asdict(self)


@dataclass(frozen=True, slots=True)
class InferenceFrame:
    generation: int
    frame_id: int
    timestamp: datetime
    image: Any


@dataclass(frozen=True, slots=True)
class AnnotationSnapshot:
    frame_id: int
    tracks: tuple[tuple[int, tuple[int, int, int, int]], ...]
    plates: tuple[tuple[tuple[int, int, int, int], str], ...]


class LatestFrameMailbox:
    """One pending inference frame; a newer frame replaces stale work."""
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._item: InferenceFrame | None = None
        self._closed = False
        self.dropped_frames = 0

    @property
    def pending_count(self) -> int:
        with self._condition: return int(self._item is not None)

    def put_latest(self, item: InferenceFrame) -> None:
        with self._condition:
            if self._item is not None: self.dropped_frames += 1
            self._item = item
            self._condition.notify()

    def get(self, timeout: float = .25) -> InferenceFrame | None:
        with self._condition:
            self._condition.wait_for(lambda: self._item is not None or self._closed, timeout=timeout)
            item, self._item = self._item, None
            return item

    def clear(self) -> None:
        with self._condition: self._item = None

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()


def events_from_completed(completed) -> tuple[ANPREvent, ...]:
    events = []
    for result in completed.ocr_results:
        if result.status is OCRResultStatus.SUCCESS and result.normalized_text:
            events.append(ANPREvent(result.normalized_text, "PROVISIONAL", result.confidence,
                completed.camera_id, completed.track_id, completed.timestamp.isoformat(),
                completed.termination_frame_index))
    fused = completed.fused_plate_result
    plate = fused.plate_text if fused.status is FusionStatus.ACCEPTED and fused.plate_text else "UNKNOWN"
    events.append(ANPREvent(plate, fused.status.name, fused.confidence, completed.camera_id,
        completed.track_id, completed.timestamp.isoformat(), completed.termination_frame_index))
    return tuple(events)


def presentation_overlay_labels(labels) -> tuple[str, ...]:
    """Hide lifecycle UNKNOWN spam without changing the underlying events."""
    return tuple(label for label in labels if not label.startswith("UNKNOWN [UNKNOWN"))


class ObservedPlateDetector:
    def __init__(self, delegate) -> None:
        self.delegate, self.current = delegate, []
        self.calls = self.rois_processed = self.total_detections = 0
    def begin_frame(self) -> None: self.current = []
    def detect_in_frame(self, track, image, frame_packet):
        before_rois = int(getattr(self.delegate, "rois_processed", 0))
        detections = self.delegate.detect_in_frame(track, image, frame_packet)
        self.calls += 1
        self.rois_processed += int(getattr(self.delegate, "rois_processed", before_rois)) - before_rois
        self.total_detections += len(detections)
        self.current.extend(detections)
        return detections
    def detect_batch_in_frame(self, tracks, image, frame_packet):
        before = int(getattr(self.delegate, "model_invocations", 0))
        before_rois = int(getattr(self.delegate, "rois_processed", 0))
        detections = self.delegate.detect_batch_in_frame(tracks, image, frame_packet)
        self.calls += int(getattr(self.delegate, "model_invocations", before)) - before
        self.rois_processed += int(getattr(self.delegate, "rois_processed", before_rois)) - before_rois
        self.total_detections += len(detections)
        self.current.extend(detections)
        return detections


class ObservedQualityAssessor:
    def __init__(self, delegate) -> None:
        self.delegate, self.current = delegate, {}
        self.valid = self.rejected = 0
    def begin_frame(self) -> None: self.current = {}
    def assess(self, detection):
        candidate = self.delegate.assess(detection)
        if candidate is None: self.rejected += 1
        else: self.valid += 1
        self.current[(detection.track_id, detection.frame.frame_index)] = candidate.selection_tier if candidate else "REJECTED"
        return candidate


class ANPRStreamService:
    def __init__(self, video: str | Path, camera_id: str, pipeline_factory: Callable[[], Any], *,
                 event_limit: int = 100, loop: bool = False, max_frames: int | None = None,
                 max_annotation_age_frames: int = 6, completed_sink=None) -> None:
        if event_limit < 1: raise ValueError("event_limit must be positive")
        if max_annotation_age_frames < 0: raise ValueError("max_annotation_age_frames cannot be negative")
        self.video, self.camera_id, self.pipeline_factory = Path(video), camera_id, pipeline_factory
        self.loop, self.max_frames = loop, max_frames
        self.max_annotation_age_frames = max_annotation_age_frames
        self._events: deque[ANPREvent] = deque(maxlen=event_limit)
        self._condition, self._mailbox = threading.Condition(), LatestFrameMailbox()
        self._latest_jpeg: bytes | None = None
        self._sequence = 0
        self._running, self._playback_state = False, "PLAYING"
        self._frame = self._latest_analyzed_frame = -1
        self._total_frames = 0
        self._source_fps = self._playback_fps = self._inference_fps = 0.0
        self._pipeline_status, self._error = "IDLE", None
        self._generation, self._restart_requested, self._playback_ended = 0, False, False
        self._inference_drained = False
        self._annotation: AnnotationSnapshot | None = None
        self._stop = threading.Event()
        self._inference_ready = threading.Event()
        self._playback_thread = self._inference_thread = None
        self._started_once = False
        self._last_final_labels: deque[str] = deque(maxlen=4)
        self._output_path: Path | None = None
        self._preview_manager: LivePreviewManager | None = None
        self._current_recognition: dict[str, Any] | None = None
        self._preview_latencies: list[float] = []
        self._processing_rate = "AUTO"
        self._completed_sink = completed_sink
        self._active_pipeline = None
        self._completed_count = self._selected_candidate_count = self._canonical_ocr_count = 0
        self._constraint_diagnostics: list[dict[str, Any]] = []

    def status(self) -> ANPRStreamStatus:
        with self._condition:
            current = self._frame / self._source_fps if self._source_fps > 0 and self._frame >= 0 else 0.0
            duration = self._total_frames / self._source_fps if self._source_fps > 0 else 0.0
            return ANPRStreamStatus(self._running, self._frame, self._inference_fps, self.video.name,
                self.camera_id, self._pipeline_status, self._error, self._playback_state, self._frame,
                self._total_frames, current, duration, self._source_fps, self._playback_fps,
                self._inference_fps, self._latest_analyzed_frame,
                "LIVE_ANALYSIS", True, self._processing_rate)

    def recent_events(self) -> tuple[ANPREvent, ...]:
        with self._condition: return tuple(reversed(self._events))
    def add_event(self, event: ANPREvent) -> None:
        with self._condition: self._events.append(event)

    def current_recognition(self) -> dict[str, Any] | None:
        with self._condition:
            return dict(self._current_recognition) if self._current_recognition else None

    def preview_metrics(self) -> dict[str, Any]:
        with self._condition:
            values = tuple(self._preview_latencies)
        return {
            "count": len(values),
            "mean_latency_ms": (sum(values) / len(values) * 1000) if values else 0.0,
        }

    def set_processing_rate(self, rate: str) -> None:
        if rate not in {"AUTO", "1.0x", "0.5x"}:
            raise ValueError("processing rate must be AUTO, 1.0x, or 0.5x")
        with self._condition:
            self._processing_rate = rate


    def reset_state(self, *, clear_events: bool = True) -> None:
        with self._condition:
            self._latest_jpeg, self._annotation = None, None
            self._sequence += 1
            self._frame = self._latest_analyzed_frame = -1
            self._playback_fps = self._inference_fps = 0.0
            self._pipeline_status, self._error = "IDLE", None
            self._last_final_labels.clear(); self._mailbox.clear()
            self._preview_manager = None; self._current_recognition = None
            self._constraint_diagnostics.clear()
            self._preview_latencies.clear()
            if clear_events: self._events.clear()
            self._condition.notify_all()

    def start(self) -> None:
        with self._condition:
            if self._started_once: return
            self._started_once, self._running, self._pipeline_status = True, True, "LOADING_MODELS"
            self._stop.clear()
            self._inference_thread = threading.Thread(
                target=self._live_sequential_loop,
                name="etris-anpr-sequential",
                daemon=True,
            )
            self._inference_thread.start()

    def pause(self) -> None:
        with self._condition:
            if self._playback_state == "PLAYING":
                self._playback_state, self._pipeline_status = "PAUSED", "PAUSED"
                self._condition.notify_all()
    def resume(self) -> None:
        with self._condition:
            if self._playback_state in {"PAUSED", "ENDED"}:
                if self._playback_state == "ENDED": self._request_restart_locked()
                self._playback_state, self._pipeline_status = "PLAYING", "RUNNING"
                self._condition.notify_all()
    def restart(self) -> None:
        with self._condition:
            self._request_restart_locked()
            self._playback_state, self._pipeline_status = "PLAYING", "RUNNING"
            self._condition.notify_all()
    def _request_restart_locked(self) -> None:
        self._generation += 1; self._restart_requested = True; self._playback_ended = False
        self._inference_drained = False
        self._frame = self._latest_analyzed_frame = -1
        self._playback_fps = self._inference_fps = 0.0
        self._annotation = self._latest_jpeg = None
        self._last_final_labels.clear(); self._events.clear(); self._mailbox.clear(); self._sequence += 1
        self._constraint_diagnostics.clear()

    def stop(self) -> None:
        self._stop.set(); self._mailbox.close()
        with self._condition: self._condition.notify_all()
        for thread in (self._playback_thread, self._inference_thread):
            if thread and thread.is_alive() and thread is not threading.current_thread(): thread.join(timeout=15)

    def mjpeg(self) -> Iterator[bytes]:
        sequence = -1
        while not self._stop.is_set():
            with self._condition:
                self._condition.wait_for(lambda: self._sequence != sequence or self._stop.is_set(), timeout=2)
                if self._stop.is_set(): break
                sequence, jpeg = self._sequence, self._latest_jpeg
            if jpeg: yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"

    def run(self, *, display: bool = False, save_output: str | Path | None = None) -> None:
        self._output_path = Path(save_output) if save_output else None
        self.start()
        try:
            while not self._stop.is_set():
                status = self.status()
                if display and self._latest_jpeg:
                    import numpy as np
                    image = cv2.imdecode(np.frombuffer(self._latest_jpeg, dtype="uint8"), cv2.IMREAD_COLOR)
                    if image is not None: cv2.imshow("ETRIS Live ANPR", image)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"): self.stop(); break
                    if key == ord(" "): self.resume() if status.playback_state == "PAUSED" else self.pause()
                if status.playback_state == "ENDED" and not self.loop: break
                time.sleep(.02)
        finally:
            if display: cv2.destroyAllWindows()
            if not self.loop: self.stop()

    def _live_sequential_loop(self) -> None:
        """Run every source frame through the locked pipeline, in order."""
        writer = None
        try:
            while not self._stop.is_set():
                pipeline = self.pipeline_factory()
                with self._condition:
                    self._active_profiler = pipeline.profiler
                    self._active_pipeline = pipeline
                pipeline.plate_detector = ObservedPlateDetector(pipeline.plate_detector)
                pipeline.quality_assessor = ObservedQualityAssessor(pipeline.quality_assessor)
                capture = cv2.VideoCapture(str(self.video))
                if not capture.isOpened():
                    raise RuntimeError(f"Unable to open video: {self.video}")
                source_fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
                total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                if self.max_frames is not None:
                    total = min(total, self.max_frames) if total else self.max_frames
                with self._condition:
                    self._source_fps, self._total_frames = source_fps, total
                    self._pipeline_status = "RUNNING"
                preview_manager = LivePreviewManager(min_frame_gap=5)
                preview_async = bool(load_model_config().get("live_performance", {}).get("preview_async", True))
                preview_selector_type = AsyncPreviewingBestFrameSelector if preview_async else PreviewingBestFrameSelector
                pipeline.best_frame_selector = preview_selector_type(
                    pipeline.best_frame_selector,
                    pipeline.ocr_coordinator.engine,
                    preview_manager,
                    self._record_preview,
                    source_fps,
                    **({"profiler": pipeline.profiler} if preview_async else {}),
                )
                with self._condition:
                    self._preview_manager = preview_manager
                if self._output_path and writer is None:
                    self._output_path.parent.mkdir(parents=True, exist_ok=True)
                    size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                    writer = cv2.VideoWriter(str(self._output_path), cv2.VideoWriter_fourcc(*"mp4v"), source_fps, size)
                    if not writer.isOpened():
                        raise RuntimeError(f"Unable to open output video: {self._output_path}")
                frame_id, started = 0, time.perf_counter()
                restart_generation = self._generation
                while not self._stop.is_set():
                    with self._condition:
                        self._condition.wait_for(lambda: self._playback_state != "PAUSED" or self._stop.is_set())
                        if self._generation != restart_generation:
                            break
                    decode_started = time.perf_counter()
                    ok, image = capture.read()
                    pipeline.profiler.add("frame_decode", (time.perf_counter() - decode_started) * 1000)
                    if not ok or (self.max_frames is not None and frame_id >= self.max_frames):
                        break
                    frame_started = time.perf_counter()
                    plate_observer, quality_observer = pipeline.plate_detector, pipeline.quality_assessor
                    plate_observer.begin_frame(); quality_observer.begin_frame()
                    tracks = pipeline.process_frame(FramePacket(self.camera_id, frame_id, datetime.now(UTC), image))
                    preview_manager.expire_except(
                        self.camera_id, {track.track_id for track in tracks}
                    )
                    with self._condition:
                        latest = preview_manager.latest()
                        if latest is None and self._current_recognition and self._current_recognition.get("status") != "ACCEPTED":
                            self._current_recognition = None
                    for completed in pipeline.pop_completed_tracks():
                        self._record_completed(completed, restart_generation, source_fps)
                    elapsed = time.perf_counter() - started
                    inference_fps = (frame_id + 1) / elapsed if elapsed else 0.0
                    snapshot = self._snapshot(tracks, plate_observer.current, quality_observer.current, frame_id)
                    with self._condition:
                        labels = tuple(self._last_final_labels)
                        self._annotation = snapshot
                        self._latest_analyzed_frame = frame_id
                        self._inference_fps = inference_fps
                    annotation_started = time.perf_counter()
                    annotated = self._compose(image.copy(), snapshot, labels, frame_id, frame_id, inference_fps)
                    pipeline.profiler.add("annotation", (time.perf_counter() - annotation_started) * 1000)
                    jpeg_started = time.perf_counter()
                    self._publish(annotated, frame_id, inference_fps)
                    pipeline.profiler.add("jpeg_encoding", (time.perf_counter() - jpeg_started) * 1000)
                    if writer: writer.write(annotated)
                    frame_id += 1
                    with self._condition:
                        rate = self._processing_rate
                    if rate != "AUTO":
                        target_seconds = (1.0 if rate == "1.0x" else 2.0) / source_fps
                        remaining = target_seconds - (time.perf_counter() - frame_started)
                        if remaining > 0:
                            self._stop.wait(remaining)
                if hasattr(pipeline.best_frame_selector, "close"):
                    pipeline.best_frame_selector.close()
                for completed in pipeline.finalize():
                    self._record_completed(completed, restart_generation, source_fps)
                capture.release()
                with self._condition:
                    restarted = self._generation != restart_generation
                    self._restart_requested = False
                if restarted:
                    continue
                if self.loop and not self._stop.is_set():
                    self.restart()
                    continue
                with self._condition:
                    self._playback_state = self._pipeline_status = "ENDED"
                    self._condition.notify_all()
                break
        except Exception as exc:
            self._fail(exc)
        finally:
            if writer: writer.release()
            self._inference_ready.set()

    def _playback_loop(self) -> None:
        capture, writer = cv2.VideoCapture(str(self.video)), None
        try:
            if not capture.isOpened(): raise RuntimeError(f"Unable to open video: {self.video}")
            source_fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
            total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if self.max_frames is not None: total = min(total, self.max_frames) if total else self.max_frames
            with self._condition:
                self._source_fps, self._total_frames, self._pipeline_status = source_fps, total, "RUNNING"
            if self._output_path:
                self._output_path.parent.mkdir(parents=True, exist_ok=True)
                size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                writer = cv2.VideoWriter(str(self._output_path), cv2.VideoWriter_fourcc(*"mp4v"), source_fps, size)
                if not writer.isOpened(): raise RuntimeError(f"Unable to open output video: {self._output_path}")
            frame_id, deadline = 0, time.perf_counter()
            times: deque[float] = deque(maxlen=max(3, int(source_fps * 2)))
            while not self._stop.is_set():
                with self._condition:
                    self._condition.wait_for(lambda: self._playback_state != "PAUSED" or self._stop.is_set(), timeout=.25)
                    if self._restart_requested:
                        capture.set(cv2.CAP_PROP_POS_FRAMES, 0); frame_id = 0; deadline = time.perf_counter(); times.clear()
                        self._restart_requested = False
                    generation = self._generation
                if self._stop.is_set(): break
                ok, image = capture.read()
                if not ok or (self.max_frames is not None and frame_id >= self.max_frames):
                    with self._condition:
                        self._playback_ended = True; self._playback_state = self._pipeline_status = "ENDED"
                        self._inference_drained = False
                        self._condition.notify_all()
                    if self.loop:
                        with self._condition:
                            self._condition.wait_for(lambda: self._inference_drained or self._stop.is_set())
                        if not self._stop.is_set(): self.restart()
                        continue
                    break
                self._mailbox.put_latest(InferenceFrame(generation, frame_id, datetime.now(UTC), image.copy()))
                with self._condition:
                    snapshot, labels = self._annotation, tuple(self._last_final_labels)
                    analyzed, ai_fps = self._latest_analyzed_frame, self._inference_fps
                annotated = self._compose(image, snapshot, labels, frame_id, analyzed, ai_fps)
                now = time.perf_counter(); times.append(now)
                playback_fps = ((len(times)-1)/(times[-1]-times[0]) if len(times)>1 and times[-1]>times[0] else 0.0)
                self._publish(annotated, frame_id, playback_fps)
                if writer: writer.write(annotated)
                frame_id += 1; deadline += 1/source_fps
                delay = deadline - time.perf_counter()
                if delay > 0: self._stop.wait(delay)
                elif delay < -(2/source_fps): deadline = time.perf_counter()
        except Exception as exc: self._fail(exc)
        finally:
            capture.release()
            if writer: writer.release()

    def _inference_loop(self) -> None:
        pipeline, pipeline_generation, processed, started = None, -1, 0, 0.0
        try:
            pipeline = self.pipeline_factory()
            pipeline_generation = self._generation
            pipeline.plate_detector = ObservedPlateDetector(pipeline.plate_detector)
            pipeline.quality_assessor = ObservedQualityAssessor(pipeline.quality_assessor)
            started = time.perf_counter()
            self._inference_ready.set()
            while not self._stop.is_set():
                item = self._mailbox.get()
                with self._condition: generation, ended = self._generation, self._playback_ended
                if pipeline is not None and pipeline_generation != generation:
                    pipeline.finalize(); pipeline = None
                if item is None:
                    if ended and pipeline is not None:
                        for completed in pipeline.finalize(): self._record_completed(completed, pipeline_generation)
                        pipeline = None
                        with self._condition:
                            self._inference_drained = True
                            self._condition.notify_all()
                    elif ended:
                        with self._condition:
                            self._inference_drained = True
                            self._condition.notify_all()
                    continue
                if item.generation != generation: continue
                if pipeline is None:
                    pipeline = self.pipeline_factory(); pipeline_generation = item.generation
                    pipeline.plate_detector = ObservedPlateDetector(pipeline.plate_detector)
                    pipeline.quality_assessor = ObservedQualityAssessor(pipeline.quality_assessor)
                    processed, started = 0, time.perf_counter()
                plate_observer, quality_observer = pipeline.plate_detector, pipeline.quality_assessor
                plate_observer.begin_frame(); quality_observer.begin_frame()
                tracks = pipeline.process_frame(FramePacket(self.camera_id, item.frame_id, item.timestamp, item.image))
                for completed in pipeline.pop_completed_tracks(): self._record_completed(completed, item.generation)
                processed += 1; elapsed = time.perf_counter() - started
                snapshot = self._snapshot(tracks, plate_observer.current, quality_observer.current, item.frame_id)
                with self._condition:
                    if item.generation == self._generation:
                        self._annotation, self._latest_analyzed_frame = snapshot, item.frame_id
                        self._inference_fps = processed / elapsed if elapsed else 0.0
        except Exception as exc: self._fail(exc)
        finally: self._inference_ready.set()

    def _record_completed(self, item, generation: int | None = None,
                          source_fps: float | None = None) -> None:
        with self._condition:
            if generation is not None and generation != self._generation: return
            self._completed_count += 1
            self._selected_candidate_count += int(getattr(item, "selected_candidate_count", 0))
            self._canonical_ocr_count += int(
                getattr(item, "ocr_result_count", len(getattr(item, "ocr_results", ())))
            )
            diagnostic = getattr(item.fused_plate_result, "constraint_diagnostic", None)
            if diagnostic is not None:
                self._constraint_diagnostics.append({
                    "track_id": item.track_id,
                    "raw_evidence": [result.normalized_text for result in item.ocr_results],
                    "raw_consensus": diagnostic.raw_consensus_text,
                    "format": diagnostic.format.value,
                    "prefix_status": diagnostic.prefix_status.value,
                    "constrained_candidate": diagnostic.constrained_text,
                    "changed_positions": list(diagnostic.changed_positions),
                    "final_status": item.fused_plate_result.status.name,
                    "confidence": item.fused_plate_result.confidence,
                    "reason_codes": list(diagnostic.reason_codes),
                })
        if self._completed_sink is not None:
            try:
                self._completed_sink(item)
            except Exception:
                # Persistence is auxiliary and must never stop canonical ANPR.
                logging.getLogger(__name__).exception("Completed-track persistence submission failed")
        for original in events_from_completed(item):
            event = ANPREvent(
                original.plate, original.status, original.confidence,
                original.camera_id, original.track_id, original.timestamp,
                original.frame_id,
                original.frame_id / source_fps if source_fps else None,
            )
            self.add_event(event)
            label = (f"OCR: {event.plate} {event.confidence:.2f} [PROVISIONAL]" if event.status == "PROVISIONAL"
                     else f"{event.plate} [{event.status} {event.confidence:.2f}]")
            with self._condition:
                self._last_final_labels.append(label)
                if event.status == "ACCEPTED":
                    self._current_recognition = event.to_dict()

    def _record_preview(self, preview: LivePlatePreview, latency: float) -> None:
        event = ANPREvent(
            preview.normalized_text, preview.status, preview.ocr_confidence,
            preview.camera_id, preview.track_id, preview.timestamp.isoformat(),
            preview.frame_id,
            preview.frame_id / self._source_fps if self._source_fps else None,
        )
        with self._condition:
            self._events.append(event)
            self._preview_latencies.append(latency)
            self._current_recognition = event.to_dict()
            self._last_final_labels.append(
                f"{event.plate} [{event.status} {event.confidence:.2f}]"
            )

    def _fail(self, exc: Exception) -> None:
        with self._condition:
            self._error = f"{type(exc).__name__}: {exc}"; self._pipeline_status = "ERROR"
            self._playback_state, self._running = "ENDED", False; self._condition.notify_all()
        self._stop.set()

    def _publish(self, image, frame_id: int, playback_fps: float) -> None:
        started = time.perf_counter()
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok: raise RuntimeError("JPEG encoding failed")
        with self._condition:
            self._latest_jpeg = encoded.tobytes(); self._frame, self._playback_fps = frame_id, playback_fps
            self._sequence += 1; self._condition.notify_all()

    def timing_report(self) -> dict[str, Any]:
        # The reusable factory exposes the current pipeline indirectly; this is
        # populated by the active loop for CLI diagnostics.
        with self._condition:
            profiler = getattr(self, "_active_profiler", None)
        return profiler.report() if profiler is not None else {}

    def performance_counts(self) -> dict[str, Any]:
        """Canonical workload counters for calibration comparisons."""
        with self._condition:
            pipeline = self._active_pipeline
            plate = getattr(pipeline, "plate_detector", None)
            quality = getattr(pipeline, "quality_assessor", None)
            invocations = int(getattr(plate, "calls", 0))
            rois = int(getattr(plate, "rois_processed", 0))
            return {
                "completed_tracks": self._completed_count,
                "plate_detector_calls": invocations,
                "plate_detector_model_invocations": invocations,
                "plate_detector_rois_processed": rois,
                "mean_batch_size": rois / invocations if invocations else 0.0,
                "plate_detections": int(getattr(plate, "total_detections", 0)),
                "quality_valid": int(getattr(quality, "valid", 0)),
                "quality_rejected": int(getattr(quality, "rejected", 0)),
                "selected_candidates": self._selected_candidate_count,
                "canonical_ocr_results": self._canonical_ocr_count,
            }

    def constraint_diagnostics(self) -> tuple[dict[str, Any], ...]:
        with self._condition:
            return tuple(dict(item) for item in self._constraint_diagnostics)

    @staticmethod
    def _snapshot(tracks, plates, tiers, frame_id: int) -> AnnotationSnapshot:
        track_data = tuple((t.track_id, tuple(map(int, (t.bbox.x1,t.bbox.y1,t.bbox.x2,t.bbox.y2)))) for t in tracks)
        plate_data = tuple((tuple(map(int, (p.bbox.x1,p.bbox.y1,p.bbox.x2,p.bbox.y2))),
            tiers.get((p.track_id,p.frame.frame_index), "DETECTED")) for p in plates)
        return AnnotationSnapshot(frame_id, track_data, plate_data)

    def _compose(self, image, snapshot, labels, frame_id, analyzed_frame, ai_fps):
        if snapshot is not None and 0 <= frame_id-snapshot.frame_id <= self.max_annotation_age_frames:
            for track_id, (x1,y1,x2,y2) in snapshot.tracks:
                cv2.rectangle(image,(x1,y1),(x2,y2),(50,210,80),2)
                cv2.putText(image,f"Track {track_id}",(x1,max(18,y1-7)),cv2.FONT_HERSHEY_SIMPLEX,.55,(50,210,80),2,cv2.LINE_AA)
                preview = self._preview_manager.get(self.camera_id, track_id) if self._preview_manager else None
                if preview:
                    cv2.putText(image,f"{preview.normalized_text} [{preview.status} {preview.ocr_confidence:.0%}]",
                                (x1,min(image.shape[0]-8,y2+20)),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,255),2,cv2.LINE_AA)
            for (x1,y1,x2,y2), tier in snapshot.plates:
                color = (0,180,255) if tier == "FALLBACK" else (255,180,0)
                cv2.rectangle(image,(x1,y1),(x2,y2),color,2)
                cv2.putText(image,f"Plate {tier}",(x1,max(18,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,.45,color,1,cv2.LINE_AA)
        cv2.rectangle(image,(8,8),(380,116),(15,20,30),-1)
        rows=("ETRIS ANPR",f"VIDEO FRAME: {frame_id}",f"AI ANALYZED: {analyzed_frame}  AI FPS: {ai_fps:.1f}",f"Camera: {self.camera_id}")
        for row,text in enumerate(rows):
            cv2.putText(image,text,(18,31+row*25),cv2.FONT_HERSHEY_SIMPLEX,.58 if row==0 else .48,(255,255,255),1,cv2.LINE_AA)
        visible_labels = presentation_overlay_labels(labels)
        for row,label in enumerate(reversed(visible_labels)):
            cv2.putText(image,label,(18,image.shape[0]-18-row*24),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,255,255),2,cv2.LINE_AA)
        return image
