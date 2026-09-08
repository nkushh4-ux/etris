from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from backend.streaming.preview import (
    LivePlatePreview,
    LivePreviewManager,
    PreviewingBestFrameSelector,
)
from cv.ocr.models import OCRResultStatus


def candidate(frame=10, track=19, quality=.8):
    return SimpleNamespace(camera_id="CAM", track_id=track, frame_index=frame,
                           quality_score=quality, timestamp=datetime.now(UTC))


def result(text="KA02MN1826", confidence=.92):
    return SimpleNamespace(status=OCRResultStatus.SUCCESS, raw_text=text,
                           normalized_text=text, confidence=confidence)


def test_preview_is_presentation_only_and_cannot_be_accepted():
    preview = LivePlatePreview("CAM", 19, 10, "KA02MN1826", "KA02MN1826",
                               .92, .8, datetime.now(UTC), 1, "PROVISIONAL")
    assert preview.status == "PROVISIONAL"
    with pytest.raises(ValueError):
        LivePlatePreview("CAM", 19, 10, "X", "X", .9, .8,
                         datetime.now(UTC), 1, "ACCEPTED")


def test_preview_is_track_bound_and_expires():
    manager = LivePreviewManager()
    manager.observe(candidate(track=19), result())
    manager.observe(candidate(track=20), result("OTHER"))
    manager.expire_except("CAM", {20})
    assert manager.get("CAM", 19) is None
    assert manager.get("CAM", 20).normalized_text == "OTHER"


def test_preview_frame_gap_throttles_attempts():
    manager = LivePreviewManager(min_frame_gap=5)
    assert manager.may_attempt("CAM", 19, 10)
    assert not manager.may_attempt("CAM", 19, 14)
    assert manager.may_attempt("CAM", 19, 15)


def test_repeated_identity_becomes_consensus_but_disagreement_does_not():
    manager = LivePreviewManager()
    assert manager.observe(candidate(10), result()).status == "PROVISIONAL"
    assert manager.observe(candidate(15), result()).status == "LIVE_CONSENSUS"
    assert manager.observe(candidate(20), result("KA02HN1826")).status == "PROVISIONAL"


def test_selector_preview_does_not_touch_canonical_fuser():
    class Selector:
        def offer(self, item): return True
        def get(self, camera_id, track_id): return ()
        def finalize(self, camera_id, track_id): return ()
    class Engine:
        def recognize(self, item): return result()
    manager = LivePreviewManager()
    seen = []
    wrapper = PreviewingBestFrameSelector(Selector(), Engine(), manager,
                                           lambda preview, latency: seen.append(preview), 30)
    canonical_fuser_evidence = []
    wrapper.offer(candidate())
    assert len(seen) == 1
    assert canonical_fuser_evidence == []
