from dataclasses import dataclass

import pytest

from scripts.calibrate_plate_width_ocr import (
    WIDTH_BANDS,
    select_representatives,
    width_band,
)


@pytest.mark.parametrize(
    ("width", "expected"),
    [(0, "<40"), (39, "<40"), (40, "40-49"), (49, "40-49"),
     (50, "50-59"), (59, "50-59"), (60, "60-69"), (69, "60-69"),
     (70, "70-79"), (79, "70-79"), (80, "80-99"), (99, "80-99"),
     (100, "100+"), (500, "100+")],
)
def test_width_band_boundaries(width, expected):
    assert width_band(width) == expected


@dataclass
class _Frame:
    frame_index: int


@dataclass
class _Detection:
    track_id: int
    frame: _Frame


@dataclass
class _Candidate:
    quality_score: float


@dataclass
class _Observation:
    track_id: int
    frame_index: int
    quality: float
    band: str = "70-79"

    @property
    def detection(self):
        return _Detection(self.track_id, _Frame(self.frame_index))

    @property
    def candidate(self):
        return _Candidate(self.quality)


def test_sampling_is_bounded_diverse_spaced_and_order_independent():
    items = [
        _Observation(1, 1, .4), _Observation(1, 2, .9), _Observation(1, 10, .8),
        _Observation(2, 3, .7), _Observation(2, 12, .6), _Observation(3, 4, .5),
    ]
    first = select_representatives(items, maximum=4, min_frame_gap=5)
    second = select_representatives(reversed(items), maximum=4, min_frame_gap=5)
    key = lambda values: [(x.track_id, x.frame_index, x.quality) for x in values]
    assert key(first) == key(second)
    assert len(first) == 4
    assert {item.track_id for item in first} == {1, 2, 3}
    assert all(item.band in WIDTH_BANDS for item in first)
