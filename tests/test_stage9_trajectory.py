from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta

import pytest

from backend.domain.cameras import CameraNode
from backend.domain.sightings import SightingStatus, VehicleSighting
from backend.trajectory.conversion import confirmed_sighting_from_anpr
from backend.trajectory.demo import build_demo_service
from backend.trajectory.models import CameraEdge, PlausibilityStatus, TrajectoryStatus
from backend.trajectory.reconstruction import CameraTopology, TrajectoryReconstructor
from backend.trajectory.repository import InMemorySightingRepository
from backend.trajectory.service import TrajectoryService
from cv.ocr.models import FusedPlateResult, FusionStatus
from cv.pipeline.models import CompletedTrackANPR, TrackTerminationReason

BASE = datetime(2026, 9, 7, 10, 0, tzinfo=UTC)
CAMERAS = {
    "A": CameraNode("A", "Alpha", 12.0, 77.0, direction="EAST"),
    "B": CameraNode("B", "Beta", 12.01, 77.01),
    "C": CameraNode("C", "Gamma", 12.02, 77.02),
}


def sighting(identifier, plate, camera, seconds, confidence=.9, track=1):
    node = CAMERAS[camera]
    return VehicleSighting(identifier, plate, camera, BASE + timedelta(seconds=seconds),
                           node.latitude, node.longitude, confidence, track, node.direction)


def service(*items, edges=()):
    repository = InMemorySightingRepository()
    for item in items:
        repository.add(item)
    reconstructor = TrajectoryReconstructor(repository, CAMERAS, CameraTopology(edges))
    return TrajectoryService(repository, reconstructor)


def completed(status=FusionStatus.ACCEPTED, text="KA02MN1826"):
    fused = FusedPlateResult(
        status, text, .93, "A", 7, 3, 3, 3, 3, 1.0, 1.0, .9, .8,
        (), (), (), (), (),
    )
    return CompletedTrackANPR("A", 7, TrackTerminationReason.TRACK_EXPIRED, 99,
                              BASE, 3, 3, (), fused)


@pytest.mark.parametrize("latitude,longitude", [(91, 0), (-91, 0), (0, 181), (0, -181)])
def test_camera_coordinate_validation(latitude, longitude):
    with pytest.raises(ValueError):
        CameraNode("A", "Camera", latitude, longitude)


def test_sighting_is_immutable():
    item = sighting("1", "P", "A", 0)
    with pytest.raises(FrozenInstanceError):
        item.camera_id = "B"


def test_accepted_anpr_converts_with_camera_metadata_and_stable_id():
    result = confirmed_sighting_from_anpr(completed(), CAMERAS["A"])
    assert result.plate_text == "KA02MN1826"
    assert result.timestamp == BASE and result.latitude == CAMERAS["A"].latitude
    assert result.plate_confidence == .93 and result.source_track_id == 7
    assert result == confirmed_sighting_from_anpr(completed(), CAMERAS["A"])


@pytest.mark.parametrize("status", [FusionStatus.UNKNOWN, FusionStatus.LOW_CONFIDENCE])
def test_unconfirmed_anpr_does_not_create_confirmed_sighting(status):
    assert confirmed_sighting_from_anpr(completed(status), CAMERAS["A"]) is None


def test_empty_accepted_plate_is_not_converted():
    assert confirmed_sighting_from_anpr(completed(text=None), CAMERAS["A"]) is None


def test_repository_add_query_duplicate_protection_and_ordering():
    repository = InMemorySightingRepository()
    later, earlier = sighting("2", "P", "B", 20), sighting("1", "P", "A", 10)
    assert repository.add(later) and repository.add(earlier)
    assert not repository.add(earlier)
    duplicate_event = VehicleSighting("different-id", earlier.plate_text, earlier.camera_id,
                                      earlier.timestamp, earlier.latitude, earlier.longitude,
                                      earlier.plate_confidence, earlier.source_track_id)
    assert not repository.add(duplicate_event)
    assert repository.get_by_plate("P") == (earlier, later)
    assert repository.get_by_camera("A") == (earlier,)


def test_low_confidence_repository_record_is_isolated_from_confirmed_queries():
    repository = InMemorySightingRepository()
    item = VehicleSighting("low", "P", "A", BASE, 12, 77, .4,
                           status=SightingStatus.LOW_CONFIDENCE)
    repository.add(item)
    assert repository.get_by_plate("P") == ()


def test_three_camera_trajectory_calculates_distance_time_and_speed():
    result = service(
        sighting("1", "P", "A", 0), sighting("2", "P", "B", 180),
        sighting("3", "P", "C", 420),
        edges=(CameraEdge("A", "B", 1500, 100, 300),
               CameraEdge("B", "C", 2000, 100, 400)),
    ).get_trajectory("P")
    assert [point.camera_id for point in result.points] == ["A", "B", "C"]
    assert result.total_distance_m == 3500 and result.total_duration_seconds == 420
    assert result.segments[0].estimated_speed_kmh == pytest.approx(30.0)
    assert result.average_speed_kmh == pytest.approx(30.0)
    assert result.camera_count == 3 and result.status is TrajectoryStatus.COMPLETE


def test_too_fast_transition_is_retained_and_marked():
    result = service(sighting("1", "P", "A", 0), sighting("2", "P", "B", 20),
                     edges=(CameraEdge("A", "B", 1000, 60, 300),)).get_trajectory("P")
    assert len(result.segments) == 1
    assert result.segments[0].plausibility_status is PlausibilityStatus.TOO_FAST
    assert result.status is TrajectoryStatus.HAS_IMPLAUSIBLE_TRANSITIONS


def test_no_topology_transition_is_retained_without_fabricated_distance():
    result = service(sighting("1", "P", "A", 0), sighting("2", "P", "C", 100)).get_trajectory("P")
    assert len(result.points) == 2 and len(result.segments) == 1
    assert result.segments[0].plausibility_status is PlausibilityStatus.NO_TOPOLOGY
    assert result.segments[0].distance_m is None


def test_unknown_plate_returns_empty_json_ready_trajectory():
    result = service().get_trajectory("UNKNOWN")
    assert result.status is TrajectoryStatus.EMPTY and result.points == ()
    assert result.to_dict()["plate"] == "UNKNOWN"


def test_time_range_filtering_and_camera_isolation():
    api = service(sighting("1", "P", "A", 0), sighting("2", "P", "B", 100),
                  sighting("3", "Q", "A", 200))
    result = api.get_trajectory("P", BASE + timedelta(seconds=50), BASE + timedelta(seconds=150))
    assert [point.camera_id for point in result.points] == ["B"]
    assert [item.plate_text for item in api.get_camera_history("A")] == ["P", "Q"]


def test_output_is_deterministic_for_insertion_order():
    items = [sighting("1", "P", "A", 0), sighting("2", "P", "B", 100)]
    edge = (CameraEdge("A", "B", 1000, 50, 200),)
    assert service(*items, edges=edge).get_trajectory("P").to_dict() == service(*reversed(items), edges=edge).get_trajectory("P").to_dict()


def test_demo_plate_reconstructs_synthetic_route():
    result = build_demo_service().get_trajectory("KA02MN1826")
    assert [point.camera_id for point in result.points] == ["CAM-01", "CAM-03", "CAM-05"]
    assert all(segment.plausibility_status is PlausibilityStatus.PLAUSIBLE for segment in result.segments)
    assert result.total_distance_m == 4200
