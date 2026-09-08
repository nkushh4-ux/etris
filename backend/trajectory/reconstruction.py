from datetime import datetime

from backend.domain.cameras import CameraNode
from backend.domain.sightings import VehicleSighting
from backend.trajectory.models import (
    CameraEdge,
    PlausibilityStatus,
    TrajectoryPoint,
    TrajectorySegment,
    TrajectoryStatus,
    VehicleTrajectory,
)
from backend.trajectory.repository import SightingRepository


class CameraTopology:
    def __init__(self, edges=()) -> None:
        self._edges: dict[tuple[str, str], CameraEdge] = {}
        for edge in edges:
            self.add(edge)

    def add(self, edge: CameraEdge) -> None:
        key = (edge.source_camera_id, edge.target_camera_id)
        if key in self._edges and self._edges[key] != edge:
            raise ValueError(f"conflicting topology edge: {key}")
        self._edges[key] = edge

    def get(self, source: str, target: str) -> CameraEdge | None:
        return self._edges.get((source, target))


class TrajectoryReconstructor:
    def __init__(self, repository: SightingRepository,
                 cameras: dict[str, CameraNode], topology: CameraTopology | None = None) -> None:
        self.repository = repository
        self.cameras = dict(cameras)
        self.topology = topology or CameraTopology()

    def reconstruct(self, plate_text: str, start_time: datetime | None = None,
                    end_time: datetime | None = None) -> VehicleTrajectory:
        sightings = self.repository.get_by_plate(plate_text)
        sightings = tuple(x for x in sightings if (start_time is None or x.timestamp >= start_time)
                          and (end_time is None or x.timestamp <= end_time))
        unique: list[VehicleSighting] = []
        seen = set()
        for item in sightings:
            if item.duplicate_key not in seen:
                seen.add(item.duplicate_key)
                unique.append(item)
        points = tuple(self._point(item) for item in unique)
        segments = tuple(self._segment(a, b) for a, b in zip(unique, unique[1:]))
        if not points:
            return VehicleTrajectory(plate_text, (), (), None, None, 0.0, 0.0, None, 0,
                                     TrajectoryStatus.EMPTY, ("NO_SIGHTINGS",))
        distance = sum(x.distance_m or 0.0 for x in segments)
        duration = (points[-1].timestamp - points[0].timestamp).total_seconds()
        statuses = {x.plausibility_status for x in segments}
        speed = (
            distance / duration * 3.6
            if duration > 0 and distance > 0
            and PlausibilityStatus.NO_TOPOLOGY not in statuses
            else None
        )
        if statuses & {PlausibilityStatus.TOO_FAST, PlausibilityStatus.TOO_SLOW}:
            status = TrajectoryStatus.HAS_IMPLAUSIBLE_TRANSITIONS
        elif PlausibilityStatus.NO_TOPOLOGY in statuses:
            status = TrajectoryStatus.INCOMPLETE_TOPOLOGY
        else:
            status = TrajectoryStatus.COMPLETE
        diagnostics = tuple(f"{x.source_camera_id}->{x.target_camera_id}:{x.plausibility_status.value}"
                            for x in segments if x.plausibility_status is not PlausibilityStatus.PLAUSIBLE)
        return VehicleTrajectory(plate_text, points, segments, points[0].timestamp,
                                 points[-1].timestamp, distance, duration, speed,
                                 len({x.camera_id for x in points}), status, diagnostics)

    def _point(self, sighting: VehicleSighting) -> TrajectoryPoint:
        camera = self.cameras.get(sighting.camera_id)
        return TrajectoryPoint(
            sighting.camera_id, camera.name if camera else sighting.camera_id,
            sighting.latitude, sighting.longitude, sighting.timestamp,
            sighting.plate_confidence, sighting.direction, sighting.sighting_id,
        )

    def _segment(self, first: VehicleSighting, second: VehicleSighting) -> TrajectorySegment:
        elapsed = (second.timestamp - first.timestamp).total_seconds()
        camera = self.cameras.get(first.camera_id)
        direction = first.direction or (camera.direction if camera else None)
        edge = self.topology.get(first.camera_id, second.camera_id)
        if edge is None:
            return TrajectorySegment(first.camera_id, second.camera_id, first.timestamp,
                                     second.timestamp, elapsed, None, None,
                                     PlausibilityStatus.NO_TOPOLOGY, direction)
        if elapsed < edge.min_travel_seconds:
            status = PlausibilityStatus.TOO_FAST
        elif elapsed > edge.max_travel_seconds:
            status = PlausibilityStatus.TOO_SLOW
        else:
            status = PlausibilityStatus.PLAUSIBLE
        speed = edge.distance_m / elapsed * 3.6 if elapsed > 0 else None
        return TrajectorySegment(first.camera_id, second.camera_id, first.timestamp,
                                 second.timestamp, elapsed, edge.distance_m, speed,
                                 status, direction)
