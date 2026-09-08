from collections import defaultdict
from datetime import UTC, datetime, timedelta
from statistics import mean, median

from backend.analytics.congestion import CongestionThresholds, classify_congestion
from backend.analytics.models import (
    CameraVolume,
    CongestionLevel,
    CongestionResult,
    HeatmapPoint,
    NetworkSummary,
    ODFlow,
    RouteDensity,
    SegmentAnalytics,
)
from backend.domain.cameras import CameraNode
from backend.domain.sightings import SightingStatus, VehicleSighting
from backend.trajectory.models import CameraEdge
from backend.trajectory.repository import SightingRepository


class TrafficAnalyticsService:
    """Deterministic, linear-time MVP analytics over the active Stage 9 repository."""

    def __init__(self, repository: SightingRepository, cameras: dict[str, CameraNode],
                 edges: tuple[CameraEdge, ...] = (), *,
                 thresholds: CongestionThresholds | None = None,
                 baseline_travel_seconds: dict[tuple[str, str], float] | None = None) -> None:
        self.repository = repository
        self.cameras = dict(cameras)
        self.edges = tuple(sorted(edges, key=lambda x: (x.source_camera_id, x.target_camera_id)))
        self._edge_map = {(x.source_camera_id, x.target_camera_id): x for x in self.edges}
        self.thresholds = thresholds or CongestionThresholds()
        self.baselines = dict(baseline_travel_seconds or {})

    @staticmethod
    def recent_window(hours: float = 24) -> tuple[datetime, datetime]:
        end = datetime.now(UTC)
        return end - timedelta(hours=hours), end

    @staticmethod
    def _validate_window(start: datetime, end: datetime) -> None:
        if end < start: raise ValueError("end_time must not precede start_time")

    def _context(self, start: datetime, end: datetime):
        self._validate_window(start, end)
        rows = tuple(x for x in self.repository.get_between(start, end)
                     if x.status is SightingStatus.CONFIRMED)
        groups: dict[str, list[VehicleSighting]] = defaultdict(list)
        for row in rows: groups[row.plate_text.strip().upper()].append(row)
        for values in groups.values(): values.sort(key=lambda x: (x.timestamp, x.camera_id, x.sighting_id))
        return rows, groups

    def camera_volume(self, start: datetime, end: datetime) -> tuple[CameraVolume, ...]:
        rows, _ = self._context(start, end)
        counts, plates = defaultdict(int), defaultdict(set)
        for row in rows:
            counts[row.camera_id] += 1; plates[row.camera_id].add(row.plate_text.strip().upper())
        hours = (end - start).total_seconds() / 3600
        camera_ids = sorted(set(self.cameras) | set(counts))
        return tuple(CameraVolume(cid, counts[cid], len(plates[cid]),
                    counts[cid] / hours if hours > 0 else 0.0, start, end) for cid in camera_ids)

    def od_flows(self, start: datetime, end: datetime) -> tuple[ODFlow, ...]:
        _, groups = self._context(start, end); counts = defaultdict(int)
        for rows in groups.values():
            origin, destination = rows[0].camera_id, rows[-1].camera_id
            if origin != destination: counts[(origin, destination)] += 1
        return tuple(ODFlow(a, b, count) for (a, b), count in sorted(counts.items()))

    def _valid_samples(self, groups):
        samples = defaultdict(list)
        for rows in groups.values():
            for first, second in zip(rows, rows[1:]):
                edge = self._edge_map.get((first.camera_id, second.camera_id))
                elapsed = (second.timestamp - first.timestamp).total_seconds()
                if edge and elapsed > 0 and edge.min_travel_seconds <= elapsed <= edge.max_travel_seconds:
                    samples[(first.camera_id, second.camera_id)].append(
                        (elapsed, edge.distance_m / elapsed * 3.6))
        return samples

    def segment_analytics(self, start: datetime, end: datetime) -> tuple[SegmentAnalytics, ...]:
        _, groups = self._context(start, end); samples = self._valid_samples(groups)
        result = []
        for key in sorted(samples):
            times = [x[0] for x in samples[key]]; speeds = [x[1] for x in samples[key]]
            result.append(SegmentAnalytics(*key, len(times), mean(times), median(times),
                                           mean(speeds), median(speeds)))
        return tuple(result)

    def congestion(self, start: datetime, end: datetime) -> tuple[CongestionResult, ...]:
        _, groups = self._context(start, end); samples = self._valid_samples(groups); result = []
        for edge in self.edges:
            key = (edge.source_camera_id, edge.target_camera_id)
            times = [x[0] for x in samples.get(key, ())]
            baseline = self.baselines.get(key, edge.min_travel_seconds)
            if baseline <= 0: raise ValueError(f"baseline travel time must be positive for {key}")
            observed = median(times) if times else None
            ratio = observed / baseline if observed is not None else None
            result.append(CongestionResult(*key, len(times), baseline, observed, ratio,
                                           classify_congestion(len(times), ratio, self.thresholds)))
        return tuple(result)

    def heatmap(self, start: datetime, end: datetime) -> tuple[HeatmapPoint, ...]:
        volumes = self.camera_volume(start, end); maximum = max((x.vehicle_count for x in volumes), default=0)
        result = []
        for volume in volumes:
            camera = self.cameras.get(volume.camera_id)
            if camera:
                result.append(HeatmapPoint(volume.camera_id, camera.latitude, camera.longitude,
                    volume.vehicle_count, volume.vehicle_count / maximum if maximum else 0.0))
        return tuple(result)

    def routes(self, start: datetime, end: datetime) -> tuple[RouteDensity, ...]:
        counts = {(x.source_camera, x.target_camera): x.vehicle_count
                  for x in self.segment_analytics(start, end)}
        maximum = max(counts.values(), default=0)
        return tuple(RouteDensity(edge.source_camera_id, edge.target_camera_id, counts.get(
            (edge.source_camera_id, edge.target_camera_id), 0), counts.get(
            (edge.source_camera_id, edge.target_camera_id), 0) / maximum if maximum else 0.0)
            for edge in self.edges)

    def summary(self, start: datetime, end: datetime) -> NetworkSummary:
        rows, groups = self._context(start, end); volumes = self.camera_volume(start, end)
        od = self.od_flows(start, end); segments = self.segment_analytics(start, end)
        congestion = self.congestion(start, end)
        busiest_camera = max(volumes, key=lambda x: (x.vehicle_count, x.camera_id), default=None)
        busiest_segment = max(segments, key=lambda x: (x.vehicle_count, x.source_camera, x.target_camera), default=None)
        speeds = [speed for values in self._valid_samples(groups).values() for _, speed in values]
        level_counts = {level: sum(x.classification is level for x in congestion) for level in CongestionLevel}
        return NetworkSummary(start, end, len(rows), len({x.plate_text.strip().upper() for x in rows}),
            sum(x.vehicle_count > 0 for x in volumes), sum(x.vehicle_count for x in od),
            busiest_camera.camera_id if busiest_camera and busiest_camera.vehicle_count else None,
            busiest_camera.vehicle_count if busiest_camera else 0,
            f"{busiest_segment.source_camera}->{busiest_segment.target_camera}" if busiest_segment else None,
            busiest_segment.vehicle_count if busiest_segment else 0,
            mean(speeds) if speeds else None, median(speeds) if speeds else None,
            level_counts[CongestionLevel.FREE_FLOW], level_counts[CongestionLevel.MODERATE],
            level_counts[CongestionLevel.CONGESTED], level_counts[CongestionLevel.SEVERE])
