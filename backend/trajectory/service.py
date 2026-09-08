from datetime import datetime

from backend.trajectory.reconstruction import TrajectoryReconstructor
from backend.trajectory.repository import SightingRepository


class TrajectoryService:
    def __init__(self, repository: SightingRepository,
                 reconstructor: TrajectoryReconstructor) -> None:
        self.repository = repository
        self.reconstructor = reconstructor

    def get_trajectory(self, plate_text: str, start_time: datetime | None = None,
                       end_time: datetime | None = None):
        return self.reconstructor.reconstruct(plate_text, start_time, end_time)

    def get_recent_sightings(self, plate_text: str, limit: int = 20):
        if limit < 1:
            raise ValueError("limit must be positive")
        return self.repository.get_by_plate(plate_text)[-limit:]

    def get_camera_history(self, camera_id: str):
        return self.repository.get_by_camera(camera_id)

