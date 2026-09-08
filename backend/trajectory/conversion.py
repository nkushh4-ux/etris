from hashlib import sha256

from backend.domain.cameras import CameraNode
from backend.domain.sightings import VehicleSighting
from cv.ocr.models import FusionStatus
from cv.pipeline.models import CompletedTrackANPR


def confirmed_sighting_from_anpr(
    completed: CompletedTrackANPR, camera: CameraNode
) -> VehicleSighting | None:
    """Convert only an accepted fused identity; never reinterpret OCR text."""
    fused = completed.fused_plate_result
    if fused.status is not FusionStatus.ACCEPTED or not fused.plate_text:
        return None
    if (
        completed.camera_id != camera.camera_id
        or fused.camera_id != camera.camera_id
        or fused.track_id != completed.track_id
    ):
        raise ValueError("completed track and camera metadata do not match")
    event = f"{camera.camera_id}|{completed.track_id}|{completed.timestamp.isoformat()}|{fused.plate_text}"
    constraint = getattr(fused, "constraint_diagnostic", None)
    source = "ANPR_FUSION_INDIA_CONSTRAINED" if constraint and constraint.constraint_applied else "ANPR_FUSION"
    return VehicleSighting(
        sighting_id=f"anpr-{sha256(event.encode('utf-8')).hexdigest()[:20]}",
        plate_text=fused.plate_text,
        camera_id=camera.camera_id,
        timestamp=completed.timestamp,
        latitude=camera.latitude,
        longitude=camera.longitude,
        plate_confidence=fused.confidence,
        source_track_id=completed.track_id,
        direction=camera.direction,
        source=source,
    )
