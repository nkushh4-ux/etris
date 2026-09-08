"""Manual Stage 12C restricted-zone validation using existing detector/tracker."""
import argparse
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

from backend.alerts.repository import InMemoryAlertRepository
from backend.alerts.service import AlertService
from cv.alerts.restricted_zone import (
    RestrictedZoneConfig,
    RestrictedZoneEngine,
    RestrictedZoneMode,
    RestrictedZoneObservation,
    RestrictedZonePolicy,
    RestrictedZoneState,
)
from cv.common.frame import FramePacket
from cv.traffic_state.detector import BMD45TrafficDetector
from cv.traffic_state.tracker import TrafficByteTracker


def load_config(path):
    data=json.loads(Path(path).read_text(encoding="utf-8"))
    zones=tuple(RestrictedZonePolicy(item["zone_id"],item["label"],tuple(map(tuple,item["polygon"])),
        item.get("enabled",True),RestrictedZoneMode(item.get("mode","DENY_BY_DEFAULT")),
        item.get("entry_confirm_seconds",1.),item.get("grace_seconds",1.5),
        tuple(item.get("allowed_vehicle_classes",())),tuple(item.get("allowed_plates",()))) for item in data["zones"])
    return RestrictedZoneConfig(data["camera_id"],zones)


def _polygon_points(zone, width, height):
    return np.asarray([(round(x * (width - 1)), round(y * (height - 1)))
        for x, y in zone.polygon], dtype=np.int32)


def annotate_frame(image, tracks, tracker, engine, config, *, hide_zones=False):
    """Draw current tracker/zone state without changing alert decisions."""
    height, width = image.shape[:2]
    if not hide_zones:
        for zone in config.zones:
            if zone.enabled:
                cv2.polylines(image, [_polygon_points(zone, width, height)], True, (0, 0, 255), 2)
                origin = tuple(_polygon_points(zone, width, height)[0])
                cv2.putText(image, zone.label, (int(origin[0]), max(18, int(origin[1]) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 2)
    for track in tracks:
        inside = any(engine.state(track.track_id, zone.zone_id) in
            (RestrictedZoneState.ENTRY_PENDING, RestrictedZoneState.VIOLATION)
            for zone in config.zones if zone.enabled)
        bbox = track.bbox
        color = (0, 0, 255) if inside else (80, 225, 160)
        cv2.rectangle(image, (int(bbox.x1), int(bbox.y1)),
            (int(bbox.x2), int(bbox.y2)), color, 3 if inside else 2)
        vehicle_class = tracker.display_class(track.track_id)
        label = (f"RESTRICTED ZONE | {vehicle_class} | Track {track.track_id}"
            if inside else f"{vehicle_class} | Track {track.track_id}")
        cv2.putText(image, label, (int(bbox.x1), max(18, int(bbox.y1) - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, .5, color, 2)
    return image


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--video",required=True)
    parser.add_argument("--config",default=str(ROOT/"configs/restricted_zone_demo.json")); parser.add_argument("--max-frames",type=int)
    parser.add_argument("--output",default=str(ROOT/"runs/stage12c_restricted_zone/restricted_zone_alerts.json")); parser.add_argument("--device",default="cuda")
    parser.add_argument("--write-video",action="store_true"); parser.add_argument("--hide-restricted-zones",action="store_true")
    args=parser.parse_args(); config=load_config(args.config); capture=cv2.VideoCapture(args.video)
    fps=float(capture.get(cv2.CAP_PROP_FPS)) or 30.; width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)); height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output=Path(args.output); output.parent.mkdir(parents=True,exist_ok=True)
    writer=(cv2.VideoWriter(str(output.parent/"restricted_zone_annotated.mp4"),cv2.VideoWriter_fourcc(*"mp4v"),fps,(width,height))
        if args.write_video else None)
    detector=BMD45TrafficDetector(ROOT/"weights/traffic/bmd45-yolov12s.pt",device=args.device,image_size=960)
    tracker=TrafficByteTracker(); engine=RestrictedZoneEngine(config); repository=InMemoryAlertRepository(); service=AlertService(repository); frame_index=0
    start=datetime(2026,1,1,tzinfo=UTC)
    while args.max_frames is None or frame_index<args.max_frames:
        ok,image=capture.read()
        if not ok: break
        packet=FramePacket(config.camera_id,frame_index,start+timedelta(seconds=frame_index/fps),image)
        tracks=tracker.update(detector.detect(packet)); observations=tuple(RestrictedZoneObservation(track.track_id,track.bbox,
            tracker.display_class(track.track_id),tracker.class_confidence(track.track_id)) for track in tracks)
        service.process_restricted_zone_events(engine.update(observations,source_time_s=frame_index/fps,width=image.shape[1],height=image.shape[0]))
        if writer:
            writer.write(annotate_frame(image,tracks,tracker,engine,config,hide_zones=args.hide_restricted_zones))
        frame_index+=1
    capture.release(); tracker.finalize_all()
    if writer: writer.release()
    output.write_text(json.dumps([alert.to_dict() for alert in repository.list()],indent=2),encoding="utf-8")
    print(f"frames_processed: {frame_index}\nalerts: {len(repository.list())}\noutput: {output}")


if __name__=="__main__": main()
