"""Preprocess real video into image-space traffic state and signal recommendations."""
import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT=Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0,str(PROJECT_ROOT))

from backend.signal_control.controller import AdaptiveSignalController
from backend.signal_control.vision_adapter import approach_states_from_video
from cv.common.frame import FramePacket
from cv.traffic_state.config import load_traffic_state_config
from cv.traffic_state.extractor import TrafficStateExtractor
from cv.traffic_state.tracker import TrafficByteTracker, TrafficStateTracker


def args():
    parser = argparse.ArgumentParser(
        description="ETRIS real-video traffic-state demo (no ANPR/OCR)"
    )

    parser.add_argument("--video", required=True)
    parser.add_argument("--config", required=True)

    parser.add_argument("--output-dir", default="runs/signal_demo")
    parser.add_argument("--max-frames", type=int)

    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--write-video", action="store_true")

    parser.add_argument(
        "--write-json",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument(
        "--calibration-preview",
        action="store_true",
        help="Draw configured ROIs on the first frame and exit without loading YOLO",
    )

    parser.add_argument("--device", default="cuda")

    parser.add_argument(
        "--detector",
        choices=("coco", "bmd45"),
        default="bmd45",
    )

    parser.add_argument(
        "--tracker",
        choices=("legacy", "bytetrack"),
        default="bytetrack",
    )

    parser.add_argument("--imgsz", type=int, default=960)

    # NEW:
    # Hides approach polygons / queue polygons / entry lines ONLY
    # from the rendered video.
    # Internal traffic-state calculations still use them.
    parser.add_argument(
        "--hide-approach-regions",
        action="store_true",
        help=(
            "Hide approach polygons, queue polygons, entry lines and "
            "approach labels from annotated video while keeping "
            "all traffic-state calculations active."
        ),
    )

    return parser.parse_args()


def points(polygon,width,height): return np.asarray([[round(x*(width-1)),round(y*(height-1))] for x,y in polygon],np.int32)


def draw_regions(image, config, hide=False):
    """
    Draw approach/queue/entry-line geometry for visualization only.

    Important:
    Setting hide=True does NOT disable the approach geometry used by the
    traffic-state extractor. It only removes the visual overlays.
    """

    if hide:
        return image

    h, w = image.shape[:2]

    for index, region in enumerate(config.approaches):
        color = (80, 225, 160) if index % 2 == 0 else (255, 190, 80)

        traffic_polygon = points(
            region.traffic_polygon,
            w,
            h,
        )

        queue_polygon = points(
            region.queue_polygon,
            w,
            h,
        )

        entry_line = points(
            region.entry_line,
            w,
            h,
        )

        # Approach polygon
        cv2.polylines(
            image,
            [traffic_polygon],
            True,
            color,
            2,
        )

        # Queue polygon
        cv2.polylines(
            image,
            [queue_polygon],
            True,
            (50, 180, 255),
            2,
        )

        # Entry line
        cv2.line(
            image,
            tuple(entry_line[0]),
            tuple(entry_line[1]),
            (255, 255, 70),
            2,
        )

        # Approach label
        anchor = tuple(
            points(
                (region.traffic_polygon[0],),
                w,
                h,
            )[0]
        )

        cv2.putText(
            image,
            f"{region.approach_id} / {region.label}",
            anchor,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )

    return image


def main():
    options=args(); video=Path(options.video); config=load_traffic_state_config(options.config)
    if not video.is_file(): raise FileNotFoundError(f"Video not found: {video}")
    output=Path(options.output_dir); output.mkdir(parents=True,exist_ok=True)
    capture=cv2.VideoCapture(str(video)); fps=float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    width,height=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)); ok,first=capture.read()
    if not ok: raise RuntimeError(f"Unable to read video: {video}")
    if options.calibration_preview:
        draw_regions(first,config); path=output/"roi_calibration_preview.jpg"; cv2.imwrite(str(path),first)
        print(f"ROI preview: {path}\nNormalized config: {options.config}"); return
    capture.set(cv2.CAP_PROP_POS_FRAMES,0)
    if options.detector=="bmd45":
        from cv.traffic_state.detector import BMD45TrafficDetector
        detector=BMD45TrafficDetector(PROJECT_ROOT/"weights/traffic/bmd45-yolov12s.pt",device=options.device,image_size=options.imgsz)
    else:
        from cv.detection.ultralytics_detector import UltralyticsVehicleDetector
        detector=UltralyticsVehicleDetector(PROJECT_ROOT/"weights/yolo26n.pt",device=options.device)
    tracker=TrafficByteTracker() if options.tracker=="bytetrack" else TrafficStateTracker()
    extractor=TrafficStateExtractor(config); controller=AdaptiveSignalController()
    writer=None
    if options.write_video:
        writer=cv2.VideoWriter(str(output/"traffic_signal_annotated.mp4"),cv2.VideoWriter_fourcc(*"mp4v"),fps,(width,height))
    events=[]; stats=defaultdict(lambda:defaultdict(list)); unique=defaultdict(set); crossings=defaultdict(set)
    last_plan=None; last_phase_order=None; phase_order_changes=0; start_clock=time.perf_counter(); frame_index=0; base=datetime(2026,1,1,tzinfo=UTC)
    while options.max_frames is None or frame_index<options.max_frames:
        ok,image=capture.read()
        if not ok: break
        video_time=frame_index/fps; timestamp=base+timedelta(seconds=video_time)
        packet=FramePacket("SIGNAL-DEMO-CAMERA",frame_index,timestamp,image)
        detections=detector.detect(packet); tracks=tracker.update(detections); state=extractor.update(tracks,frame_index=frame_index,
            video_time_s=video_time,width=width,height=height)
        if state.controller_update_due:
            inputs=approach_states_from_video(state,config.intersection_id,timestamp)
            last_plan=controller.plan(config.intersection_id,inputs,generated_at=timestamp)
            current_order=tuple(x.approach_id for x in last_plan.ordered_phases)
            if last_phase_order is not None and current_order!=last_phase_order: phase_order_changes+=1
            last_phase_order=current_order
            decisions={x.approach_id:x for x in last_plan.ordered_phases}
            events.append({"source":"REAL_VIDEO_ESTIMATE","video_time_s":video_time,"frame_index":frame_index,
                "approaches":[{"approach_id":x.approach_id,"label":x.label,"vehicle_count":x.active_vehicle_count,
                "queue_length":x.queue_length,"occupancy":x.image_space_occupancy,"arrival_rate_vpm":x.arrival_rate_vpm,
                "average_waiting_time_s":x.average_waiting_time_s,"heavy_vehicle_count":x.heavy_vehicle_count,
                "queued_track_ids":list(x.queued_track_ids),"controller":decisions[x.approach_id].to_dict()} for x in state.approaches],
                "phase_order":[x.approach_id for x in last_plan.ordered_phases],"cycle_duration_s":last_plan.cycle_duration_s,
                "limitations":"Image-space occupancy and slow/stationary congestion estimate; not verified physical lane occupancy or red-signal queue length."})
            for item in state.approaches:
                decision=decisions[item.approach_id]; stats[item.approach_id]["pressure"].append(decision.pressure)
                stats[item.approach_id]["green"].append(decision.green_duration_s)
        for item in state.approaches:
            s=stats[item.approach_id]; s["active"].append(item.active_vehicle_count); s["occupancy"].append(item.image_space_occupancy)
            s["queue"].append(item.queue_length); s["wait"].append(item.average_waiting_time_s); s["heavy"].append(item.heavy_vehicle_count); s["arrival"].append(item.arrival_rate_vpm)
            unique[item.approach_id].update(item.active_track_ids)
        draw_regions(image,config,hide=options.hide_approach_regions,)
        decision_map={x.approach_id:x for x in last_plan.ordered_phases} if last_plan else {}
        for track in tracks:
            queued=any(track.track_id in item.queued_track_ids for item in state.approaches); b=track.bbox
            cv2.rectangle(image,(int(b.x1),int(b.y1)),(int(b.x2),int(b.y2)),(0,80,255) if queued else (80,225,160),2)
            label=tracker.display_class(track.track_id) if hasattr(tracker,"display_class") else track.class_name
            cv2.putText(image,f"#{track.track_id} {label} {track.detection_confidence:.2f}",(int(b.x1),max(15,int(b.y1)-4)),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
        for index,item in enumerate(state.approaches):
            d=decision_map.get(item.approach_id); text=f"{item.approach_id} V:{item.active_vehicle_count} Q:{item.queue_length} Occ:{item.image_space_occupancy:.0%} Arr:{item.arrival_rate_vpm:.1f}/m Wait:{item.average_waiting_time_s:.1f}s"
            if d: text+=f" P:{d.pressure:.3f} Green:{d.green_duration_s}s"
            cv2.rectangle(image,(10,130+index*32),(min(width-10,900),157+index*32),(5,15,12),-1); cv2.putText(image,text,(18,150+index*32),cv2.FONT_HERSHEY_SIMPLEX,.48,(230,245,240),1)
        if writer: writer.write(image)
        if not options.no_display:
            cv2.imshow("ETRIS Traffic State Estimate",image)
            if cv2.waitKey(1)&0xFF==ord("q"): break
        frame_index+=1
    elapsed=time.perf_counter()-start_clock; capture.release(); tracking=tracker.diagnostics(); tracker.finalize_all()
    if writer: writer.release()
    cv2.destroyAllWindows()
    if options.write_json: (output/"traffic_state_events.json").write_text(json.dumps(events,indent=2),encoding="utf-8")
    print(f"video: {video}\nresolution: {width}x{height}\nsource_fps: {fps:.3f}\nframes_processed: {frame_index}\nvideo_duration_processed_s: {frame_index/fps:.3f}\nprocessing_fps: {frame_index/elapsed if elapsed else 0:.3f}")
    for approach in config.approaches:
        s=stats[approach.approach_id]; avg=lambda key:sum(s[key])/len(s[key]) if s[key] else 0
        maximum=lambda key:max(s[key],default=0)
        before,after=extractor.initialization_counts(approach.approach_id)
        print(f"{approach.approach_id}: unique_tracks={len(unique[approach.approach_id])} entry_crossings={extractor.entry_crossing_count(approach.approach_id)} initialized_before_line={before} initialized_after_line={after} mean_arrival_vpm={avg('arrival'):.2f} max_arrival_vpm={maximum('arrival'):.2f} mean_active={avg('active'):.2f} max_active={maximum('active')} mean_occupancy={avg('occupancy'):.4f} max_occupancy={maximum('occupancy'):.4f} max_queue={maximum('queue')} mean_wait={avg('wait'):.2f} max_wait={maximum('wait'):.2f} heavy_active_events={sum(s['heavy'])} mean_pressure={avg('pressure'):.3f} max_pressure={maximum('pressure'):.3f} mean_green={avg('green'):.2f} max_green={maximum('green')}")
    short_pct=(tracking.tracks_lifetime_under_5_frames/tracking.tracks_created*100) if tracking.tracks_created else 0
    print(f"tracking: tracks_created={tracking.tracks_created} tracks_expired={tracking.tracks_expired} mean_lifetime_frames={tracking.mean_track_lifetime_frames:.2f} median_lifetime_frames={tracking.median_track_lifetime_frames:.2f} under_3_frames={tracking.tracks_lifetime_under_3_frames} under_5_frames={tracking.tracks_lifetime_under_5_frames} short_lived_pct={short_pct:.2f} mean_matched_iou={tracking.mean_matched_iou:.4f} unmatched_detections_per_frame={tracking.unmatched_detections_per_frame:.3f}")
    if hasattr(detector,"raw_counts"):
        print("detections_by_raw_class:",dict(sorted(detector.raw_counts.items())))
        print("detection_zones:",{f"{raw}:{zone}":count for (raw,zone),count in sorted(detector.zone_counts.items())})
    try:
        import torch
        print(f"peak_cuda_vram_mb: {torch.cuda.max_memory_allocated()/1048576:.1f}" if torch.cuda.is_available() else "peak_cuda_vram_mb: unavailable")
    except ImportError: print("peak_cuda_vram_mb: unavailable")
    print(f"plan_updates: {len(events)} phase_order_changes: {phase_order_changes}")


if __name__=="__main__": main()
