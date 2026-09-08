"""Stage 12A restricted-parking video demo using the existing traffic stack."""
import argparse
import json
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

from backend.alerts.repository import InMemoryAlertRepository
from backend.alerts.service import AlertService
from cv.alerts.config import load_restricted_parking_config
from cv.alerts.mask_stabilizer import SemanticMaskStabilizer
from cv.alerts.parking_state import ParkingObservation
from cv.alerts.restricted_parking import ParkingState, RestrictedParkingEngine
from cv.alerts.restricted_surface import policy_mask
from cv.alerts.surface_segmentation import (
    HuggingFaceSegFormerCityscapesProvider,
    TorchScriptCityscapesProvider,
)
from cv.common.frame import FramePacket
from cv.traffic_state.tracker import TrafficByteTracker


def options():
    parser=argparse.ArgumentParser(); parser.add_argument("--video",required=True); parser.add_argument("--config",default=str(ROOT/"configs/restricted_parking_demo.json"))
    parser.add_argument("--output-dir",default=str(ROOT/"runs/restricted_parking_demo")); parser.add_argument("--max-frames",type=int)
    parser.add_argument("--device",default="cuda"); parser.add_argument("--imgsz",type=int,default=960); parser.add_argument("--calibration-preview",action="store_true")
    parser.add_argument("--hide-restricted-zones",action="store_true"); parser.add_argument("--no-display",action="store_true"); parser.add_argument("--write-video",action="store_true")
    parser.add_argument("--show-semantic-mask",action="store_true"); parser.add_argument("--show-policy-mask",action="store_true"); parser.add_argument("--show-restricted-mask",action="store_true")
    return parser.parse_args()


def polygon_points(zone,width,height): return np.asarray([(round(x*(width-1)),round(y*(height-1))) for x,y in zone.polygon],np.int32)


def overlay_mask(image,mask,color,alpha=.28):
    if mask is None: return
    layer=np.zeros_like(image); layer[mask]=color; cv2.addWeighted(layer,alpha,image,1,0,image)


def ground_contact_region(bbox,width,height,fraction):
    x1=max(0,int(bbox.x1)); x2=min(width,int(np.ceil(bbox.x2))); y2=min(height,int(np.ceil(bbox.y2)))
    y1=max(0,int(np.floor(bbox.y2-(bbox.y2-bbox.y1)*fraction)))
    return [x1,y1,x2,y2]


def main():
    args=options(); config=load_restricted_parking_config(args.config); output=Path(args.output_dir); output.mkdir(parents=True,exist_ok=True)
    capture=cv2.VideoCapture(args.video); fps=float(capture.get(cv2.CAP_PROP_FPS)) or 30.; width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)); height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ok,first=capture.read()
    if not ok: raise RuntimeError("unable to read video")
    provider=None; stabilizer=None; semantic_result=None
    if config.semantic_enabled:
        model_path=Path(config.semantic_model_path)
        if not model_path.is_absolute(): model_path=ROOT/model_path
        provider_class=HuggingFaceSegFormerCityscapesProvider if config.semantic_provider=="huggingface_segformer" else TorchScriptCityscapesProvider
        provider=provider_class(model_path,device=args.device,input_size=config.semantic_input_size)
        stabilizer=SemanticMaskStabilizer(config.stabilization_frames,config.semantic_refresh_interval_s)
        preview_frames=[first]
        while len(preview_frames)<config.stabilization_frames:
            ok,image=capture.read()
            if not ok: break
            preview_frames.append(image)
        for index,image in enumerate(preview_frames):
            semantic_result=stabilizer.add(provider.segment(image),index/fps)
        first=preview_frames[-1]
    if args.calibration_preview:
        resolver=RestrictedParkingEngine(config).resolver
        if args.show_semantic_mask and semantic_result is not None:
            overlay_mask(first,semantic_result.road_mask,(180,100,20)); overlay_mask(first,semantic_result.sidewalk_mask,(180,80,220))
        restricted=np.zeros((height,width),bool)
        for zone in config.zones:
            if args.show_policy_mask: overlay_mask(first,policy_mask(zone,width,height),(255,170,50))
            zone_restricted=resolver.resolve(zone,width,height,semantic_result)
            if zone_restricted is not None: restricted|=zone_restricted
            if args.show_restricted_mask: overlay_mask(first,zone_restricted,(0,0,255))
            cv2.polylines(first,[polygon_points(zone,width,height)],True,(0,80,255),3)
        cv2.imwrite(str(output/"semantic_preview.jpg"),first)
        if semantic_result is not None:
            cv2.imwrite(str(output/"sidewalk_mask.png"),semantic_result.sidewalk_mask.astype(np.uint8)*255)
            cv2.imwrite(str(output/"road_mask.png"),semantic_result.road_mask.astype(np.uint8)*255)
        cv2.imwrite(str(output/"restricted_mask.png"),restricted.astype(np.uint8)*255)
        print(output/"semantic_preview.jpg"); return
    capture.set(cv2.CAP_PROP_POS_FRAMES,0)
    from cv.traffic_state.detector import BMD45TrafficDetector
    detector=BMD45TrafficDetector(ROOT/"weights/traffic/bmd45-yolov12s.pt",device=args.device,image_size=args.imgsz)
    tracker=TrafficByteTracker(); engine=RestrictedParkingEngine(config); repository=InMemoryAlertRepository(); service=AlertService(repository)
    writer=cv2.VideoWriter(str(output/"restricted_parking_annotated.mp4"),cv2.VideoWriter_fourcc(*"mp4v"),fps,(width,height)) if args.write_video else None
    base=datetime(2026,1,1,tzinfo=UTC); frame_index=0; start_clock=time.perf_counter(); segmentation_ms=[]; cached_mask_ms=[]; contact_debug=[]
    while args.max_frames is None or frame_index<args.max_frames:
        ok,image=capture.read()
        if not ok: break
        source_time=frame_index/fps; packet=FramePacket(config.camera_id,frame_index,base+timedelta(seconds=source_time),image)
        if provider is not None and stabilizer.needs_refresh(source_time):
            started=time.perf_counter(); semantic_result=stabilizer.add(provider.segment(image),source_time); segmentation_ms.append((time.perf_counter()-started)*1000)
        tracks=tracker.update(detector.detect(packet)); observations=[]
        for track in tracks:
            observations.append(ParkingObservation(track.track_id,track.bbox,track.detection_confidence,tracker.display_class(track.track_id),tracker.raw_class(track.track_id),tracker.class_confidence(track.track_id)))
        cache_started=time.perf_counter(); parking_events=engine.update(observations,source_time_s=source_time,width=width,height=height,semantic_result=semantic_result); cached_mask_ms.append((time.perf_counter()-cache_started)*1000); service.process_parking_events(parking_events)
        if args.show_semantic_mask and semantic_result is not None:
            overlay_mask(image,semantic_result.road_mask,(180,100,20)); overlay_mask(image,semantic_result.sidewalk_mask,(180,80,220))
        if not args.hide_restricted_zones:
            for zone in config.zones:
                if args.show_restricted_mask: overlay_mask(image,engine.resolver.resolve(zone,width,height,semantic_result),(0,0,255))
                cv2.polylines(image,[polygon_points(zone,width,height)],True,(0,80,255),2)
        for track in tracks:
            states=[engine.state(track.track_id,zone.zone_id) for zone in config.zones]; illegal=ParkingState.ILLEGALLY_PARKED in states
            pending=ParkingState.STATIONARY_PENDING in states; blink=int(source_time*4)%2==0
            b=track.bbox; color=(0,0,255) if illegal and blink else (0,170,255) if pending else (80,225,160)
            cv2.rectangle(image,(int(b.x1),int(b.y1)),(int(b.x2),int(b.y2)),color,3 if illegal else 2)
            label=f"#{track.track_id} {tracker.display_class(track.track_id)} {track.detection_confidence:.2f}"
            if illegal:
                parked=max((engine.stationary_seconds(track.track_id,zone.zone_id,source_time) for zone in config.zones),default=0.)
                evidence=next((engine.evidence(track.track_id,zone.zone_id) for zone in config.zones if engine.state(track.track_id,zone.zone_id) is ParkingState.ILLEGALLY_PARKED),None)
                surface=evidence.surface_class if evidence else "UNKNOWN"
                label=f"ILLEGALLY PARKED | {label} | Surface: {surface} | Stationary: {parked:04.1f}s"
            cv2.putText(image,label,(int(b.x1),max(18,int(b.y1)-5)),cv2.FONT_HERSHEY_SIMPLEX,.48,color,2)
            if len(contact_debug)<50:
                for zone in config.zones:
                    evidence=engine.evidence(track.track_id,zone.zone_id)
                    if evidence is not None:
                        contact_debug.append({"frame_index":frame_index,"track_id":track.track_id,
                            "vehicle_class":tracker.display_class(track.track_id),"zone_id":zone.zone_id,
                            "restricted_overlap":evidence.restricted_overlap,"surface_class":evidence.surface_class,
                            "surface_confidence":evidence.surface_confidence,"ground_contact_bbox":ground_contact_region(b,width,height,config.ground_contact_fraction)})
                        break
        if writer: writer.write(image)
        if not args.no_display:
            cv2.imshow("ETRIS Restricted Parking",image)
            if cv2.waitKey(1)&0xff==ord("q"): break
        frame_index+=1
    tracker.finalize_all(); capture.release()
    if writer: writer.release()
    cv2.destroyAllWindows(); alerts=[x.to_dict() for x in repository.list()]
    (output/"restricted_parking_alerts.json").write_text(json.dumps(alerts,indent=2),encoding="utf-8")
    (output/"ground_contact_debug.json").write_text(json.dumps(contact_debug,indent=2),encoding="utf-8")
    elapsed=time.perf_counter()-start_clock
    print(f"frames_processed: {frame_index}\nalerts: {len(alerts)}\nactive: {sum(x['status']!='CLEARED' for x in alerts)}")
    print(f"combined_pipeline_fps: {frame_index/elapsed if elapsed else 0:.3f}\nsegmentation_refreshes: {len(segmentation_ms)}\nmean_segmentation_ms: {sum(segmentation_ms)/len(segmentation_ms) if segmentation_ms else 0:.2f}")
    print(f"mean_cached_mask_overhead_ms: {sum(cached_mask_ms)/len(cached_mask_ms) if cached_mask_ms else 0:.3f}")


if __name__=="__main__": main()
