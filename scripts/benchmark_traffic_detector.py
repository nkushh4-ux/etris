"""Deterministic Stage 11.5 visual spot-check; never used by ANPR."""
import argparse
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

from cv.common.frame import FramePacket
from cv.traffic_state.config import load_traffic_state_config
from cv.traffic_state.geometry import point_in_polygon

FRAMES=(0,75,150,250,400,550)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--video",default=str(ROOT/"traffic_2.mp4"))
    parser.add_argument("--config",default=str(ROOT/"configs/traffic_signal_demo.json")); parser.add_argument("--output-dir",default=str(ROOT/"runs/signal_demo_v2/spot_checks"))
    parser.add_argument("--imgsz",type=int,default=960); parser.add_argument("--device",default="cuda"); options=parser.parse_args()
    from cv.detection.ultralytics_detector import UltralyticsVehicleDetector
    from cv.traffic_state.detector import BMD45TrafficDetector, display_vehicle_class
    detectors={"BEFORE":UltralyticsVehicleDetector(ROOT/"weights/yolo26n.pt",device=options.device),
               "AFTER":BMD45TrafficDetector(ROOT/"weights/traffic/bmd45-yolov12s.pt",device=options.device,image_size=options.imgsz)}
    config=load_traffic_state_config(options.config); output=Path(options.output_dir); output.mkdir(parents=True,exist_ok=True)
    capture=cv2.VideoCapture(options.video); rows=[]; counts={key:Counter() for key in detectors}
    for frame_index in FRAMES:
        capture.set(cv2.CAP_PROP_POS_FRAMES,frame_index); ok,image=capture.read()
        if not ok: raise RuntimeError(f"cannot read frame {frame_index}")
        pair=[]
        for title,detector in detectors.items():
            shown=image.copy(); packet=FramePacket("SPOT",frame_index,datetime(2026,1,1,tzinfo=UTC),image)
            detections=[]
            for detection in detector.detect(packet):
                anchor=(detection.bbox.center[0]/image.shape[1],detection.bbox.y2/image.shape[0])
                if any(point_in_polygon(anchor,region.traffic_polygon) for region in config.approaches): detections.append(detection)
            for detection in detections:
                counts[title][detection.class_name]+=1; b=detection.bbox
                label=display_vehicle_class(detection.class_name) if title=="AFTER" else detection.class_name
                cv2.rectangle(shown,(int(b.x1),int(b.y1)),(int(b.x2),int(b.y2)),(70,230,160),2)
                cv2.putText(shown,f"{label} {detection.confidence:.2f}",(int(b.x1),max(15,int(b.y1)-3)),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
            cv2.rectangle(shown,(0,0),(shown.shape[1],42),(5,15,12),-1); cv2.putText(shown,f"{title} / FRAME {frame_index} / {len(detections)} DETECTIONS",(14,28),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2)
            target=output/f"{frame_index:04d}_{title.lower()}.jpg"; cv2.imwrite(str(target),shown); pair.append(cv2.resize(shown,(960,540)))
        rows.append(np.hstack(pair))
    capture.release(); cv2.imwrite(str(output/"before_after_contact_sheet.jpg"),np.vstack(rows))
    print({key:dict(value) for key,value in counts.items()})


if __name__=="__main__": main()
