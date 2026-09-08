"""Traffic-only BMD-45 detector and presentation taxonomy."""
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from cv.common.frame import FramePacket
from cv.detection.models import BoundingBox, VehicleDetection
from cv.tracking.association import intersection_over_union

BMD_DISPLAY_CLASSES = {
    "Hatchback":"Car", "Sedan":"Car", "SUV":"SUV", "MUV":"MUV",
    "Bus":"Bus", "Truck":"Truck", "Three-wheeler":"Auto-rickshaw",
    "Two-wheeler":"Motorcycle", "LCV":"LCV", "Mini-bus":"Mini-bus",
    "Tempo-traveller":"Tempo Traveller", "Bicycle":"Bicycle", "Van":"Van",
}
BMD_CONTROL_CLASSES = {
    "Hatchback":"car", "Sedan":"car", "SUV":"car", "MUV":"car",
    "Bus":"bus", "Truck":"truck", "Three-wheeler":"car",
    "Two-wheeler":"motorcycle", "LCV":"truck", "Mini-bus":"bus",
    "Tempo-traveller":"car", "Bicycle":"motorcycle", "Van":"car",
}
HEAVY_RAW_CLASSES = frozenset({"Bus", "Mini-bus", "Truck", "LCV"})


def display_vehicle_class(raw_class: str) -> str:
    return BMD_DISPLAY_CLASSES.get(raw_class, "Vehicle")


def control_vehicle_class(raw_class: str) -> str:
    return BMD_CONTROL_CLASSES.get(raw_class, raw_class.lower())


def stable_display_class(evidence: dict[str,float], minimum_consensus: float=.55) -> str:
    """Return a confidence-weighted class sourced only from detector evidence."""
    if not evidence or sum(evidence.values()) <= 0: return "Vehicle"
    raw, support = sorted(evidence.items(), key=lambda item:(-item[1], item[0]))[0]
    return display_vehicle_class(raw) if support/sum(evidence.values()) >= minimum_consensus else "Vehicle"


def class_aware_nms(detections: Sequence[VehicleDetection], iou_threshold: float=.55) -> tuple[VehicleDetection,...]:
    kept=[]
    for detection in sorted(detections,key=lambda x:(-x.confidence,x.class_name,x.bbox.x1,x.bbox.y1)):
        if any(control_vehicle_class(old.class_name)==control_vehicle_class(detection.class_name)
               and intersection_over_union(old.bbox,detection.bbox)>=iou_threshold for old in kept): continue
        kept.append(detection)
    return tuple(sorted(kept,key=lambda x:(x.bbox.x1,x.bbox.y1,x.class_name,-x.confidence)))


def detection_zone(detection: VehicleDetection) -> str:
    y=detection.bbox.y2/max(detection.frame.height,1)
    return "FAR" if y < .48 else "MID" if y < .72 else "NEAR"


class BMD45TrafficDetector:
    """BMD-45 inference isolated from the locked ANPR detector path."""
    def __init__(self, weights: str|Path, *, device="cuda", image_size=960,
                 general_threshold=.25, two_wheeler_threshold=.16, three_wheeler_threshold=.20,
                 far_field_enabled=False, far_field_polygon=(), far_field_scale=1.5):
        self.weights=Path(weights); self.device=device; self.image_size=image_size
        self.thresholds={"Two-wheeler":two_wheeler_threshold,"Three-wheeler":three_wheeler_threshold}
        self.general_threshold=general_threshold; self.far_field_enabled=far_field_enabled
        self.far_field_polygon=tuple(far_field_polygon); self.far_field_scale=far_field_scale
        if not self.weights.is_file(): raise FileNotFoundError(self.weights)
        from ultralytics import YOLO
        self.model=YOLO(str(self.weights)); self.raw_counts=Counter(); self.zone_counts=Counter()

    def detect(self, frame: FramePacket) -> tuple[VehicleDetection,...]:
        minimum=min(self.general_threshold,*self.thresholds.values())
        kwargs=dict(source=frame.image,conf=minimum,imgsz=self.image_size,device=self.device,verbose=False)
        import torch
        with torch.inference_mode(): result=self.model.predict(**kwargs)[0]
        detections=[]
        if result.boxes is not None:
            for box in result.boxes:
                raw=str(result.names[int(box.cls.item())]); confidence=float(box.conf.item())
                if raw not in BMD_DISPLAY_CLASSES or confidence < self.thresholds.get(raw,self.general_threshold): continue
                xyxy=box.xyxy[0].tolist()
                detection=VehicleDetection(frame,BoundingBox(*map(float,xyxy)),raw,confidence,"bmd45-yolov12s",self.weights.stem)
                detections.append(detection); self.raw_counts[raw]+=1; self.zone_counts[(raw,detection_zone(detection))]+=1
        # A far-field pass is intentionally disabled until the 960 benchmark proves it necessary.
        return class_aware_nms(detections)
