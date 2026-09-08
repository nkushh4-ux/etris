"""Source-time semantic-mask caching for fixed cameras."""
from collections import deque

import numpy as np

from cv.alerts.surface_segmentation import SurfaceSegmentationResult


class SemanticMaskStabilizer:
    def __init__(self,stabilization_frames=3,refresh_interval_s=5.):
        if stabilization_frames<1 or refresh_interval_s<=0: raise ValueError("invalid stabilization settings")
        self.stabilization_frames=stabilization_frames; self.refresh_interval_s=refresh_interval_s
        self._samples=deque(maxlen=stabilization_frames); self._last_segmented_s=None; self._stable=None

    def needs_refresh(self,source_time_s):
        return self._stable is None or len(self._samples)<self.stabilization_frames or source_time_s-(self._last_segmented_s or 0)>=self.refresh_interval_s

    def add(self,result:SurfaceSegmentationResult,source_time_s:float):
        if self._samples and result.labels.shape!=self._samples[0].labels.shape: self._samples.clear()
        self._samples.append(result); self._last_segmented_s=source_time_s
        labels=np.stack([x.labels for x in self._samples]); confidence=np.stack([x.confidence for x in self._samples])
        # Deterministic per-pixel majority vote; lower class ID breaks ties.
        classes=len(result.raw_class_names); votes=np.stack([(labels==index).sum(0) for index in range(classes)])
        stable_labels=votes.argmax(0).astype(np.uint8); stable_confidence=np.mean(confidence,axis=0).astype(np.float32)
        self._stable=SurfaceSegmentationResult(stable_labels,stable_confidence,result.raw_class_names,result.road_class_ids,result.sidewalk_class_ids)
        return self._stable

    @property
    def result(self): return self._stable
