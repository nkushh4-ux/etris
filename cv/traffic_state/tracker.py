from dataclasses import dataclass
from math import hypot
from statistics import mean, median

from cv.detection.models import BoundingBox
from cv.tracking.association import intersection_over_union
from cv.tracking.models import TrackState, TrackStatus
from cv.traffic_state.detector import control_vehicle_class, stable_display_class


@dataclass(frozen=True, slots=True)
class TrafficTrackingDiagnostics:
    tracks_created:int; tracks_expired:int; mean_track_lifetime_frames:float
    median_track_lifetime_frames:float; tracks_lifetime_under_3_frames:int
    tracks_lifetime_under_5_frames:int; mean_matched_iou:float
    unmatched_detections_per_frame:float

    def to_dict(self): return {name:getattr(self,name) for name in self.__dataclass_fields__}


class TrafficStateTracker:
    """Traffic-only class-aware IoU/centroid tracker; ANPR tracking is untouched."""
    def __init__(self, minimum_iou=.05, centroid_gate=.06, max_missed_frames=12):
        if not 0<=minimum_iou<=1 or centroid_gate<=0 or max_missed_frames<0: raise ValueError("invalid traffic tracker settings")
        self.minimum_iou=minimum_iou; self.centroid_gate=centroid_gate; self.max_missed_frames=max_missed_frames
        self._tracks={}; self._centers={}; self._next_id=1; self._lifetimes=[]; self._matched_ious=[]; self._unmatched=[]
        self.tracks_created=0; self.tracks_expired=0

    def update(self,detections):
        detections=tuple(detections); frame=detections[0].frame if detections else None
        width=frame.width if frame else 1; height=frame.height if frame else 1; diagonal=hypot(width,height) or 1
        candidates=[]
        for track_id,track in sorted(self._tracks.items()):
            predicted=self._predicted_bbox(track_id,track.bbox)
            for index,detection in enumerate(detections):
                if detection.class_name!=track.class_name: continue
                iou=intersection_over_union(predicted,detection.bbox)
                distance=hypot(predicted.center[0]-detection.bbox.center[0],predicted.center[1]-detection.bbox.center[1])/diagonal
                if iou>=self.minimum_iou or distance<=self.centroid_gate:
                    score=.75*iou+.25*max(0,1-distance/self.centroid_gate)
                    candidates.append((-score,track_id,index,iou))
        candidates.sort(); used_tracks=set(); used_detections=set(); matches=[]
        for _,track_id,index,iou in candidates:
            if track_id not in used_tracks and index not in used_detections:
                used_tracks.add(track_id); used_detections.add(index); matches.append((track_id,index,iou))
        for track_id,index,iou in matches:
            track=self._tracks[track_id]; self._centers.setdefault(track_id,[]).append(track.bbox.center)
            self._centers[track_id]=self._centers[track_id][-2:]; track.update(detections[index]); self._matched_ious.append(iou)
        for track_id,track in list(self._tracks.items()):
            if track_id not in used_tracks:
                track.missed_frames+=1; track.age+=1; track.status=TrackStatus.LOST
                if track.missed_frames>self.max_missed_frames: self._expire(track_id)
        unmatched=[index for index in range(len(detections)) if index not in used_detections]; self._unmatched.append(len(unmatched))
        for index in unmatched:
            detection=detections[index]; track_id=self._next_id; self._next_id+=1; self.tracks_created+=1
            self._tracks[track_id]=TrackState(track_id,detection.frame.camera_id,detection.bbox,detection.class_name,
                detection.frame.frame_index,detection.frame.frame_index,detection.frame.timestamp,detection.frame.timestamp,
                detection.confidence,history=[detection.bbox])
            self._centers[track_id]=[]
        return tuple(self._tracks[key] for key in sorted(self._tracks) if self._tracks[key].missed_frames==0)

    def _predicted_bbox(self,track_id,bbox):
        centers=self._centers.get(track_id,[])
        if len(centers)<2:return bbox
        dx=centers[-1][0]-centers[-2][0]; dy=centers[-1][1]-centers[-2][1]
        return BoundingBox(bbox.x1+dx,bbox.y1+dy,bbox.x2+dx,bbox.y2+dy)

    def _expire(self,track_id):
        track=self._tracks.pop(track_id); self._centers.pop(track_id,None); self.tracks_expired+=1
        self._lifetimes.append(track.last_frame_index-track.first_frame_index+1)

    def finalize_all(self):
        for track_id in list(self._tracks): self._expire(track_id)

    def diagnostics(self):
        lifetimes=list(self._lifetimes)+[x.last_frame_index-x.first_frame_index+1 for x in self._tracks.values()]
        return TrafficTrackingDiagnostics(self.tracks_created,self.tracks_expired,mean(lifetimes) if lifetimes else 0,
            median(lifetimes) if lifetimes else 0,sum(x<3 for x in lifetimes),sum(x<5 for x in lifetimes),
            mean(self._matched_ious) if self._matched_ious else 0,mean(self._unmatched) if self._unmatched else 0)


class TrafficByteTracker(TrafficStateTracker):
    """Traffic-only confidence-stratified association with ByteTrack semantics.

    High-confidence detections create tracks. Lower-confidence detections may
    recover existing tracks, preventing weak observations from creating noise.
    """
    def __init__(self, activation_threshold=.16, low_confidence_threshold=.10,
                 minimum_iou=.05, centroid_gate=.06, max_missed_frames=30,
                 class_consensus_threshold=.55):
        if not 0 <= low_confidence_threshold <= activation_threshold <= 1: raise ValueError("invalid ByteTrack thresholds")
        super().__init__(minimum_iou,centroid_gate,max_missed_frames)
        self.activation_threshold=activation_threshold; self.low_confidence_threshold=low_confidence_threshold
        self.class_consensus_threshold=class_consensus_threshold; self._class_evidence={}

    def update(self,detections):
        detections=tuple(x for x in detections if x.confidence>=self.low_confidence_threshold)
        frame=detections[0].frame if detections else None
        width=frame.width if frame else 1; height=frame.height if frame else 1; diagonal=hypot(width,height) or 1
        high=[(i,x) for i,x in enumerate(detections) if x.confidence>=self.activation_threshold]
        low=[(i,x) for i,x in enumerate(detections) if x.confidence<self.activation_threshold]
        used_tracks=set(); used_indices=set(); matches=[]
        for pool in (high,low):
            candidates=[]
            for track_id,track in sorted(self._tracks.items()):
                if track_id in used_tracks: continue
                predicted=self._predicted_bbox(track_id,track.bbox)
                for index,detection in pool:
                    iou=intersection_over_union(predicted,detection.bbox)
                    distance=hypot(predicted.center[0]-detection.bbox.center[0],predicted.center[1]-detection.bbox.center[1])/diagonal
                    if iou>=self.minimum_iou or distance<=self.centroid_gate:
                        candidates.append((-(.75*iou+.25*max(0,1-distance/self.centroid_gate)),track_id,index,iou))
            for _,track_id,index,iou in sorted(candidates):
                if track_id not in used_tracks and index not in used_indices:
                    used_tracks.add(track_id); used_indices.add(index); matches.append((track_id,index,iou))
        for track_id,index,iou in matches:
            detection=detections[index]; track=self._tracks[track_id]
            self._centers.setdefault(track_id,[]).append(track.bbox.center); self._centers[track_id]=self._centers[track_id][-2:]
            self._class_evidence.setdefault(track_id,{})[detection.class_name]=self._class_evidence.get(track_id,{}).get(detection.class_name,0)+detection.confidence
            track.update(detection); track.class_name=control_vehicle_class(self.raw_class(track_id)); self._matched_ious.append(iou)
        for track_id,track in list(self._tracks.items()):
            if track_id not in used_tracks:
                track.missed_frames+=1; track.age+=1; track.status=TrackStatus.LOST
                if track.missed_frames>self.max_missed_frames: self._expire(track_id)
        unmatched_high=[(i,x) for i,x in high if i not in used_indices]; self._unmatched.append(len(unmatched_high))
        for index,detection in unmatched_high:
            track_id=self._next_id; self._next_id+=1; self.tracks_created+=1
            self._class_evidence[track_id]={detection.class_name:detection.confidence}
            self._tracks[track_id]=TrackState(track_id,detection.frame.camera_id,detection.bbox,control_vehicle_class(detection.class_name),
                detection.frame.frame_index,detection.frame.frame_index,detection.frame.timestamp,detection.frame.timestamp,
                detection.confidence,history=[detection.bbox]); self._centers[track_id]=[]
        return tuple(self._tracks[key] for key in sorted(self._tracks) if self._tracks[key].missed_frames==0)

    def raw_class(self,track_id):
        evidence=self._class_evidence.get(track_id,{})
        return min(evidence,key=lambda key:(-evidence[key],key)) if evidence else "Vehicle"

    def display_class(self,track_id):
        return stable_display_class(self._class_evidence.get(track_id,{}),self.class_consensus_threshold)

    def class_confidence(self,track_id):
        evidence=self._class_evidence.get(track_id,{})
        return max(evidence.values())/sum(evidence.values()) if evidence and sum(evidence.values()) else 0.

    def _expire(self,track_id):
        super()._expire(track_id); self._class_evidence.pop(track_id,None)
