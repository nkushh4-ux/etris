from collections import defaultdict, deque
from dataclasses import dataclass
from math import hypot

from cv.traffic_state.geometry import (
    bbox_coverage_area,
    line_side,
    moves_in_direction,
    point_in_polygon,
    polygon_mask,
    segments_intersect,
    union_bbox_occupancy,
)
from cv.traffic_state.models import (
    ExtractedApproachState,
    TrackObservation,
    TrafficStateConfig,
    TrafficStateFrame,
)

VEHICLE_CLASSES={"car","motorcycle","bus","truck"}


@dataclass
class _TrackMemory:
    anchor: tuple[float,float]
    time_s: float
    last_seen_s: float
    approach_id: str | None
    queue_started_s: float | None = None


class TrafficStateExtractor:
    def __init__(self, config: TrafficStateConfig) -> None:
        self.config=config; self._tracks: dict[int,_TrackMemory]={}
        self._arrivals={x.approach_id:deque() for x in config.approaches}
        self._crossed={x.approach_id:set() for x in config.approaches}
        self._initialized_before={x.approach_id:set() for x in config.approaches}
        self._initialized_after={x.approach_id:set() for x in config.approaches}
        self._next_update_s=0.0

    def update(self, tracks, *, frame_index:int, video_time_s:float, width:int, height:int) -> TrafficStateFrame:
        if video_time_s < 0: raise ValueError("video_time_s must be non-negative")
        self._expire(video_time_s)
        observations=tuple(TrackObservation(x.track_id,x.bbox,x.class_name) for x in tracks if x.class_name in VEHICLE_CLASSES)
        masks={a.approach_id:polygon_mask(a.traffic_polygon,width,height) for a in self.config.approaches}
        assigned=defaultdict(list); speeds={}; anchors={}
        diagonal=hypot(width,height) or 1.0
        for track in sorted(observations,key=lambda x:x.track_id):
            anchor=(track.bbox.x1+track.bbox.x2)/2/max(width,1), track.bbox.y2/max(height,1)
            anchors[track.track_id]=anchor
            candidates=[]
            for approach in self.config.approaches:
                if point_in_polygon(anchor,approach.traffic_polygon):
                    candidates.append((bbox_coverage_area(track.bbox,masks[approach.approach_id]),approach.approach_id,approach))
            selected=sorted(candidates,key=lambda x:(-x[0],x[1]))[0][2] if candidates else None
            previous=self._tracks.get(track.track_id)
            speed=None
            if previous and video_time_s>previous.time_s:
                dx=(anchor[0]-previous.anchor[0])*width; dy=(anchor[1]-previous.anchor[1])*height
                speed=hypot(dx,dy)/diagonal/(video_time_s-previous.time_s)
            speeds[track.track_id]=speed
            if selected:
                assigned[selected.approach_id].append(track)
                if previous is None:
                    midpoint=((selected.entry_line[0][0]+selected.entry_line[1][0])/2,
                              (selected.entry_line[0][1]+selected.entry_line[1][1])/2)
                    after=(midpoint[0]+selected.entry_direction[0]*.01,midpoint[1]+selected.entry_direction[1]*.01)
                    target=self._initialized_after if line_side(anchor,selected.entry_line)*line_side(after,selected.entry_line)>=0 else self._initialized_before
                    target[selected.approach_id].add(track.track_id)
                if previous and track.track_id not in self._crossed[selected.approach_id]:
                    crossed=segments_intersect((previous.anchor,anchor),selected.entry_line)
                    if crossed and moves_in_direction(previous.anchor,anchor,selected.entry_direction):
                        self._arrivals[selected.approach_id].append((video_time_s,track.track_id)); self._crossed[selected.approach_id].add(track.track_id)
            self._tracks[track.track_id]=_TrackMemory(anchor,video_time_s,video_time_s,
                selected.approach_id if selected else None,
                previous.queue_started_s if previous and previous.approach_id==(selected.approach_id if selected else None) else None)
        states=[]
        for approach in sorted(self.config.approaches,key=lambda x:x.approach_id):
            cutoff=video_time_s-self.config.arrival_window_s; arrivals=self._arrivals[approach.approach_id]
            while arrivals and arrivals[0][0]<cutoff: arrivals.popleft()
            active=assigned[approach.approach_id]; queued=[]; waits=[]
            for track in active:
                memory=self._tracks[track.track_id]; inside_queue=point_in_polygon(anchors[track.track_id],approach.queue_polygon)
                slow=speeds[track.track_id] is not None and speeds[track.track_id] <= self.config.stationary_motion_threshold
                if inside_queue and slow:
                    if memory.queue_started_s is None: memory.queue_started_s=video_time_s
                    dwell=video_time_s-memory.queue_started_s
                    if dwell>=self.config.queue_stationary_min_s: queued.append(track.track_id); waits.append(dwell)
                else: memory.queue_started_s=None
            occupancy=union_bbox_occupancy(tuple(x.bbox for x in active),approach.traffic_polygon,width,height)
            states.append(ExtractedApproachState(approach.approach_id,approach.label,len(active),len(queued),
                occupancy,sum(waits)/len(waits) if waits else 0.0,len(arrivals)/self.config.arrival_window_s*60,
                sum(x.class_name in {"bus","truck"} for x in active),tuple(x.track_id for x in active),tuple(queued)))
        due=video_time_s+1e-9>=self._next_update_s
        if due:
            while self._next_update_s<=video_time_s+1e-9: self._next_update_s+=self.config.controller_update_interval_s
        return TrafficStateFrame(video_time_s,frame_index,tuple(states),due)

    def _expire(self, now:float) -> None:
        stale=[key for key,value in self._tracks.items() if now-value.last_seen_s>self.config.track_expiry_s]
        for key in stale:
            del self._tracks[key]

    @property
    def tracked_memory_count(self): return len(self._tracks)

    def entry_crossing_count(self, approach_id: str) -> int:
        return len(self._crossed[approach_id])

    def initialization_counts(self, approach_id: str) -> tuple[int,int]:
        return len(self._initialized_before[approach_id]),len(self._initialized_after[approach_id])
