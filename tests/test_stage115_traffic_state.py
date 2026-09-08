from datetime import UTC, datetime

import numpy as np
import pytest

from backend.signal_control.controller import AdaptiveSignalController
from backend.signal_control.vision_adapter import (
    approach_states_from_video,
    plan_from_video,
)
from cv.common.frame import FramePacket
from cv.detection.models import BoundingBox, VehicleDetection
from cv.traffic_state.detector import (
    class_aware_nms,
    control_vehicle_class,
    display_vehicle_class,
    stable_display_class,
)
from cv.traffic_state.extractor import TrafficStateExtractor
from cv.traffic_state.geometry import point_in_polygon, union_bbox_occupancy
from cv.traffic_state.models import ApproachRegion, TrackObservation, TrafficStateConfig
from cv.traffic_state.tracker import TrafficByteTracker, TrafficStateTracker

POLY=((0.0,.4),(1.0,.4),(1.0,1.0),(0.0,1.0))
QUEUE=((0.0,.5),(1.0,.5),(1.0,1.0),(0.0,1.0))
LINE=((0.0,.5),(1.0,.5))


def region(identifier="A",poly=POLY): return ApproachRegion(identifier,identifier,poly,QUEUE,LINE)
def config(*regions,**kwargs): return TrafficStateConfig("I",regions or (region(),),**kwargs)
def track(identifier=1,x=.5,y=.7,cls="car",size=.1):
    return TrackObservation(identifier,BoundingBox((x-size/2)*100,(y-size)*100,(x+size/2)*100,y*100),cls)
def update(extractor,tracks,time,index=0): return extractor.update(tracks,frame_index=index,video_time_s=time,width=100,height=100)


def test_normalized_polygon_validation():
    with pytest.raises(ValueError): region(poly=((0,0),(1.1,0),(0,1)))
    with pytest.raises(ValueError): region(poly=((0,0),(1,1)))


def test_point_in_polygon_includes_boundary():
    assert point_in_polygon((.5,.5),POLY)
    assert not point_in_polygon((.5,.2),POLY)


def test_overlap_assigns_track_once_deterministically():
    result=update(TrafficStateExtractor(config(region("B"),region("A"))),[track()],0)
    counts={x.approach_id:x.active_vehicle_count for x in result.approaches}
    assert counts=={"A":1,"B":0}


def test_entry_crossing_counted_once_and_visibility_does_not_inflate():
    extractor=TrafficStateExtractor(config(arrival_window_s=30))
    update(extractor,[track(y=.3)],0)
    crossed=update(extractor,[track(y=.7)],1)
    repeated=update(extractor,[track(y=.72)],2)
    assert crossed.approaches[0].arrival_rate_vpm==2
    assert repeated.approaches[0].arrival_rate_vpm==2


def test_trajectory_segment_crosses_line_inside_same_roi_once():
    extractor=TrafficStateExtractor(config())
    update(extractor,[track(y=.45)],0); update(extractor,[track(y=.65)],1)
    assert extractor.entry_crossing_count("A")==1
    update(extractor,[track(y=.45)],2); update(extractor,[track(y=.65)],3)
    assert extractor.entry_crossing_count("A")==1


def test_non_crossing_and_reverse_direction_are_rejected():
    no_cross=TrafficStateExtractor(config()); update(no_cross,[track(y=.7)],0); update(no_cross,[track(y=.8)],1)
    reverse=TrafficStateExtractor(config()); update(reverse,[track(y=.7)],0); update(reverse,[track(y=.45)],1)
    assert no_cross.entry_crossing_count("A")==0
    assert reverse.entry_crossing_count("A")==0


def test_track_born_after_line_is_diagnostic_not_arrival():
    extractor=TrafficStateExtractor(config()); update(extractor,[track(y=.7)],0)
    assert extractor.entry_crossing_count("A")==0
    assert extractor.initialization_counts("A")== (0,1)


def test_arrival_sliding_window_expires_crossing():
    extractor=TrafficStateExtractor(config(arrival_window_s=10,track_expiry_s=50))
    update(extractor,[track(y=.3)],0); update(extractor,[track(y=.7)],1)
    assert update(extractor,[track(y=.7)],12).approaches[0].arrival_rate_vpm==0


def test_occupancy_is_bounded_and_overlaps_are_not_double_counted():
    box=BoundingBox(20,50,60,90)
    one=union_bbox_occupancy((box,),POLY,100,100)
    duplicate=union_bbox_occupancy((box,box),POLY,100,100)
    huge=union_bbox_occupancy((BoundingBox(-100,-100,500,500),),POLY,100,100)
    assert 0<one<1 and duplicate==one and huge==1


def test_moving_track_is_not_queued():
    extractor=TrafficStateExtractor(config(queue_stationary_min_s=1))
    update(extractor,[track(x=.2)],0)
    assert update(extractor,[track(x=.8)],2).approaches[0].queue_length==0


def test_stationary_track_requires_persistence_and_wait_uses_video_time():
    extractor=TrafficStateExtractor(config(queue_stationary_min_s=1))
    update(extractor,[track()],0)
    assert update(extractor,[track()],.5).approaches[0].queue_length==0
    queued=update(extractor,[track()],1.5).approaches[0]
    later=update(extractor,[track()],3).approaches[0]
    assert queued.queue_length==1 and queued.average_waiting_time_s==1
    assert later.average_waiting_time_s==2.5


def test_movement_resumes_and_queue_episode_ends():
    extractor=TrafficStateExtractor(config(queue_stationary_min_s=.5,track_expiry_s=10))
    update(extractor,[track(x=.4)],0); update(extractor,[track(x=.4)],.5)
    assert update(extractor,[track(x=.4)],1).approaches[0].queue_length==1
    moving=update(extractor,[track(x=.8)],1.1).approaches[0]
    assert moving.queue_length==0 and moving.average_waiting_time_s==0


def test_heavy_vehicle_count_uses_bus_and_truck_only():
    result=update(TrafficStateExtractor(config()),[track(1,.2,cls="bus"),track(2,.4,cls="truck"),track(3,.6,cls="car"),track(4,.8,cls="motorcycle")],0)
    assert result.approaches[0].heavy_vehicle_count==2


def test_non_vehicle_classes_are_ignored():
    result=update(TrafficStateExtractor(config()),[track(cls="person")],0)
    assert result.approaches[0].active_vehicle_count==0


def test_approach_state_conversion_preserves_extracted_values():
    frame=update(TrafficStateExtractor(config()),[track(cls="truck")],0)
    states=approach_states_from_video(frame,"I",datetime(2026,1,1,tzinfo=UTC))
    extracted=frame.approaches[0]; state=states[0]
    assert (state.vehicle_count,state.queue_length,state.lane_occupancy,state.arrival_rate_vpm,state.heavy_vehicle_count)==(
        extracted.active_vehicle_count,extracted.queue_length,extracted.image_space_occupancy,extracted.arrival_rate_vpm,extracted.heavy_vehicle_count)


def test_controller_receives_extracted_values_unchanged():
    frame=update(TrafficStateExtractor(config()),[track()],0)
    timestamp=datetime(2026,1,1,tzinfo=UTC); controller=AdaptiveSignalController()
    direct=controller.plan("I",approach_states_from_video(frame,"I",timestamp),generated_at=timestamp)
    assert plan_from_video(frame,"I",timestamp,controller)==direct


def test_controller_update_cadence_uses_source_time():
    extractor=TrafficStateExtractor(config(controller_update_interval_s=2))
    assert update(extractor,[],0).controller_update_due
    assert not update(extractor,[],1.99).controller_update_due
    assert update(extractor,[],2).controller_update_due
    assert update(extractor,[],6.1).controller_update_due


def test_results_are_deterministic():
    first=update(TrafficStateExtractor(config()),[track(2,.7),track(1,.3)],0)
    second=update(TrafficStateExtractor(config()),[track(1,.3),track(2,.7)],0)
    assert first==second


def test_empty_frame_returns_zero_state():
    item=update(TrafficStateExtractor(config()),[],0).approaches[0]
    assert item.active_vehicle_count==item.queue_length==item.heavy_vehicle_count==0
    assert item.image_space_occupancy==item.average_waiting_time_s==item.arrival_rate_vpm==0


def test_track_expiry_cleans_internal_motion_state():
    extractor=TrafficStateExtractor(config(track_expiry_s=1))
    update(extractor,[track()],0); assert extractor.tracked_memory_count==1
    update(extractor,[],1.1); assert extractor.tracked_memory_count==0


def test_source_video_time_not_wall_clock_controls_waiting():
    extractor=TrafficStateExtractor(config(queue_stationary_min_s=.5,track_expiry_s=10))
    update(extractor,[track()],100); update(extractor,[track()],100.5)
    state=update(extractor,[track()],105.5).approaches[0]
    assert state.average_waiting_time_s==5


def test_larger_bbox_overlap_wins_ambiguous_assignment():
    left=((0,.4),(.65,.4),(.65,1),(0,1)); right=((.35,.4),(1,.4),(1,1),(.35,1))
    extractor=TrafficStateExtractor(config(region("A",left),region("B",right)))
    # Anchor is in both, but most of this bbox lies in B.
    result=update(extractor,[TrackObservation(1,BoundingBox(40,50,95,80),"car")],0)
    assert {x.approach_id:x.active_vehicle_count for x in result.approaches}=={"A":0,"B":1}


def detection(index,x1,x2,cls="car"):
    frame=FramePacket("C",index,datetime(2026,1,1,tzinfo=UTC),np.zeros((100,100,3),np.uint8))
    return VehicleDetection(frame,BoundingBox(x1,30,x2,60),cls,.9,"fake","1")


def test_traffic_tracker_preserves_id_across_temporary_miss():
    tracker=TrafficStateTracker(max_missed_frames=2)
    first=tracker.update([detection(0,10,30)])[0]; assert tracker.update([])==()
    resumed=tracker.update([detection(2,12,32)])[0]
    assert resumed.track_id==first.track_id and tracker.diagnostics().tracks_created==1


def test_traffic_tracker_is_one_to_one_and_nearby_vehicles_do_not_merge():
    tracker=TrafficStateTracker()
    initial=tracker.update([detection(0,10,30),detection(0,35,55)])
    updated=tracker.update([detection(1,12,32),detection(1,33,53)])
    assert len(initial)==len(updated)==2
    assert len({x.track_id for x in updated})==2
    assert {x.track_id for x in updated}=={x.track_id for x in initial}


def test_traffic_tracking_is_deterministic_and_reports_diagnostics():
    def run():
        tracker=TrafficStateTracker(max_missed_frames=0)
        outputs=[]
        for rows in ([detection(0,10,30),detection(0,60,80)],[detection(1,12,32),detection(1,58,78)],[]):
            outputs.append(tuple((x.track_id,x.bbox) for x in tracker.update(rows)))
        return outputs,tracker.diagnostics()
    assert run()==run()
    _,diagnostics=run()
    assert diagnostics.tracks_created==2 and diagnostics.tracks_expired==2
    assert diagnostics.mean_matched_iou>0 and diagnostics.unmatched_detections_per_frame>0


def test_bmd_class_mapping_preserves_specific_vehicle_types():
    assert display_vehicle_class("Three-wheeler")=="Auto-rickshaw"
    assert display_vehicle_class("Two-wheeler")=="Motorcycle"
    assert display_vehicle_class("Bus")=="Bus" and display_vehicle_class("Truck")=="Truck"
    assert display_vehicle_class("LCV")=="LCV" and display_vehicle_class("Van")=="Van"


def test_track_class_voting_and_uncertain_fallback():
    assert stable_display_class({"Three-wheeler":2.4,"Hatchback":.2})=="Auto-rickshaw"
    assert stable_display_class({"Bus":.9,"Truck":.8},minimum_consensus=.6)=="Vehicle"


def test_bytetrack_adapter_keeps_unique_id_and_uses_low_confidence_for_recovery():
    tracker=TrafficByteTracker(activation_threshold=.5,low_confidence_threshold=.1,max_missed_frames=2)
    strong=detection(0,10,30,"Two-wheeler")
    weak=VehicleDetection(strong.frame,BoundingBox(12,30,32,60),"Two-wheeler",.2,"fake","1")
    first=tracker.update([strong])[0]; recovered=tracker.update([weak])[0]
    assert first.track_id==recovered.track_id and recovered.class_name=="motorcycle"
    assert tracker.display_class(first.track_id)=="Motorcycle"


def test_bytetrack_does_not_create_tracks_from_low_confidence_only():
    tracker=TrafficByteTracker(activation_threshold=.5,low_confidence_threshold=.1)
    item=detection(0,10,30,"Three-wheeler")
    weak=VehicleDetection(item.frame,item.bbox,item.class_name,.2,item.detector_name,item.detector_version)
    assert tracker.update([weak])==()


def test_far_field_merge_nms_removes_duplicate_without_cross_class_collapse():
    one=detection(0,10,30,"Two-wheeler")
    duplicate=VehicleDetection(one.frame,BoundingBox(10.5,30,30.5,60),"Two-wheeler",.8,"far","1")
    auto=VehicleDetection(one.frame,one.bbox,"Three-wheeler",.7,"far","1")
    merged=class_aware_nms([duplicate,auto,one])
    assert len(merged)==2 and {x.class_name for x in merged}=={"Two-wheeler","Three-wheeler"}


def test_control_mapping_preserves_stage11_heavy_semantics():
    assert control_vehicle_class("Bus")=="bus" and control_vehicle_class("Mini-bus")=="bus"
    assert control_vehicle_class("Truck")=="truck" and control_vehicle_class("LCV")=="truck"
    assert control_vehicle_class("SUV")=="car" and control_vehicle_class("Three-wheeler")=="car"
