from typing import ClassVar

import numpy as np
import pytest

from cv.alerts.mask_stabilizer import SemanticMaskStabilizer
from cv.alerts.parking_state import (
    ParkingObservation,
    RestrictedParkingConfig,
    RestrictedParkingZone,
)
from cv.alerts.restricted_parking import ParkingState, RestrictedParkingEngine
from cv.alerts.restricted_surface import (
    RestrictedSurfaceMode,
    RestrictedSurfaceResolver,
    policy_mask,
)
from cv.alerts.surface_segmentation import (
    HuggingFaceSegFormerCityscapesProvider,
    HuggingFaceSegFormerIDDProvider,
    SurfaceSegmentationResult,
)
from cv.detection.models import BoundingBox

ZONE=RestrictedParkingZone("Z","FOOTPATH",((.25,.25),(.75,.25),(.75,.75),(.25,.75)))


def semantic(sidewalk_slice=(slice(20,80),slice(20,80)),confidence=1.):
    labels=np.zeros((100,100),np.uint8); labels[sidewalk_slice]=1
    return SurfaceSegmentationResult(labels,np.full((100,100),confidence,np.float32),("road","sidewalk"),frozenset({0}),frozenset({1}))


def semantic_config(mode="SEMANTIC",**kwargs):
    return RestrictedParkingConfig("CAM",(ZONE,),surface_mode=mode,semantic_enabled=True,
        semantic_model_path="unused-test-provider.ts",**kwargs)


def test_sidewalk_is_restricted_but_road_is_not_in_semantic_mode():
    resolver=RestrictedSurfaceResolver(RestrictedSurfaceMode.SEMANTIC)
    mask=resolver.resolve(ZONE,100,100,semantic())
    assert mask[50,50] and not mask[10,10]


def test_hybrid_is_semantic_intersection_policy():
    resolver=RestrictedSurfaceResolver(RestrictedSurfaceMode.HYBRID)
    result=resolver.resolve(ZONE,100,100,semantic((slice(0,100),slice(0,50))))
    expected=semantic((slice(0,100),slice(0,50))).sidewalk_mask&policy_mask(ZONE,100,100)
    assert np.array_equal(result,expected)


def test_policy_mode_ignores_semantic_labels():
    resolver=RestrictedSurfaceResolver(RestrictedSurfaceMode.POLICY_DEFINED)
    assert np.array_equal(resolver.resolve(ZONE,100,100,semantic((slice(0,0),slice(0,0)))),policy_mask(ZONE,100,100))
    evidence=resolver.evaluate(BoundingBox(40,20,60,70),policy_mask(ZONE,100,100),semantic(),
        width=100,height=100,ground_contact_fraction=.25,minimum_overlap=.4)
    assert evidence.surface_class=="POLICY_DEFINED" and evidence.raw_surface_class is None


def test_low_surface_confidence_is_uncertain_and_does_not_alert():
    engine=RestrictedParkingEngine(semantic_config(minimum_surface_confidence=.55))
    item=ParkingObservation(1,BoundingBox(40,30,60,60),.9,"Car","Sedan",.9)
    assert not engine.update((item,),source_time_s=0,width=100,height=100,semantic_result=semantic(confidence=.2))
    assert engine.state(1,"Z") is ParkingState.UNCERTAIN


def test_hybrid_policy_semantic_and_dwell_confirms_violation():
    engine=RestrictedParkingEngine(semantic_config(mode="HYBRID"))
    item=ParkingObservation(1,BoundingBox(40,30,60,60),.9,"Car","Sedan",.9)
    for source_time in (0,1,5):
        events=engine.update((item,),source_time_s=source_time,width=100,height=100,semantic_result=semantic())
    assert engine.state(1,"Z") is ParkingState.ILLEGALLY_PARKED
    assert events[0].policy_mode=="HYBRID" and events[0].surface_confidence==1.


def test_ground_contact_overlap_controls_occupancy():
    resolver=RestrictedSurfaceResolver(RestrictedSurfaceMode.SEMANTIC)
    result=semantic((slice(50,70),slice(40,60))); mask=resolver.resolve(ZONE,100,100,result)
    sufficient=resolver.evaluate(BoundingBox(40,20,60,70),mask,result,width=100,height=100,ground_contact_fraction=.25,minimum_overlap=.4)
    insufficient=resolver.evaluate(BoundingBox(70,20,90,70),mask,result,width=100,height=100,ground_contact_fraction=.25,minimum_overlap=.4)
    assert sufficient.occupied and sufficient.restricted_overlap>=.4
    assert not insufficient.occupied


def test_clipped_bbox_is_safe_uncertain():
    resolver=RestrictedSurfaceResolver(RestrictedSurfaceMode.SEMANTIC); result=semantic(); mask=resolver.resolve(ZONE,100,100,result)
    evidence=resolver.evaluate(BoundingBox(-2,20,20,60),mask,result,width=100,height=100,ground_contact_fraction=.25,minimum_overlap=.4)
    assert evidence.uncertain and not evidence.occupied


def test_mask_stabilization_and_source_time_cache_are_deterministic():
    stabilizer=SemanticMaskStabilizer(3,5); first=semantic(); second=semantic((slice(0,0),slice(0,0)))
    assert stabilizer.needs_refresh(0); stabilizer.add(first,0); stabilizer.add(first,1); stable=stabilizer.add(second,2)
    assert stable.sidewalk_mask[50,50] and not stabilizer.needs_refresh(4.9) and stabilizer.needs_refresh(7)
    duplicate=SemanticMaskStabilizer(3,5); duplicate.add(first,0); duplicate.add(first,1)
    assert np.array_equal(stable.labels,duplicate.add(second,2).labels)


def test_semantic_mode_requires_enabled_provider_configuration():
    with pytest.raises(ValueError): RestrictedParkingConfig("CAM",(ZONE,),surface_mode="SEMANTIC")
    with pytest.raises(ValueError): semantic_config(ground_contact_fraction=0)
    with pytest.raises(ValueError): semantic_config(minimum_restricted_overlap=1.1)


def test_polygon_only_backward_compatibility():
    engine=RestrictedParkingEngine(RestrictedParkingConfig("CAM",(ZONE,)))
    item=ParkingObservation(1,BoundingBox(40,30,60,60),.9,"Car","Sedan",.9)
    engine.update((item,),source_time_s=0,width=100,height=100)
    assert engine.state(1,"Z") is ParkingState.ON_RESTRICTED_SURFACE


class FakeProcessor:
    def __call__(self,**kwargs):
        import torch
        size=kwargs["size"]
        return {"pixel_values":torch.zeros((1,3,size["height"],size["width"]))}


class FakeConfig:
    id2label: ClassVar = {0:"road",1:"sidewalk",2:"building"}


class FakeModel:
    config=FakeConfig()
    def to(self,device): self.device=device; return self
    def eval(self): return self
    def __call__(self,**kwargs):
        import torch
        logits=torch.zeros((1,3,2,3)); logits[:,1]=4
        return type("Output",(),{"logits":logits})()


def fake_provider(model=None,device="cpu"):
    return HuggingFaceSegFormerCityscapesProvider("unused",device=device,input_size=(16,24),
        _model=model or FakeModel(),_processor=FakeProcessor())


def test_huggingface_provider_maps_checkpoint_labels_and_confidence():
    provider=fake_provider(); result=provider.segment(np.zeros((7,11,3),np.uint8))
    assert provider.device.type=="cpu"
    assert result.road_class_ids==frozenset({0}) and result.sidewalk_class_ids==frozenset({1})
    assert result.raw_class_names==("road","sidewalk","building")
    assert result.labels.shape==(7,11) and result.confidence.shape==(7,11)
    assert np.all(result.labels==1) and np.all(result.confidence>.9)


def test_huggingface_provider_cuda_request_falls_back_to_cpu(monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda,"is_available",lambda:False)
    assert fake_provider(device="cuda").device.type=="cpu"


def test_huggingface_provider_missing_model_directory():
    with pytest.raises(FileNotFoundError):
        HuggingFaceSegFormerCityscapesProvider("definitely-missing-model")


def test_huggingface_provider_rejects_missing_cityscapes_labels():
    model=FakeModel(); model.config=type("Config",(),{"id2label":{0:"background",1:"car"}})()
    with pytest.raises(ValueError,match="missing configured labels"):
        fake_provider(model)


def test_idd_provider_uses_label_names_not_indexes():
    model=FakeModel()
    model.config=type("Config",(),{"id2label":{
        0:"building",1:"non-drivable fallback",2:"parking",3:"sidewalk",
        4:"road",5:"drivable fallback",
    }})()
    provider=HuggingFaceSegFormerIDDProvider("unused",device="cpu",input_size=(16,24),
        _model=model,_processor=FakeProcessor())
    assert provider.road_class_ids==frozenset({2,4,5})
    assert provider.sidewalk_class_ids==frozenset({1,3})


def test_idd_provider_rejects_checkpoint_with_incomplete_hierarchy():
    model=FakeModel(); model.config=type("Config",(),{"id2label":{0:"road",1:"sidewalk"}})()
    with pytest.raises(ValueError,match="drivable fallback"):
        HuggingFaceSegFormerIDDProvider("unused",device="cpu",_model=model,_processor=FakeProcessor())


def test_huggingface_provider_rejects_invalid_model_output():
    class InvalidModel(FakeModel):
        def __call__(self,**kwargs): return type("Output",(),{"logits":None})()
    with pytest.raises(RuntimeError,match=r"\[1,classes,H,W\]"):
        fake_provider(InvalidModel()).segment(np.zeros((7,11,3),np.uint8))
