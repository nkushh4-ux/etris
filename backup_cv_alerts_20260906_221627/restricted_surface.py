from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

from cv.alerts.parking_state import RestrictedParkingZone
from cv.alerts.surface_segmentation import SurfaceSegmentationResult


class RestrictedSurfaceMode(str,Enum): SEMANTIC="SEMANTIC"; POLICY_DEFINED="POLICY_DEFINED"; HYBRID="HYBRID"


def policy_mask(zone:RestrictedParkingZone,width:int,height:int):
    mask=np.zeros((height,width),np.uint8)
    points=np.asarray([(round(x*(width-1)),round(y*(height-1))) for x,y in zone.polygon],np.int32)
    cv2.fillPoly(mask,[points],1); return mask.astype(bool)


@dataclass(frozen=True,slots=True)
class RestrictedSurfaceEvidence:
    occupied:bool; uncertain:bool; restricted_overlap:float; surface_confidence:float
    surface_class:str; raw_surface_class:str|None; reason_codes:tuple[str,...]


class RestrictedSurfaceResolver:
    def __init__(self,mode:RestrictedSurfaceMode,restricted_classes=("SIDEWALK",),minimum_surface_confidence=.55):
        self.mode=mode; self.restricted_classes=frozenset(x.upper() for x in restricted_classes); self.minimum_surface_confidence=minimum_surface_confidence

    def resolve(self,zone,width,height,semantic:SurfaceSegmentationResult|None):
        policy=policy_mask(zone,width,height)
        if self.mode is RestrictedSurfaceMode.POLICY_DEFINED: return policy
        if semantic is None: return None
        semantic_mask=np.zeros((height,width),bool)
        if "SIDEWALK" in self.restricted_classes: semantic_mask|=semantic.sidewalk_mask
        if "ROAD" in self.restricted_classes: semantic_mask|=semantic.road_mask
        return semantic_mask if self.mode is RestrictedSurfaceMode.SEMANTIC else semantic_mask&policy

    def evaluate(self,bbox,restricted,semantic,*,width,height,ground_contact_fraction,minimum_overlap):
        x1=max(0,int(bbox.x1)); x2=min(width,int(np.ceil(bbox.x2))); y2=min(height,int(np.ceil(bbox.y2)))
        bbox_height=bbox.y2-bbox.y1; y1=max(0,int(np.floor(bbox.y2-bbox_height*ground_contact_fraction)))
        clipped=bbox.x1<0 or bbox.y1<0 or bbox.x2>width or bbox.y2>height
        if restricted is None: return RestrictedSurfaceEvidence(False,True,0.,0.,"UNKNOWN",None,("SEMANTIC_MASK_UNAVAILABLE",))
        if x2<=x1 or y2<=y1 or (x2-x1)*(y2-y1)<9 or clipped:
            return RestrictedSurfaceEvidence(False,True,0.,0.,"UNKNOWN",None,("INVALID_GROUND_CONTACT_REGION",))
        region=restricted[y1:y2,x1:x2]; overlap=float(region.mean())
        surface_confidence=float(semantic.confidence[y1:y2,x1:x2][region].mean()) if semantic is not None and region.any() else (1. if self.mode is RestrictedSurfaceMode.POLICY_DEFINED else 0.)
        near=minimum_overlap*.8<=overlap<minimum_overlap
        low_surface=self.mode is not RestrictedSurfaceMode.POLICY_DEFINED and surface_confidence<self.minimum_surface_confidence
        reasons=[]
        if overlap>=minimum_overlap: reasons.append("ON_RESTRICTED_SURFACE")
        else: reasons.append("INSUFFICIENT_OVERLAP")
        if low_surface: reasons.append("LOW_SEGMENTATION_CONFIDENCE")
        if near: reasons.append("OVERLAP_AMBIGUOUS")
        occupied=overlap>=minimum_overlap and not low_surface
        raw="sidewalk" if occupied and "SIDEWALK" in self.restricted_classes else None
        return RestrictedSurfaceEvidence(occupied,low_surface or near,overlap,surface_confidence,"SIDEWALK" if raw else "POLICY_DEFINED",raw,tuple(reasons))
