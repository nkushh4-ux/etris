"""Replaceable semantic-surface providers for Stage 12A v2."""
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class SurfaceSegmentationResult:
    labels: np.ndarray
    confidence: np.ndarray
    raw_class_names: tuple[str, ...]
    road_class_ids: frozenset[int]
    sidewalk_class_ids: frozenset[int]

    def __post_init__(self):
        if self.labels.ndim != 2 or self.confidence.shape != self.labels.shape:
            raise ValueError("semantic labels and confidence must be equally sized 2D arrays")

    @property
    def road_mask(self): return np.isin(self.labels,tuple(self.road_class_ids))

    @property
    def sidewalk_mask(self): return np.isin(self.labels,tuple(self.sidewalk_class_ids))

    @property
    def other_mask(self): return ~(self.road_mask|self.sidewalk_mask)


class SurfaceSegmentationProvider(Protocol):
    def segment(self,frame:np.ndarray)->SurfaceSegmentationResult: ...


CITYSCAPES_TRAIN_CLASSES=("road","sidewalk","building","wall","fence","pole","traffic light",
    "traffic sign","vegetation","terrain","sky","person","rider","car","truck","bus","train","motorcycle","bicycle")


class TorchScriptCityscapesProvider:
    """Runs a deployed TorchScript model producing Cityscapes 19-class logits."""
    def __init__(self,weights:str|Path,*,device="cuda",input_size=(512,1024)):
        import torch
        self.weights=Path(weights); self.device=torch.device(device if str(device).startswith("cuda") and torch.cuda.is_available() else "cpu")
        if not self.weights.is_file(): raise FileNotFoundError(f"Cityscapes TorchScript weights not found: {self.weights}")
        self.input_size=tuple(input_size); self.model=torch.jit.load(str(self.weights),map_location=self.device).eval()

    def segment(self,frame):
        import torch
        height,width=frame.shape[:2]; resized=cv2.resize(frame,(self.input_size[1],self.input_size[0]))
        rgb=cv2.cvtColor(resized,cv2.COLOR_BGR2RGB).astype(np.float32)/255
        tensor=torch.from_numpy(rgb.transpose(2,0,1)).unsqueeze(0).to(self.device)
        mean=torch.tensor((.485,.456,.406),device=self.device).view(1,3,1,1); std=torch.tensor((.229,.224,.225),device=self.device).view(1,3,1,1)
        with torch.inference_mode(): output=self.model((tensor-mean)/std)
        if isinstance(output,dict): output=output.get("out")
        if isinstance(output,(tuple,list)): output=output[0]
        if output is None or output.ndim!=4 or output.shape[1]!=19: raise RuntimeError("semantic model must return [1,19,H,W] Cityscapes logits")
        probabilities=output.softmax(1); confidence,labels=probabilities.max(1)
        labels=cv2.resize(labels[0].byte().cpu().numpy(),(width,height),interpolation=cv2.INTER_NEAREST)
        confidence=cv2.resize(confidence[0].float().cpu().numpy(),(width,height),interpolation=cv2.INTER_LINEAR)
        return SurfaceSegmentationResult(labels,confidence,CITYSCAPES_TRAIN_CLASSES,frozenset({0}),frozenset({1}))


class HuggingFaceSegFormerCityscapesProvider:
    """Offline Hugging Face SegFormer provider using checkpoint label metadata."""

    def __init__(self,weights:str|Path,*,device="cuda",input_size=(512,512),_model=None,_processor=None):
        import torch

        self.weights=Path(weights)
        if _model is None and not self.weights.is_dir():
            raise FileNotFoundError(f"SegFormer model directory not found: {self.weights}")
        self.device=torch.device(device if str(device).startswith("cuda") and torch.cuda.is_available() else "cpu")
        self.input_size=tuple(input_size)
        if len(self.input_size)!=2 or any(not isinstance(x,int) or x<=0 for x in self.input_size):
            raise ValueError("input_size must contain two positive integers")
        if _model is None or _processor is None:
            from transformers import (
                SegformerForSemanticSegmentation,
                SegformerImageProcessor,
            )

            _processor=SegformerImageProcessor.from_pretrained(self.weights,local_files_only=True)
            _model=SegformerForSemanticSegmentation.from_pretrained(self.weights,local_files_only=True)
        self.processor=_processor
        self.model=_model.to(self.device).eval()
        id2label={int(index):str(name) for index,name in self.model.config.id2label.items()}
        self.raw_class_names=tuple(id2label.get(index,f"class_{index}") for index in range(max(id2label)+1))
        by_name={name.strip().lower():index for index,name in id2label.items()}
        if "road" not in by_name or "sidewalk" not in by_name:
            raise ValueError("SegFormer checkpoint must define road and sidewalk labels")
        self.road_class_ids=frozenset({by_name["road"]})
        self.sidewalk_class_ids=frozenset({by_name["sidewalk"]})
        self.last_inference_ms=0.

    def segment(self,frame):
        import torch
        from torch.nn import functional

        if frame.ndim!=3 or frame.shape[2]!=3:
            raise ValueError("semantic input must be a BGR HxWx3 frame")
        height,width=frame.shape[:2]
        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        inputs=self.processor(images=rgb,return_tensors="pt",do_resize=True,
            size={"height":self.input_size[0],"width":self.input_size[1]})
        pixel_values=inputs["pixel_values"].to(self.device)
        started=time.perf_counter()
        with torch.inference_mode():
            output=self.model(pixel_values=pixel_values)
            logits=output.logits if hasattr(output,"logits") else output[0] if isinstance(output,(tuple,list)) else None
            if logits is None or logits.ndim!=4 or logits.shape[0]!=1 or logits.shape[1]!=len(self.raw_class_names):
                raise RuntimeError("SegFormer model must return [1,classes,H,W] logits")
            logits=functional.interpolate(logits,size=(height,width),mode="bilinear",align_corners=False)
            confidence,labels=logits.softmax(1).max(1)
        if self.device.type=="cuda": torch.cuda.synchronize(self.device)
        self.last_inference_ms=(time.perf_counter()-started)*1000
        return SurfaceSegmentationResult(labels[0].byte().cpu().numpy(),confidence[0].float().cpu().numpy(),
            self.raw_class_names,self.road_class_ids,self.sidewalk_class_ids)
