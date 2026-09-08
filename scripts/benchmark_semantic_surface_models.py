"""Benchmark local semantic-surface providers without changing alert policy."""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

from cv.alerts.surface_segmentation import (
    HuggingFaceSegFormerCityscapesProvider,
    HuggingFaceSegFormerIDDProvider,
)

PROVIDERS={
    "huggingface_segformer_cityscapes":HuggingFaceSegFormerCityscapesProvider,
    "huggingface_segformer_idd":HuggingFaceSegFormerIDDProvider,
}
COLORS={"ROAD":(255,120,30),"SIDEWALK":(220,70,220),"OTHER":(70,70,70)}


def options():
    parser=argparse.ArgumentParser()
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image"); source.add_argument("--video")
    parser.add_argument("--frame-index",type=int,default=0)
    parser.add_argument("--config",required=True)
    parser.add_argument("--output-dir",default=str(ROOT/"runs/semantic_model_benchmark"))
    parser.add_argument("--device",default="cuda")
    return parser.parse_args()


def load_frame(image_path=None,video_path=None,frame_index=0):
    if frame_index<0: raise ValueError("frame_index must be non-negative")
    if image_path:
        frame=cv2.imread(str(image_path))
    else:
        capture=cv2.VideoCapture(str(video_path)); capture.set(cv2.CAP_PROP_POS_FRAMES,frame_index)
        ok,frame=capture.read(); capture.release()
        if not ok: frame=None
    if frame is None: raise RuntimeError("unable to read benchmark image/video frame")
    return frame


def create_provider(settings,device):
    kind=settings["provider"]
    if kind not in PROVIDERS: raise ValueError(f"unsupported semantic provider: {kind}")
    weights=Path(settings["model_path"])
    if not weights.is_absolute(): weights=ROOT/weights
    kwargs={"device":device,"input_size":tuple(settings.get("input_size",(512,512)))}
    if kind=="huggingface_segformer_idd":
        if "road_labels" in settings: kwargs["road_labels"]=tuple(settings["road_labels"])
        if "sidewalk_labels" in settings: kwargs["sidewalk_labels"]=tuple(settings["sidewalk_labels"])
    return PROVIDERS[kind](weights,**kwargs)


def mapped_masks(result):
    return {"ROAD":result.road_mask,"SIDEWALK":result.sidewalk_mask,"OTHER":result.other_mask}


def contact_report(result,regions):
    height,width=result.labels.shape; output=[]; masks=mapped_masks(result)
    for item in regions:
        coordinates=np.asarray(item["bbox"],dtype=float)
        if coordinates.shape!=(4,): raise ValueError("ground-contact bbox must contain four values")
        if item.get("normalized",False): coordinates*=np.asarray((width,height,width,height))
        x1,y1,x2,y2=(int(value) for value in coordinates)
        x1=max(0,x1); y1=max(0,y1); x2=min(width,x2); y2=min(height,y2)
        if x2<=x1 or y2<=y1: raise ValueError("ground-contact bbox is empty after clipping")
        shares={name:float(mask[y1:y2,x1:x2].mean()*100) for name,mask in masks.items()}
        output.append({"name":item.get("name",f"region_{len(output)+1}"),"bbox":[x1,y1,x2,y2],
            "percentages":shares,"surface_class":max(shares,key=shares.get),
            "mean_confidence":float(result.confidence[y1:y2,x1:x2].mean())})
    return output


def save_outputs(frame,result,target):
    target.mkdir(parents=True,exist_ok=True); masks=mapped_masks(result)
    cv2.imwrite(str(target/"raw_labels.png"),result.labels.astype(np.uint16))
    overlay=frame.copy()
    for name,mask in masks.items():
        cv2.imwrite(str(target/f"{name.lower()}_mask.png"),mask.astype(np.uint8)*255)
        layer=np.zeros_like(frame); layer[mask]=COLORS[name]
        overlay=cv2.addWeighted(layer,.28,overlay,1,0)
    cv2.imwrite(str(target/"surface_overlay.jpg"),overlay)


def benchmark(frame,config,output_dir,device):
    import torch

    reports=[]
    for settings in config["models"]:
        if not settings.get("enabled",True): continue
        provider=create_provider(settings,device)
        if provider.device.type=="cuda":
            torch.cuda.reset_peak_memory_stats(provider.device); torch.cuda.synchronize(provider.device)
        started=time.perf_counter(); result=provider.segment(frame)
        if provider.device.type=="cuda": torch.cuda.synchronize(provider.device)
        elapsed_ms=(time.perf_counter()-started)*1000
        masks=mapped_masks(result); target=output_dir/settings["name"]; save_outputs(frame,result,target)
        report={"name":settings["name"],"provider":settings["provider"],"device":provider.device.type,
            "model_path":settings["model_path"],"inference_latency_ms":elapsed_ms,
            "cuda_peak_allocated_mib":float(torch.cuda.max_memory_allocated(provider.device)/1048576) if provider.device.type=="cuda" else 0.,
            "percentages":{name:float(mask.mean()*100) for name,mask in masks.items()},
            "ground_contact_regions":contact_report(result,config.get("ground_contact_regions",[]))}
        (target/"report.json").write_text(json.dumps(report,indent=2),encoding="utf-8"); reports.append(report)
        del provider
    return reports


def main():
    args=options(); config=json.loads(Path(args.config).read_text(encoding="utf-8"))
    frame=load_frame(args.image,args.video,args.frame_index); output=Path(args.output_dir)
    reports=benchmark(frame,config,output,args.device)
    output.mkdir(parents=True,exist_ok=True)
    (output/"benchmark_report.json").write_text(json.dumps(reports,indent=2),encoding="utf-8")
    print(json.dumps(reports,indent=2))


if __name__=="__main__": main()
