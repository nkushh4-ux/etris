import json

import cv2
import numpy as np
import pytest

from cv.alerts.surface_segmentation import SurfaceSegmentationResult
from scripts import benchmark_semantic_surface_models as benchmark_module


class FakeProvider:
    def __init__(self):
        import torch

        self.device=torch.device("cpu")

    def segment(self,frame):
        labels=np.asarray([[0,0,1],[2,1,2]],dtype=np.uint8)
        confidence=np.full(labels.shape,.8,dtype=np.float32)
        return SurfaceSegmentationResult(labels,confidence,("road","sidewalk","building"),
            frozenset({0}),frozenset({1}))


def test_benchmark_saves_masks_overlays_percentages_and_contacts(tmp_path,monkeypatch):
    monkeypatch.setattr(benchmark_module,"create_provider",lambda settings,device:FakeProvider())
    config={"models":[{"name":"fake","provider":"fake","model_path":"unused"}],
        "ground_contact_regions":[{"name":"vehicle","bbox":[0,0,2,2]}]}
    reports=benchmark_module.benchmark(np.zeros((2,3,3),np.uint8),config,tmp_path,"cpu")
    assert reports[0]["percentages"]==pytest.approx({"ROAD":100/3,"SIDEWALK":100/3,"OTHER":100/3})
    assert reports[0]["ground_contact_regions"][0]["surface_class"]=="ROAD"
    assert (tmp_path/"fake"/"raw_labels.png").is_file()
    assert (tmp_path/"fake"/"road_mask.png").is_file()
    assert (tmp_path/"fake"/"sidewalk_mask.png").is_file()
    assert (tmp_path/"fake"/"other_mask.png").is_file()
    assert (tmp_path/"fake"/"surface_overlay.jpg").is_file()
    assert json.loads((tmp_path/"fake"/"report.json").read_text())["device"]=="cpu"


def test_load_frame_accepts_image_and_rejects_invalid_region(tmp_path):
    image_path=tmp_path/"frame.png"; cv2.imwrite(str(image_path),np.zeros((4,5,3),np.uint8))
    assert benchmark_module.load_frame(image_path=image_path).shape==(4,5,3)
    result=FakeProvider().segment(np.zeros((2,3,3),np.uint8))
    try:
        benchmark_module.contact_report(result,[{"bbox":[2,1,1,0]}])
    except ValueError as error:
        assert "empty" in str(error)
    else:
        raise AssertionError("empty contact region was accepted")
