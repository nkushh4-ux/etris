import copy
import hashlib
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from cv.ocr.engine import OCREngine
from cv.ocr.models import OCRResult, OCRResultStatus
from cv.ocr.normalization import DEFAULT_SEPARATORS, normalize_plate_text
from cv.plate.models import PlateCandidate

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True, slots=True)
class SVTRv2Config:
    config_path: Path
    checkpoint_path: Path
    openocr_root: Path
    device: str = "auto"
    separators: str = DEFAULT_SEPARATORS

    def __post_init__(self) -> None:
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be one of: auto, cpu, cuda")

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "SVTRv2Config":
        def resolve(value: str) -> Path:
            path = Path(value)
            return path if path.is_absolute() else PROJECT_ROOT / path

        return cls(
            config_path=resolve(values["config_path"]),
            checkpoint_path=resolve(values["checkpoint_path"]),
            openocr_root=resolve(values.get("openocr_root", "external/OpenOCR")),
            device=values.get("device", "auto"),
            separators=values.get("separators", DEFAULT_SEPARATORS),
        )

    def validate_paths(self) -> None:
        for path in (self.openocr_root, self.config_path, self.checkpoint_path):
            if not path.exists():
                raise FileNotFoundError(f"Required SVTRv2 path does not exist: {path}")


class SVTRv2Engine(OCREngine):
    """In-memory OpenOCR SVTRv2-S adapter with one model load per instance."""

    engine_name = "openocr_svtrv2_s"

    def __init__(
        self,
        config: SVTRv2Config,
        *,
        backend_factory: Callable[[SVTRv2Config], Any] | None = None,
        model_identifier: str | None = None,
    ) -> None:
        config.validate_paths()
        self.config = config
        self.model_identifier = model_identifier or self._checkpoint_identifier()
        self._recognizer = (backend_factory or self._load_recognizer)(config)
        self._inference_lock = threading.Lock()

    def _checkpoint_identifier(self) -> str:
        digest = hashlib.sha256()
        with self.config.checkpoint_path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return f"svtrv2-s:{digest.hexdigest()[:16]}"

    @staticmethod
    def _load_recognizer(config: SVTRv2Config) -> Any:
        root = str(config.openocr_root)
        if root not in sys.path:
            sys.path.insert(0, root)
        # Vendor imports remain isolated from production interfaces.
        from tools.engine.config import Config
        from tools.infer_rec import OpenRecognizer

        values = copy.deepcopy(Config(str(config.config_path)).cfg)
        dictionary = config.openocr_root / "tools" / "utils" / "EN_symbol_dict.txt"
        values["Global"]["pretrained_model"] = str(config.checkpoint_path)
        values["Global"]["checkpoints"] = None
        values["Global"]["distributed"] = False
        values["Global"]["character_dict_path"] = str(dictionary)
        values["PostProcess"]["character_dict_path"] = str(dictionary)
        use_gpu = {"auto": "auto", "cpu": "false", "cuda": "true"}[config.device]
        return OpenRecognizer(config=values, backend="torch", use_gpu=use_gpu)

    def recognize(self, candidate: PlateCandidate) -> OCRResult:
        common = dict(
            camera_id=candidate.camera_id,
            track_id=candidate.track_id,
            frame_index=candidate.frame_index,
            timestamp=candidate.timestamp,
            engine=self.engine_name,
            engine_version=self.model_identifier,
            evidence_id=(
                f"{candidate.camera_id}:{candidate.track_id}:"
                f"{candidate.frame_index}:{self.engine_name}:{self.model_identifier}"
            ),
            plate_quality_score=candidate.quality_score,
            detector_confidence=candidate.detector_confidence,
            bbox=(candidate.bbox.x1, candidate.bbox.y1, candidate.bbox.x2, candidate.bbox.y2),
        )
        crop = candidate.crop
        if not isinstance(crop, np.ndarray) or crop.size == 0 or crop.ndim not in (2, 3):
            return OCRResult(raw_text="", normalized_text="", confidence=0.0,
                             status=OCRResultStatus.REJECTED,
                             rejected_reason="invalid_crop", **common)
        try:
            # This checkout skips DecodeImagePIL for img_numpy and starts at
            # RecTVResize, whose input contract is a PIL image (`w, h =
            # img.size`). A raw ndarray therefore is not a valid in-memory
            # input even though OpenRecognizer names the argument img_numpy.
            if crop.ndim == 2:
                image = Image.fromarray(crop).convert("RGB")
            elif crop.shape[2] == 3:
                # Stage 3 crops originate in OpenCV and are BGR.
                image = Image.fromarray(crop[:, :, ::-1]).convert("RGB")
            elif crop.shape[2] == 4:
                image = Image.fromarray(crop[:, :, [2, 1, 0, 3]]).convert("RGB")
            else:
                return OCRResult(raw_text="", normalized_text="", confidence=0.0,
                                 status=OCRResultStatus.REJECTED,
                                 rejected_reason="invalid_crop_channels", **common)
            with self._inference_lock:
                output = self._recognizer(img_numpy=image, batch_num=1)
            if len(output) != 1 or not isinstance(output[0], dict):
                raise ValueError("unexpected OpenOCR result shape")
            raw_text = str(output[0].get("text", ""))
            confidence = max(0.0, min(1.0, float(output[0].get("score", 0.0))))
            normalized = normalize_plate_text(raw_text, self.config.separators)
            if not normalized:
                return OCRResult(raw_text=raw_text, normalized_text="", confidence=confidence,
                                 status=OCRResultStatus.REJECTED,
                                 rejected_reason="empty_ocr_result", **common)
            return OCRResult(raw_text=raw_text, normalized_text=normalized,
                             confidence=confidence, **common)
        except Exception as exc:
            return OCRResult(raw_text="", normalized_text="", confidence=0.0,
                             status=OCRResultStatus.FAILED,
                             rejected_reason=f"model_inference_error:{type(exc).__name__}",
                             **common)
