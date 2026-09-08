"""Run local Awiros ANPR-OCR sequentially on the fixed benchmark manifest."""

import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
from evaluate_ocr_benchmark import load_manifest, normalize_text

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "ocr_benchmark.yaml"
FIELDS = (
    "sample_id", "raw_text", "normalized_text", "confidence",
    "latency_ms", "engine", "engine_version", "raw_output_json",
)


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def load_awiros_source(source_dir: Path):
    source = source_dir / "test.py"
    spec = importlib.util.spec_from_file_location("awiros_anpr_test", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load Awiros source: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def model_identifier(weights: Path) -> str:
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()[:16]
    return f"model.safetensors-sha256:{digest}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config.resolve())
    manifest = (ROOT / config["manifest"]).resolve()
    crops_dir = (ROOT / config["crops_dir"]).resolve()
    samples = load_manifest(manifest, crops_dir)
    awiros_config = config["awiros"]
    source_dir = (ROOT / awiros_config["source_dir"]).resolve()
    weights = (ROOT / awiros_config["weights"]).resolve()
    dictionary = (ROOT / awiros_config["dictionary"]).resolve()
    paddleocr_dir = (ROOT / awiros_config["paddleocr_dir"]).resolve()
    required = (source_dir / "test.py", weights, dictionary,
                paddleocr_dir / "ppocr" / "__init__.py")
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.validate_only:
        print(f"Validated {len(samples)} crops and local Awiros artifacts")
        return

    awiros = load_awiros_source(source_dir)
    sys.path.insert(0, str(paddleocr_dir))
    import paddle
    from ppocr.modeling.architectures import build_model
    from ppocr.postprocess import build_post_process

    requested_device = awiros_config["device"]
    device = requested_device
    if device == "gpu" and not paddle.is_compiled_with_cuda():
        device = "cpu"
    paddle.set_device(device)
    post_process = build_post_process({
        "name": "CTCLabelDecode",
        "character_dict_path": str(dictionary),
        "use_space_char": True,
    })
    architecture = copy.deepcopy(awiros.MODEL_CONFIG)["Architecture"]
    model = build_model(architecture)
    model.set_state_dict(awiros.load_safetensors_to_paddle(paddle, str(weights)))
    model.eval()
    identifier = model_identifier(weights)
    separators = config["normalization"]["separators"]

    rows = []
    for sample in samples:
        start = time.perf_counter()
        image = cv2.imread(str(sample["crop_path"]))
        if image is None:
            raise ValueError(f"Unable to read crop: {sample['crop_path']}")
        tensor = paddle.to_tensor(np.expand_dims(
            awiros.preprocess(image, awiros.IMAGE_SHAPE), axis=0
        ))
        with paddle.no_grad():
            predictions = model(tensor)
        if isinstance(predictions, dict):
            prediction = predictions.get("ctc", next(iter(predictions.values())))
        elif isinstance(predictions, (list, tuple)):
            prediction = predictions[0]
        else:
            prediction = predictions
        decoded = post_process(prediction.numpy())
        if isinstance(decoded, (list, tuple)) and decoded:
            text, confidence = decoded[0]
            raw_text = str(text)
            confidence = float(confidence)
        else:
            raw_text, confidence = "", 0.0
        latency_ms = (time.perf_counter() - start) * 1000.0
        rows.append({
            "sample_id": sample["sample_id"],
            "raw_text": raw_text,
            "normalized_text": normalize_text(raw_text, separators),
            "confidence": f"{confidence:.6f}",
            "latency_ms": f"{latency_ms:.3f}",
            "engine": "awiros_anpr_ocr",
            "engine_version": identifier,
            "raw_output_json": json.dumps({
                "prediction": raw_text,
                "confidence": confidence,
                "model_identifier": identifier,
                "device": device,
            }),
        })

    output = (ROOT / config["output_dir"] / awiros_config["output"]).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} Awiros results to {output}")


if __name__ == "__main__":
    main()
