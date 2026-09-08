"""Benchmark the local OpenOCR SVTRv2-S model on the fixed 40 crops."""

import argparse
import copy
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
import yaml
from evaluate_ocr_benchmark import load_manifest, normalize_text

ROOT = Path(__file__).resolve().parents[1]

OPENOCR_ROOT = ROOT / "external" / "OpenOCR"

MODEL_CONFIG = (
    OPENOCR_ROOT
    / "weights"
    / "svtrv2_s"
    / "drive-download-20260904T165653Z-1-001"
    / "config_infer.yml"
)

CHECKPOINT = MODEL_CONFIG.parent / "best.pth"

MANIFEST = (
    ROOT
    / "data"
    / "ocr_benchmark"
    / "development_video"
    / "manifest.csv"
)

CROPS_DIR = (
    ROOT
    / "data"
    / "ocr_benchmark"
    / "development_video"
    / "crops"
)

OUTPUT = (
    ROOT
    / "runs"
    / "ocr_benchmark"
    / "svtrv2_results.csv"
)

OCR_BENCHMARK_CONFIG = ROOT / "configs" / "ocr_benchmark.yaml"

WARMUP_ITERATIONS = 3

FIELDS = (
    "sample_id",
    "raw_text",
    "normalized_text",
    "confidence",
    "latency_ms",
    "engine",
    "engine_version",
    "raw_output_json",
)


def checkpoint_identifier(path: Path) -> str:
    """Return a stable identifier for the exact checkpoint used."""
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)

    return f"svtrv2-s-best.pth-sha256:{digest.hexdigest()[:16]}"


def validate_inputs() -> tuple[list[dict], str]:
    """Validate benchmark files and load manifest/settings."""

    required_paths = (
        OPENOCR_ROOT,
        MODEL_CONFIG,
        CHECKPOINT,
        MANIFEST,
        CROPS_DIR,
        OCR_BENCHMARK_CONFIG,
    )

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"Required path does not exist: {path}")

    with OCR_BENCHMARK_CONFIG.open(encoding="utf-8") as stream:
        benchmark_config = yaml.safe_load(stream)

    separators = benchmark_config["normalization"]["separators"]

    samples = load_manifest(MANIFEST, CROPS_DIR)

    if len(samples) != 40:
        raise RuntimeError(
            f"Expected 40 benchmark samples, found {len(samples)}"
        )

    return samples, separators


def load_recognizer():
    """Construct OpenOCR once using the supplied config and checkpoint."""

    if str(OPENOCR_ROOT) not in sys.path:
        sys.path.insert(0, str(OPENOCR_ROOT))

    from tools.engine.config import Config
    from tools.infer_rec import OpenRecognizer

    config = copy.deepcopy(Config(str(MODEL_CONFIG)).cfg)

    dictionary = (
        OPENOCR_ROOT
        / "tools"
        / "utils"
        / "EN_symbol_dict.txt"
    )

    # Fail fast if the inference configuration is wrong.
    infer_gtc = config["Architecture"]["Decoder"].get("infer_gtc")
    postprocess_name = config["PostProcess"].get("name")

    if infer_gtc is not False:
        raise RuntimeError(
            "Invalid SVTRv2 inference config: "
            f"Architecture.Decoder.infer_gtc={infer_gtc!r}. "
            "Expected False."
        )

    if postprocess_name != "CTCLabelDecode":
        raise RuntimeError(
            "Invalid SVTRv2 inference config: "
            f"PostProcess.name={postprocess_name!r}. "
            "Expected 'CTCLabelDecode'."
        )

    config["Global"]["pretrained_model"] = str(CHECKPOINT)
    config["Global"]["checkpoints"] = None
    config["Global"]["distributed"] = False
    config["Global"]["character_dict_path"] = str(dictionary)

    config["PostProcess"]["character_dict_path"] = str(dictionary)

    recognizer = OpenRecognizer(
        config=config,
        backend="torch",
        use_gpu="auto",
    )

    return recognizer


def synchronize_if_cuda(recognizer) -> None:
    """Synchronize CUDA so latency measurements are accurate."""

    device = getattr(recognizer, "device", None)

    if device is not None and device.type == "cuda":
        torch.cuda.synchronize(device)


def recognize_one(recognizer, crop_path: Path) -> dict:
    """Run recognition on exactly one crop using OpenOCR's path interface."""

    if not crop_path.exists():
        raise FileNotFoundError(f"Crop does not exist: {crop_path}")

    result = recognizer(
        img_path=str(crop_path),
        batch_num=1,
    )

    if len(result) != 1:
        raise RuntimeError(
            f"Expected exactly one OCR result for {crop_path.name}, "
            f"received {len(result)}"
        )

    prediction = result[0]

    if not isinstance(prediction, dict):
        raise TypeError(
            f"Expected OCR result dict, received "
            f"{type(prediction).__name__}: {prediction!r}"
        )

    return prediction


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate benchmark inputs without loading/running the OCR model.",
    )

    args = parser.parse_args()

    samples, separators = validate_inputs()

    if args.validate_only:
        print(
            f"Validated SVTRv2 benchmark inputs "
            f"for {len(samples)} crops"
        )
        return

    recognizer = load_recognizer()

    print(f"Device: {recognizer.device}")
    print(f"Config: {MODEL_CONFIG}")
    print(f"Checkpoint: {CHECKPOINT}")
    print(f"Samples: {len(samples)}")

    #
    # Warm-up
    #
    # IMPORTANT:
    # OpenOCR's working SVTRv2 interface uses img_path.
    # Do not pass NumPy arrays through img_numpy here.
    #

    warmup_path = Path(samples[0]["crop_path"])

    print(
        f"Warming up SVTRv2-S "
        f"for {WARMUP_ITERATIONS} iterations..."
    )

    for _ in range(WARMUP_ITERATIONS):
        recognize_one(
            recognizer=recognizer,
            crop_path=warmup_path,
        )

    synchronize_if_cuda(recognizer)

    identifier = checkpoint_identifier(CHECKPOINT)

    rows = []

    print("Running benchmark...")

    for index, sample in enumerate(samples, start=1):
        crop_path = Path(sample["crop_path"])

        synchronize_if_cuda(recognizer)

        start = time.perf_counter()

        prediction = recognize_one(
            recognizer=recognizer,
            crop_path=crop_path,
        )

        synchronize_if_cuda(recognizer)

        latency_ms = (
            time.perf_counter() - start
        ) * 1000.0

        raw_text = str(
            prediction.get("text", "")
        )

        confidence = float(
            prediction.get("score", 0.0)
        )

        normalized_text = normalize_text(
            raw_text,
            separators,
        )

        rows.append(
            {
                "sample_id": sample["sample_id"],
                "raw_text": raw_text,
                "normalized_text": normalized_text,
                "confidence": f"{confidence:.6f}",
                "latency_ms": f"{latency_ms:.3f}",
                "engine": "openocr_svtrv2_s",
                "engine_version": identifier,
                "raw_output_json": json.dumps(
                    {
                        "text": raw_text,
                        "score": confidence,
                        "model_identifier": identifier,
                        "device": str(recognizer.device),
                        "crop_path": str(crop_path),
                    },
                    ensure_ascii=False,
                ),
            }
        )

        print(
            f"[{index:02d}/{len(samples)}] "
            f"{sample['sample_id']} -> "
            f"{raw_text!r} "
            f"conf={confidence:.4f} "
            f"latency={latency_ms:.2f} ms"
        )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=FIELDS,
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print(
        f"Wrote {len(rows)} SVTRv2 results to:"
    )
    print(OUTPUT)


if __name__ == "__main__":
    main()