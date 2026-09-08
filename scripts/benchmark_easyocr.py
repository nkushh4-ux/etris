"""Run EasyOCR sequentially on the fixed development OCR manifest."""

import argparse
import csv
import importlib.metadata
import json
import time
from pathlib import Path

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config.resolve())
    manifest = (ROOT / config["manifest"]).resolve()
    crops_dir = (ROOT / config["crops_dir"]).resolve()
    samples = load_manifest(manifest, crops_dir)
    if args.validate_only:
        print(f"Validated {len(samples)} manifest crops for EasyOCR")
        return

    import easyocr

    engine_config = config["easyocr"]
    reader = easyocr.Reader(
        engine_config["languages"],
        gpu=engine_config["gpu"],
    )
    version = importlib.metadata.version("easyocr")
    separators = config["normalization"]["separators"]
    allowlist = config["recognition"]["allowlist"]
    rows = []
    for sample in samples:
        start = time.perf_counter()
        result = reader.readtext(
            str(sample["crop_path"]),
            detail=1,
            paragraph=False,
            batch_size=1,
            workers=0,
            allowlist=allowlist,
        )
        latency_ms = (time.perf_counter() - start) * 1000.0
        ordered = sorted(
            result,
            key=lambda item: (
                sum(point[1] for point in item[0]) / len(item[0]),
                sum(point[0] for point in item[0]) / len(item[0]),
            ),
        )
        raw_text = " ".join(str(item[1]) for item in ordered)
        confidences = [float(item[2]) for item in ordered]
        confidence = sum(confidences) / len(confidences) if confidences else 0.0
        rows.append({
            "sample_id": sample["sample_id"],
            "raw_text": raw_text,
            "normalized_text": normalize_text(raw_text, separators),
            "confidence": f"{confidence:.6f}",
            "latency_ms": f"{latency_ms:.3f}",
            "engine": "easyocr",
            "engine_version": version,
            "raw_output_json": json.dumps(result, ensure_ascii=False, default=str),
        })

    output = (ROOT / config["output_dir"] / engine_config["output"]).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} EasyOCR results to {output}")


if __name__ == "__main__":
    main()
