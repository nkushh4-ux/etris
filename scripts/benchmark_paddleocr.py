"""Run PaddleOCR sequentially on the fixed development OCR manifest."""

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


def result_payload(result) -> dict:
    payload = result.json
    if callable(payload):
        payload = payload()
    if not isinstance(payload, dict):
        raise TypeError("PaddleOCR result.json did not return a dictionary")
    return payload.get("res", payload)


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
        print(f"Validated {len(samples)} manifest crops for PaddleOCR")
        return

    from paddleocr import PaddleOCR

    engine_config = config["paddleocr"]
    engine = PaddleOCR(
        lang=engine_config["language"],
        device=engine_config["device"],
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        enable_mkldnn=False,
    )
    version = importlib.metadata.version("paddleocr")
    separators = config["normalization"]["separators"]
    rows = []
    for sample in samples:
        start = time.perf_counter()
        results = list(engine.predict(input=str(sample["crop_path"])))
        latency_ms = (time.perf_counter() - start) * 1000.0
        payloads = [result_payload(result) for result in results]
        texts = [str(text) for payload in payloads for text in payload.get("rec_texts", [])]
        scores = [float(score) for payload in payloads for score in payload.get("rec_scores", [])]
        raw_text = " ".join(texts)
        confidence = sum(scores) / len(scores) if scores else 0.0
        rows.append({
            "sample_id": sample["sample_id"],
            "raw_text": raw_text,
            "normalized_text": normalize_text(raw_text, separators),
            "confidence": f"{confidence:.6f}",
            "latency_ms": f"{latency_ms:.3f}",
            "engine": "paddleocr",
            "engine_version": version,
            "raw_output_json": json.dumps(payloads, ensure_ascii=False, default=str),
        })

    output = (ROOT / config["output_dir"] / engine_config["output"]).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} PaddleOCR results to {output}")


if __name__ == "__main__":
    main()
