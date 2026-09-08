"""Evaluate fixed-manifest OCR results, including content-hash deduplication."""

import argparse
import csv
import hashlib
import statistics
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "ocr_benchmark.yaml"


def normalize_text(text: str, separators: str) -> str:
    """Uppercase and remove separators only; never substitute characters."""

    translation = str.maketrans("", "", separators)
    return text.upper().translate(translation)


def edit_distance(first: str, second: str) -> int:
    previous = list(range(len(second) + 1))
    for row, left in enumerate(first, 1):
        current = [row]
        for column, right in enumerate(second, 1):
            current.append(min(
                current[-1] + 1,
                previous[column] + 1,
                previous[column - 1] + (left != right),
            ))
        previous = current
    return previous[-1]


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: Path, crops_dir: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 40:
        raise ValueError(f"Expected exactly 40 manifest samples, found {len(rows)}")
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("Manifest sample_id values must be unique")
    for row in rows:
        if row["readability"] not in {"readable", "not_a_plate"}:
            raise ValueError(f"Invalid readability for {row['sample_id']}")
        if row["readability"] == "not_a_plate" and row["gt_text"].strip():
            raise ValueError(f"not_a_plate GT must be blank: {row['sample_id']}")
        if row["readability"] == "readable" and not row["gt_text"].strip():
            raise ValueError(f"readable GT must not be blank: {row['sample_id']}")
        row["crop_path"] = crops_dir / f"{row['sample_id']}.jpg"
        if not row["crop_path"].is_file():
            raise FileNotFoundError(row["crop_path"])
        row["content_hash"] = file_hash(row["crop_path"])
    return rows


def load_results(path: Path, sample_ids: set[str]) -> dict[str, dict]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    ids = [row["sample_id"] for row in rows]
    if len(rows) != 40 or set(ids) != sample_ids or len(set(ids)) != 40:
        raise ValueError(f"Results must contain exactly the 40 manifest samples: {path}")
    return {row["sample_id"]: row for row in rows}


def calculate_metrics(samples: list[dict], results: dict[str, dict], separators: str) -> dict:
    readable = [sample for sample in samples if sample["readability"] == "readable"]
    negatives = [sample for sample in samples if sample["readability"] == "not_a_plate"]
    distances = []
    normalized_distances = []
    exact = 0
    no_reads = 0
    for sample in readable:
        gt = normalize_text(sample["gt_text"], separators)
        prediction = normalize_text(results[sample["sample_id"]]["normalized_text"], separators)
        distance = edit_distance(gt, prediction)
        denominator = max(len(gt), len(prediction), 1)
        distances.append((distance, denominator))
        normalized_distances.append(distance / denominator)
        exact += prediction == gt
        no_reads += not prediction
    false_reads = sum(
        bool(normalize_text(results[sample["sample_id"]]["normalized_text"], separators))
        for sample in negatives
    )
    latencies = [float(results[sample["sample_id"]]["latency_ms"]) for sample in samples]
    confidences = [float(results[sample["sample_id"]]["confidence"]) for sample in samples]
    return {
        "samples": len(samples),
        "readable_samples": len(readable),
        "not_a_plate_samples": len(negatives),
        "exact_accuracy": exact / len(readable) if readable else None,
        "character_accuracy": 1.0 - sum(x for x, _ in distances) / sum(y for _, y in distances) if distances else None,
        "normalized_edit_distance": statistics.mean(normalized_distances) if normalized_distances else None,
        "no_read_rate": no_reads / len(readable) if readable else None,
        "false_read_rate": false_reads / len(negatives) if negatives else None,
        "mean_confidence": statistics.mean(confidences) if confidences else None,
        "mean_latency_ms": statistics.mean(latencies) if latencies else None,
        "median_latency_ms": statistics.median(latencies) if latencies else None,
    }


def deduplicate(samples: list[dict]) -> list[dict]:
    seen = {}
    output = []
    for sample in samples:
        digest = sample["content_hash"]
        if digest in seen:
            previous = seen[digest]
            if (
                previous.get("readability") != sample.get("readability")
                or previous.get("gt_text") != sample.get("gt_text")
            ):
                raise ValueError("Identical crops have inconsistent ground truth")
        else:
            seen[digest] = sample
            output.append(sample)
    return output


def metric_rows(engine: str, mode: str, samples: list[dict], results: dict, separators: str) -> list[dict]:
    groups = [("all", "all", samples)]
    for field in ("quality_band", "width_band"):
        values = defaultdict(list)
        for sample in samples:
            values[sample[field]].append(sample)
        groups.extend((field, value, group) for value, group in sorted(values.items()))
    return [
        {"engine": engine, "dataset_mode": mode, "group_type": kind, "group": name,
         **calculate_metrics(group, results, separators)}
        for kind, name, group in groups
    ]


def self_test() -> None:
    assert normalize_text(" ka-01 ab/1234 ", " -/") == "KA01AB1234"
    assert normalize_text("OI-B8", "-") == "OIB8"
    assert edit_distance("ABC", "ABC") == 0
    assert edit_distance("ABC", "ADC") == 1
    synthetic = [
        {"sample_id": "a", "readability": "readable", "gt_text": "AB12"},
        {"sample_id": "b", "readability": "not_a_plate", "gt_text": ""},
    ]
    results = {
        "a": {"normalized_text": "AB-12", "latency_ms": "10", "confidence": "0.9"},
        "b": {"normalized_text": "NOISE", "latency_ms": "20", "confidence": "0.5"},
    }
    metrics = calculate_metrics(synthetic, results, " -")
    assert metrics["exact_accuracy"] == 1.0
    assert metrics["character_accuracy"] == 1.0
    assert metrics["false_read_rate"] == 1.0
    assert metrics["median_latency_ms"] == 15.0
    duplicate_samples = [{"content_hash": "x"}, {"content_hash": "x"}, {"content_hash": "y"}]
    assert len(deduplicate(duplicate_samples)) == 2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--validate-manifest", action="store_true")
    args = parser.parse_args()
    with args.config.resolve().open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    manifest = (ROOT / config["manifest"]).resolve()
    crops_dir = (ROOT / config["crops_dir"]).resolve()
    samples = load_manifest(manifest, crops_dir)
    if args.self_test:
        self_test()
    if args.self_test or args.validate_manifest:
        print(f"Validation passed: {len(samples)} samples, {len(deduplicate(samples))} unique hashes")
        return

    output_dir = (ROOT / config["output_dir"]).resolve()
    separators = config["normalization"]["separators"]
    all_rows = []
    engines = (
        ("easyocr", "easyocr"),
        ("paddleocr", "paddleocr"),
        ("awiros", "awiros"),
    )
    for engine, section in engines:
        results = load_results(output_dir / config[section]["output"], {row["sample_id"] for row in samples})
        all_rows.extend(metric_rows(engine, "all_40", samples, results, separators))
        all_rows.extend(metric_rows(engine, "deduplicated", deduplicate(samples), results, separators))
    comparison = output_dir / config["evaluation"]["comparison"]
    comparison.parent.mkdir(parents=True, exist_ok=True)
    with comparison.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    report = output_dir / config["evaluation"]["report"]
    with report.open("w", encoding="utf-8") as stream:
        stream.write("# OCR benchmark comparison\n\n")
        stream.write(f"All samples: {len(samples)}; unique content hashes: {len(deduplicate(samples))}.\n\n")
        stream.write("| Engine | Dataset | Exact | Character accuracy | Norm. edit distance | No-read | False-read | Mean latency ms | Median latency ms |\n")
        stream.write("|---|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in all_rows:
            if row["group_type"] != "all":
                continue
            values = [row[key] for key in ("exact_accuracy", "character_accuracy", "normalized_edit_distance", "no_read_rate", "false_read_rate", "mean_latency_ms", "median_latency_ms")]
            formatted = ["n/a" if value is None else f"{value:.4f}" for value in values]
            stream.write(f"| {row['engine']} | {row['dataset_mode']} | {' | '.join(formatted)} |\n")
        stream.write("\nDetailed quality-band and width-band metrics are in `comparison.csv`.\n")
    print(f"Wrote comparison to {comparison} and {report}")


if __name__ == "__main__":
    main()
