from __future__ import annotations

import random
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCE_DIR = PROJECT_ROOT / "data" / "plate_source"
OUTPUT_DIR = PROJECT_ROOT / "data" / "plate_detection"

IMAGE_DIR = SOURCE_DIR / "Images"
LABEL_DIR = SOURCE_DIR / "Labels"

SEED = 42

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

CLASS_ID = 0
CLASS_NAME = "license_plate"

SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


@dataclass(frozen=True, slots=True)
class Sample:
    """One valid image/plate annotation pair."""

    stem: str
    image_path: Path
    xml_path: Path
    plate_number: str
    bbox: tuple[int, int, int, int]


@dataclass(slots=True)
class PreparationReport:
    images_found: int = 0
    xml_found: int = 0
    valid_samples: int = 0
    missing_images: int = 0
    malformed_xml: int = 0
    missing_objects: int = 0
    invalid_boxes: int = 0
    output_samples: int = 0
    train_samples: int = 0
    val_samples: int = 0
    test_samples: int = 0
    dimension_mismatches: int = 0


def find_images() -> dict[str, Path]:
    """Return source images indexed by filename stem."""

    if not IMAGE_DIR.is_dir():
        raise FileNotFoundError(f"Image directory not found: {IMAGE_DIR}")

    images: dict[str, Path] = {}

    for path in IMAGE_DIR.iterdir():
        if not path.is_file():
            continue

        if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            continue

        if path.stem in images:
            raise RuntimeError(
                f"Duplicate image stem detected: {path.stem}"
            )

        images[path.stem] = path

    return images


def parse_annotation(
    xml_path: Path,
    image_path: Path,
    report: PreparationReport,
) -> Sample | None:
    """Parse one Pascal VOC annotation and validate its plate box."""

    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        report.malformed_xml += 1
        return None

    objects = root.findall("object")

    if not objects:
        report.missing_objects += 1
        return None

    # Our current verified dataset has one plate object per image.
    if len(objects) != 1:
        report.invalid_boxes += 1
        print(
            f"[SKIP] {xml_path.name}: expected 1 object, "
            f"found {len(objects)}"
        )
        return None

    obj = objects[0]

    plate_number = (obj.findtext("name") or "").strip()

    if not plate_number:
        report.invalid_boxes += 1
        print(f"[SKIP] {xml_path.name}: missing plate identifier")
        return None

    bbox = obj.find("bndbox")

    if bbox is None:
        report.invalid_boxes += 1
        print(f"[SKIP] {xml_path.name}: missing bounding box")
        return None

    try:
        xmin = int(bbox.findtext("xmin", ""))
        ymin = int(bbox.findtext("ymin", ""))
        xmax = int(bbox.findtext("xmax", ""))
        ymax = int(bbox.findtext("ymax", ""))
    except ValueError:
        report.invalid_boxes += 1
        print(f"[SKIP] {xml_path.name}: non-integer bounding box")
        return None

    try:
        with Image.open(image_path) as image:
            image_width, image_height = image.size
    except Exception as exc:
        report.invalid_boxes += 1
        print(f"[SKIP] {image_path.name}: cannot read image: {exc}")
        return None

    xml_width_text = root.findtext("size/width")
    xml_height_text = root.findtext("size/height")

    try:
        xml_width = int(xml_width_text) if xml_width_text else None
        xml_height = int(xml_height_text) if xml_height_text else None
    except ValueError:
        xml_width = None
        xml_height = None

    if (
        xml_width is not None
        and xml_height is not None
        and (xml_width != image_width or xml_height != image_height)
    ):
        report.dimension_mismatches += 1

    # Never silently clamp invalid annotations.
    if xmin < 0 or ymin < 0:
        report.invalid_boxes += 1
        print(f"[SKIP] {xml_path.name}: negative bounding-box coordinate")
        return None

    if xmax > image_width or ymax > image_height:
        report.invalid_boxes += 1
        print(
            f"[SKIP] {xml_path.name}: bbox outside actual image "
            f"({image_width}x{image_height})"
        )
        return None

    if xmin >= xmax or ymin >= ymax:
        report.invalid_boxes += 1
        print(f"[SKIP] {xml_path.name}: zero/negative bbox")
        return None

    return Sample(
        stem=xml_path.stem,
        image_path=image_path,
        xml_path=xml_path,
        plate_number=plate_number,
        bbox=(xmin, ymin, xmax, ymax),
    )


def collect_samples(
    report: PreparationReport,
) -> list[Sample]:
    """Collect all valid image/annotation pairs."""

    images = find_images()
    xml_files = sorted(LABEL_DIR.glob("*.xml"))

    report.images_found = len(images)
    report.xml_found = len(xml_files)

    samples: list[Sample] = []

    for xml_path in xml_files:
        image_path = images.get(xml_path.stem)

        if image_path is None:
            report.missing_images += 1
            print(
                f"[SKIP] {xml_path.name}: corresponding image not found"
            )
            continue

        sample = parse_annotation(
            xml_path,
            image_path,
            report,
        )

        if sample is not None:
            samples.append(sample)

    report.valid_samples = len(samples)

    return samples


def split_samples(
    samples: list[Sample],
) -> dict[str, list[Sample]]:
    """
    Split by plate identity.

    This prevents multiple images of the same plate from being distributed
    across train/validation/test and creating optimistic evaluation.
    """

    groups: dict[str, list[Sample]] = defaultdict(list)

    for sample in samples:
        groups[sample.plate_number].append(sample)

    group_items = list(groups.values())

    random.Random(SEED).shuffle(group_items)

    total = len(samples)

    target_train = round(total * TRAIN_RATIO)
    target_val = round(total * VAL_RATIO)

    splits: dict[str, list[Sample]] = {
        "train": [],
        "val": [],
        "test": [],
    }

    for group in group_items:
        current_train = len(splits["train"])
        current_val = len(splits["val"])

        if current_train < target_train:
            split_name = "train"
        elif current_val < target_val:
            split_name = "val"
        else:
            split_name = "test"

        splits[split_name].extend(group)

    return splits


def convert_bbox_to_yolo(
    bbox: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
) -> str:
    """Convert absolute pixel coordinates to normalized YOLO format."""

    xmin, ymin, xmax, ymax = bbox

    box_width = xmax - xmin
    box_height = ymax - ymin

    center_x = xmin + (box_width / 2.0)
    center_y = ymin + (box_height / 2.0)

    x_center_norm = center_x / image_width
    y_center_norm = center_y / image_height
    width_norm = box_width / image_width
    height_norm = box_height / image_height

    values = (
        x_center_norm,
        y_center_norm,
        width_norm,
        height_norm,
    )

    if not all(0.0 < value <= 1.0 for value in values[2:]):
        raise ValueError(f"Invalid normalized dimensions: {values}")

    if not (
        0.0 <= x_center_norm <= 1.0
        and 0.0 <= y_center_norm <= 1.0
    ):
        raise ValueError(f"Invalid normalized center: {values}")

    return (
        f"{CLASS_ID} "
        f"{x_center_norm:.8f} "
        f"{y_center_norm:.8f} "
        f"{width_norm:.8f} "
        f"{height_norm:.8f}"
    )


def prepare_output_directory() -> None:
    """Create a clean generated dataset directory."""

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    for split in ("train", "val", "test"):
        (OUTPUT_DIR / split / "images").mkdir(
            parents=True,
            exist_ok=True,
        )
        (OUTPUT_DIR / split / "labels").mkdir(
            parents=True,
            exist_ok=True,
        )


def write_sample(
    sample: Sample,
    split: str,
) -> None:
    """Copy image and write its YOLO annotation."""

    destination_image = (
        OUTPUT_DIR
        / split
        / "images"
        / sample.image_path.name
    )

    destination_label = (
        OUTPUT_DIR
        / split
        / "labels"
        / f"{sample.stem}.txt"
    )

    shutil.copy2(sample.image_path, destination_image)

    with Image.open(sample.image_path) as image:
        image_width, image_height = image.size

    yolo_line = convert_bbox_to_yolo(
        sample.bbox,
        image_width,
        image_height,
    )

    destination_label.write_text(
        yolo_line + "\n",
        encoding="utf-8",
    )


def write_data_yaml() -> None:
    """Write the Ultralytics dataset configuration."""

    dataset_path = OUTPUT_DIR.as_posix()

    yaml_content = f"""path: "{dataset_path}"
train: train/images
val: val/images
test: test/images

nc: 1
names:
  0: {CLASS_NAME}
"""

    (OUTPUT_DIR / "data.yaml").write_text(
        yaml_content,
        encoding="utf-8",
    )


def validate_generated_dataset(
    splits: dict[str, list[Sample]],
) -> None:
    """Verify image/label counts and generated YOLO coordinates."""

    for split, samples in splits.items():
        image_dir = OUTPUT_DIR / split / "images"
        label_dir = OUTPUT_DIR / split / "labels"

        images = {
            path.stem
            for path in image_dir.iterdir()
            if path.is_file()
        }

        labels = {
            path.stem
            for path in label_dir.iterdir()
            if path.is_file()
        }

        if images != labels:
            missing_labels = sorted(images - labels)
            missing_images = sorted(labels - images)

            raise RuntimeError(
                f"{split}: image/label mismatch. "
                f"Missing labels={missing_labels[:10]}, "
                f"missing images={missing_images[:10]}"
            )

        for label_path in label_dir.glob("*.txt"):
            lines = [
                line.strip()
                for line in label_path.read_text(
                    encoding="utf-8"
                ).splitlines()
                if line.strip()
            ]

            if len(lines) != 1:
                raise RuntimeError(
                    f"{label_path}: expected exactly one annotation"
                )

            values = lines[0].split()

            if len(values) != 5:
                raise RuntimeError(
                    f"{label_path}: expected 5 YOLO values"
                )

            if int(values[0]) != CLASS_ID:
                raise RuntimeError(
                    f"{label_path}: unexpected class ID"
                )

            coordinates = [float(value) for value in values[1:]]

            if not all(0.0 <= value <= 1.0 for value in coordinates):
                raise RuntimeError(
                    f"{label_path}: normalized coordinate outside [0,1]"
                )


def print_report(
    report: PreparationReport,
    splits: dict[str, list[Sample]],
) -> None:
    """Print preparation results."""

    print()
    print("=" * 60)
    print("ETRIS PLATE DATASET PREPARATION")
    print("=" * 60)

    print(f"Source images             : {report.images_found}")
    print(f"Source XML annotations    : {report.xml_found}")
    print(f"Valid samples             : {report.valid_samples}")
    print(f"Missing images            : {report.missing_images}")
    print(f"Malformed XML             : {report.malformed_xml}")
    print(f"Missing objects           : {report.missing_objects}")
    print(f"Invalid bounding boxes    : {report.invalid_boxes}")
    print(f"Dimension mismatches      : {report.dimension_mismatches}")

    print()
    print(f"Train samples             : {len(splits['train'])}")
    print(f"Validation samples        : {len(splits['val'])}")
    print(f"Test samples              : {len(splits['test'])}")

    print()
    print(f"Output directory          : {OUTPUT_DIR}")
    print(f"Class                     : {CLASS_ID} = {CLASS_NAME}")

    print("=" * 60)


def main() -> int:
    """Prepare the YOLO11 plate-detection dataset."""

    try:
        if not SOURCE_DIR.is_dir():
            raise FileNotFoundError(
                f"Source dataset not found: {SOURCE_DIR}"
            )

        if not LABEL_DIR.is_dir():
            raise FileNotFoundError(
                f"Label directory not found: {LABEL_DIR}"
            )

        report = PreparationReport()

        samples = collect_samples(report)

        if not samples:
            raise RuntimeError(
                "No valid plate samples were found."
            )

        splits = split_samples(samples)

        prepare_output_directory()

        for split, split_samples_list in splits.items():
            for sample in split_samples_list:
                write_sample(sample, split)

        write_data_yaml()

        validate_generated_dataset(splits)

        report.output_samples = sum(
            len(items) for items in splits.values()
        )
        report.train_samples = len(splits["train"])
        report.val_samples = len(splits["val"])
        report.test_samples = len(splits["test"])

        print_report(report, splits)

        return 0

    except Exception as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())