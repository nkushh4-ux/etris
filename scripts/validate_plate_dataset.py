from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class ValidationReport:
    image_count: int = 0
    xml_count: int = 0
    valid_pairs: int = 0
    missing_images: int = 0
    malformed_xml: int = 0
    missing_objects: int = 0
    multiple_objects: int = 0
    invalid_boxes: int = 0
    out_of_bounds_boxes: int = 0
    dimension_mismatches: int = 0


def parse_int(
    element: ET.Element | None,
    field_name: str,
    xml_path: Path,
) -> int:
    if element is None or element.text is None:
        raise ValueError(f"Missing {field_name} in {xml_path.name}")

    try:
        return int(element.text)
    except ValueError as exc:
        raise ValueError(
            f"Invalid {field_name} in {xml_path.name}: {element.text!r}"
        ) from exc


def validate_dataset(dataset_root: Path) -> ValidationReport:
    images_dir = dataset_root / "Images"
    labels_dir = dataset_root / "Labels"

    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

    report = ValidationReport()

    images = {
        path.stem: path
        for path in images_dir.iterdir()
        if path.is_file()
    }

    xml_files = sorted(labels_dir.glob("*.xml"))

    report.image_count = len(images)
    report.xml_count = len(xml_files)

    for xml_path in xml_files:
        image_path = images.get(xml_path.stem)

        if image_path is None:
            report.missing_images += 1
            continue

        try:
            root = ET.parse(xml_path).getroot()
        except ET.ParseError:
            report.malformed_xml += 1
            continue

        objects = root.findall("object")

        if not objects:
            report.missing_objects += 1
            continue

        if len(objects) > 1:
            report.multiple_objects += 1

        size = root.find("size")

        try:
            xml_width = parse_int(
                size.find("width") if size is not None else None,
                "width",
                xml_path,
            )
            xml_height = parse_int(
                size.find("height") if size is not None else None,
                "height",
                xml_path,
            )
        except ValueError:
            report.dimension_mismatches += 1
            continue

        try:
            from PIL import Image

            with Image.open(image_path) as image:
                actual_width, actual_height = image.size
        except Exception as exc:
            print(
                f"[ERROR] Could not read image {image_path.name}: {exc}",
                file=sys.stderr,
            )
            report.dimension_mismatches += 1
            continue

        if (xml_width, xml_height) != (actual_width, actual_height):
            report.dimension_mismatches += 1

        sample_valid = True

        for obj in objects:
            bbox = obj.find("bndbox")

            if bbox is None:
                report.invalid_boxes += 1
                sample_valid = False
                continue

            try:
                xmin = parse_int(bbox.find("xmin"), "xmin", xml_path)
                ymin = parse_int(bbox.find("ymin"), "ymin", xml_path)
                xmax = parse_int(bbox.find("xmax"), "xmax", xml_path)
                ymax = parse_int(bbox.find("ymax"), "ymax", xml_path)
            except ValueError:
                report.invalid_boxes += 1
                sample_valid = False
                continue

            if xmin >= xmax or ymin >= ymax:
                report.invalid_boxes += 1
                sample_valid = False
                continue

            if (
                xmin < 0
                or ymin < 0
                or xmax > actual_width
                or ymax > actual_height
            ):
                report.out_of_bounds_boxes += 1
                sample_valid = False

        if sample_valid:
            report.valid_pairs += 1

    return report


def print_report(report: ValidationReport) -> None:
    print("\n" + "=" * 55)
    print("ETRIS INDIAN PLATE DATASET VALIDATION")
    print("=" * 55)

    print(f"Images                  : {report.image_count}")
    print(f"XML annotations         : {report.xml_count}")
    print(f"Valid image/XML pairs   : {report.valid_pairs}")
    print(f"Missing images          : {report.missing_images}")
    print(f"Malformed XML           : {report.malformed_xml}")
    print(f"Missing objects         : {report.missing_objects}")
    print(f"Multiple objects        : {report.multiple_objects}")
    print(f"Invalid bounding boxes  : {report.invalid_boxes}")
    print(f"Out-of-bounds boxes     : {report.out_of_bounds_boxes}")
    print(f"Dimension mismatches    : {report.dimension_mismatches}")

    print("=" * 55)


def main() -> int:
    dataset_root = Path(__file__).resolve().parents[1] / "data" / "plate_source"

    try:
        report = validate_dataset(dataset_root)
    except Exception as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 1

    print_report(report)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())