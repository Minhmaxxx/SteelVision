"""PASCAL VOC XML -> YOLO label conversion.

NEU-DET and GC10-DET ship bounding-box annotations as PASCAL VOC XML files.
This module discovers (image, xml) pairs under a raw dataset directory and
parses them into normalized YOLO boxes:

    class_id x_center y_center width height    (all values in [0, 1])
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class YoloBox:
    class_name: str  # normalized: lowercase, underscores
    x_center: float
    y_center: float
    width: float
    height: float

    def to_line(self, class_id: int) -> str:
        return (
            f"{class_id} {self.x_center:.6f} {self.y_center:.6f} {self.width:.6f} {self.height:.6f}"
        )


@dataclass
class Sample:
    image: Path
    xml: Path
    boxes: list[YoloBox] = field(default_factory=list)

    @property
    def class_names(self) -> set[str]:
        return {box.class_name for box in self.boxes}


@dataclass
class ParseReport:
    n_images: int = 0
    n_boxes: int = 0
    n_degenerate: int = 0
    n_missing_image: int = 0
    n_unannotated_images: int = 0
    class_counts: dict[str, int] = field(default_factory=dict)

    def merge(self, other: ParseReport) -> None:
        self.n_images += other.n_images
        self.n_boxes += other.n_boxes
        self.n_degenerate += other.n_degenerate
        self.n_missing_image += other.n_missing_image
        self.n_unannotated_images += other.n_unannotated_images
        for name, count in other.class_counts.items():
            self.class_counts[name] = self.class_counts.get(name, 0) + count


def normalize_class(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def parse_voc_xml(xml_path: Path, report: ParseReport | None = None) -> list[YoloBox]:
    """Parse one VOC XML file into normalized YOLO boxes.

    Boxes are clipped to the image bounds; degenerate boxes (negative or
    sub-pixel after clipping) are counted in ``report.n_degenerate`` and
    skipped.
    """
    report = report if report is not None else ParseReport()
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    width = float(size.findtext("width", "0")) if size is not None else 0.0
    height = float(size.findtext("height", "0")) if size is not None else 0.0

    boxes: list[YoloBox] = []
    for obj in root.iterfind("object"):
        name = (obj.findtext("name") or "").strip()
        bnd = obj.find("bndbox")
        if not name or bnd is None:
            report.n_degenerate += 1
            continue
        xmin = float(bnd.findtext("xmin", "0"))
        ymin = float(bnd.findtext("ymin", "0"))
        xmax = float(bnd.findtext("xmax", "0"))
        ymax = float(bnd.findtext("ymax", "0"))
        # Canvas fallback when <size> is missing: cover the largest coordinate seen.
        if width <= 0 or height <= 0:
            width, height = max(width, xmax), max(height, ymax)
        xmin = min(max(xmin, 0.0), width)
        xmax = min(max(xmax, 0.0), width)
        ymin = min(max(ymin, 0.0), height)
        ymax = min(max(ymax, 0.0), height)
        box_w, box_h = xmax - xmin, ymax - ymin
        if box_w < 1.0 or box_h < 1.0:
            report.n_degenerate += 1
            continue
        boxes.append(
            YoloBox(
                class_name=normalize_class(name),
                x_center=(xmin + xmax) / 2 / width,
                y_center=(ymin + ymax) / 2 / height,
                width=box_w / width,
                height=box_h / height,
            )
        )
        report.class_counts[boxes[-1].class_name] = (
            report.class_counts.get(boxes[-1].class_name, 0) + 1
        )
    report.n_boxes += len(boxes)
    return boxes


def index_images(root: Path) -> dict[str, Path]:
    """Map lowercase image stem -> first matching path under ``root``."""
    index: dict[str, Path] = {}
    for path in root.rglob("*"):
        if path.suffix.lower() in IMAGE_EXTS:
            index.setdefault(path.stem.lower(), path)
    return index


def discover_samples(raw_root: Path, report: ParseReport | None = None) -> list[Sample]:
    """Find every (image, xml) pair under ``raw_root``.

    Layout-agnostic: XML files are located anywhere under the root and matched
    to images by filename stem (case-insensitive). Samples with no image file
    are counted in ``report.n_missing_image`` and skipped.
    """
    report = report if report is not None else ParseReport()
    images = index_images(raw_root)
    samples: list[Sample] = []
    for xml_path in sorted(raw_root.rglob("*.xml")):
        filename = ET.parse(xml_path).getroot().findtext("filename") or ""
        stem = Path(filename).stem.lower() or xml_path.stem.lower()
        image = images.get(stem)
        if image is None:
            report.n_missing_image += 1
            continue
        samples.append(Sample(image=image, xml=xml_path))
    annotated_images = {s.image for s in samples}
    report.n_unannotated_images = len(images) - len(annotated_images)
    for sample in samples:
        sample.boxes = parse_voc_xml(sample.xml, report)
    report.n_images = len(samples)
    return samples
