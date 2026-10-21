"""Read-only F104 validation and private working-copy preparation."""

import argparse
import hashlib
import json
import math
import shutil
from collections import Counter
from pathlib import Path

import yaml
from PIL import Image

from app.config import project_path

CLASS_NAMES = {0: "obstacle", 1: "marker"}
EXPECTED_COUNTS = {"train": 40, "val": 10}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
DEFAULT_DATA = project_path("source/2회차_day1/dataset/data.yaml")


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def content_digest(files: dict) -> str:
    return hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_definition(data_path: Path):
    definition = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    if not isinstance(definition, dict):
        raise ValueError("data.yaml must contain a mapping")
    names = definition.get("names")
    if isinstance(names, list):
        names = dict(enumerate(names))
    if names != CLASS_NAMES:
        raise ValueError("F104 requires exactly {0: obstacle, 1: marker}")
    root = Path(definition.get("path", "."))
    if not root.is_absolute():
        root = data_path.parent / root
    root = root.resolve()
    # Restrict this experiment to the defined, local YOLO directory layout.
    for split in EXPECTED_COUNTS:
        if definition.get(split) != f"images/{split}":
            raise ValueError(f"F104 expects {split}: images/{split}")
    return root


def validate_dataset(data_path: Path, *, expected_counts=None):
    expected_counts = EXPECTED_COUNTS if expected_counts is None else expected_counts
    root = read_definition(data_path)
    errors, warnings, splits, hashes = [], [], {}, {}
    image_hashes = {}
    for split in ("train", "val"):
        image_dir, label_dir = root / "images" / split, root / "labels" / split
        if not image_dir.is_dir() or not label_dir.is_dir():
            errors.append(f"Missing images or labels directory: {split}")
            continue
        images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file())
        labels = sorted(p for p in label_dir.glob("*.txt") if p.is_file())
        if len(images) != expected_counts[split]:
            errors.append(f"{split}: expected {expected_counts[split]} images, found {len(images)}")
        stems = [p.stem.casefold() for p in images]
        if len(stems) != len(set(stems)):
            errors.append(f"{split}: duplicate image stems")
        counts = Counter({name: 0 for name in CLASS_NAMES.values()})
        backgrounds, instances, sizes = 0, 0, set()
        image_hashes[split] = set()
        for image in images:
            image_key = image.relative_to(root).as_posix()
            hashes[image_key] = file_hash(image)
            image_hashes[split].add(hashes[image_key])
            try:
                with Image.open(image) as opened:
                    sizes.add(opened.size)
                    opened.verify()
                with Image.open(image) as opened:
                    opened.load()
            except Exception as exc:
                errors.append(f"Unreadable image {image_key}: {type(exc).__name__}")
            label = label_dir / f"{image.stem}.txt"
            if not label.is_file():
                errors.append(f"Missing label: {image_key}")
                continue
            label_key = label.relative_to(root).as_posix()
            hashes[label_key] = file_hash(label)
            try:
                lines = [line.strip() for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]
            except UnicodeError:
                errors.append(f"Invalid UTF-8 label: {label_key}")
                continue
            if not lines:
                backgrounds += 1
            for number, line in enumerate(lines, 1):
                prefix = f"{label_key}:{number}"
                fields = line.split()
                if len(fields) != 5:
                    errors.append(f"{prefix}: expected class_id x_center y_center width height")
                    continue
                try:
                    class_id = int(fields[0])
                    x, y, width, height = map(float, fields[1:])
                except ValueError:
                    errors.append(f"{prefix}: invalid numeric fields")
                    continue
                if class_id not in CLASS_NAMES:
                    errors.append(f"{prefix}: invalid class ID {class_id}")
                    continue
                if any(not math.isfinite(value) or not 0 <= value <= 1 for value in (x, y, width, height)):
                    errors.append(f"{prefix}: coordinates must be finite and normalized to 0..1")
                    continue
                if width <= 0 or height <= 0:
                    errors.append(f"{prefix}: width and height must be positive")
                    continue
                if min(x - width / 2, y - height / 2) < -1e-6 or max(x + width / 2, y + height / 2) > 1 + 1e-6:
                    errors.append(f"{prefix}: bounding box extends outside image")
                    continue
                counts[CLASS_NAMES[class_id]] += 1
                instances += 1
        for label in labels:
            if label.stem.casefold() not in stems:
                errors.append(f"Orphan label: {label.relative_to(root).as_posix()}")
        splits[split] = {"image_count": len(images), "label_count": len(labels),
                         "background_images": backgrounds, "instances": instances,
                         "class_distribution": dict(counts), "image_sizes": sorted(sizes)}
        if any(count == 0 for count in counts.values()):
            warnings.append(f"{split}: one or more classes have no instances")
    overlap = image_hashes.get("train", set()) & image_hashes.get("val", set())
    if overlap:
        warnings.append(f"Train/val have {len(overlap)} identical image hash(es); inspect possible leakage")
    val_hashes = {key: value for key, value in hashes.items() if key.startswith(("images/val/", "labels/val/"))}
    return {"valid": not errors, "data_yaml": str(data_path.resolve()), "root": str(root),
            "names": CLASS_NAMES, "splits": splits, "errors": errors, "warnings": warnings,
            "files": hashes, "dataset_sha256": content_digest(hashes),
            "val_sha256": content_digest(val_hashes)}


def stage_dataset(report: dict, destination: Path) -> Path:
    if not report["valid"]:
        raise ValueError("Cannot stage an invalid dataset")
    destination.mkdir(parents=True, exist_ok=False)
    source = Path(report["root"])
    for relative, digest in report["files"].items():
        original = source / relative
        if file_hash(original) != digest:
            raise ValueError(f"Dataset changed since validation: {relative}")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, target)  # Independent private copy; no shared cache/hard links.
        if file_hash(target) != digest:
            raise ValueError(f"Working copy hash mismatch: {relative}")
    definition = {"path": destination.resolve().as_posix(), "train": "images/train",
                  "val": "images/val", "names": CLASS_NAMES}
    data = destination / "data.yaml"
    data.write_text(yaml.safe_dump(definition, sort_keys=False), encoding="utf-8")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=project_path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=project_path, default=project_path("artifacts/f104/dataset_validation.json"))
    args = parser.parse_args()
    try:
        report = validate_dataset(args.data)
    except Exception as exc:
        report = {"valid": False, "errors": [f"{type(exc).__name__}: {exc}"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "files"}, ensure_ascii=False, indent=2))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
