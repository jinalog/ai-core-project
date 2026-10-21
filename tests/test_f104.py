import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import yaml
from PIL import Image

from experiments.f104_dataset_validate import CLASS_NAMES, stage_dataset, validate_dataset
from experiments.f104_finetune import evaluate, iou, match_counts, require_best, write_csv


def demo(root, counts=None):
    counts = counts or {"train": 2, "val": 1}
    for split, count in counts.items():
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        for index in range(count):
            Image.new("RGB", (16, 16), (index * 3, 100, 150)).save(root / "images" / split / f"{index}.jpg")
            (root / "labels" / split / f"{index}.txt").write_text("0 .3 .3 .2 .2\n1 .7 .7 .2 .2\n", encoding="utf-8")
    data = root / "data.yaml"
    data.write_text(yaml.safe_dump({"path": str(root), "train": "images/train", "val": "images/val",
                                    "names": CLASS_NAMES}), encoding="utf-8")
    return data


class F104DatasetTests(unittest.TestCase):
    def test_valid_pairs_distribution_background_and_read_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = demo(root)
            (root / "labels/train/0.txt").write_text("", encoding="utf-8")
            before = {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}
            result = validate_dataset(data, expected_counts={"train": 2, "val": 1})
            self.assertTrue(result["valid"])
            self.assertEqual(result["splits"]["train"]["background_images"], 1)
            self.assertEqual(result["splits"]["val"]["class_distribution"], {"obstacle": 1, "marker": 1})
            self.assertEqual(before, {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()})

    def test_labels_invalid_ids_coordinates_and_dimensions(self):
        invalid = ("2 .5 .5 .2 .2", "0.0 .5 .5 .2 .2", "0 nan .5 .2 .2", "1 inf .5 .2 .2",
                   "0 -0.1 .5 .2 .2", "0 .5 .5 1.1 .2", "0 .5 .5 0 .2", "0 .05 .5 .2 .2",
                   "0 .5 .5 .2", "abc .5 .5 .2 .2")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = demo(root)
            for line in invalid:
                with self.subTest(line=line):
                    (root / "labels/train/0.txt").write_text(line, encoding="utf-8")
                    self.assertFalse(validate_dataset(data, expected_counts={"train": 2, "val": 1})["valid"])

    def test_missing_label_orphan_corrupt_image_and_wrong_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = demo(root)
            (root / "labels/train/0.txt").unlink()
            (root / "labels/val/orphan.txt").write_text("", encoding="utf-8")
            (root / "images/val/0.jpg").write_bytes(b"not an image")
            result = validate_dataset(data)
            self.assertFalse(result["valid"])
            for text in ("Missing label", "Orphan label", "Unreadable image", "expected 40"):
                self.assertTrue(any(text in error for error in result["errors"]))

    def test_wrong_class_definition_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = demo(root)
            content = yaml.safe_load(data.read_text(encoding="utf-8"))
            content["names"] = {0: "person", 1: "bicycle"}
            data.write_text(yaml.safe_dump(content), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "obstacle"):
                validate_dataset(data)

    def test_private_copy_hashes_and_source_independence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = demo(root / "source")
            report = validate_dataset(data, expected_counts={"train": 2, "val": 1})
            staged = stage_dataset(report, root / "working")
            working = validate_dataset(staged, expected_counts={"train": 2, "val": 1})
            self.assertEqual(report["dataset_sha256"], working["dataset_sha256"])
            self.assertEqual(report["val_sha256"], working["val_sha256"])
            (root / "working/labels/val/0.txt").write_text("", encoding="utf-8")
            self.assertEqual(report["val_sha256"], validate_dataset(data, expected_counts={"train": 2, "val": 1})["val_sha256"])
            self.assertNotEqual(report["val_sha256"], validate_dataset(staged, expected_counts={"train": 2, "val": 1})["val_sha256"])

    def test_stage_rejects_changed_source_and_invalid_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = demo(root / "source")
            report = validate_dataset(data, expected_counts={"train": 2, "val": 1})
            (root / "source/labels/train/0.txt").write_text("", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "changed"):
                stage_dataset(report, root / "working")
            with self.assertRaisesRegex(ValueError, "invalid"):
                stage_dataset({"valid": False}, root / "invalid")


class F104EvaluationTests(unittest.TestCase):
    def test_fixed_threshold_matching_is_class_aware_and_one_to_one(self):
        target = {"class_id": 0, "bbox": [0, 0, 10, 10]}
        prediction = {**target, "confidence": .8}
        wrong_class = {**prediction, "class_id": 1, "confidence": .7}
        low = {**prediction, "confidence": .1}
        self.assertEqual(iou(target["bbox"], target["bbox"]), 1)
        self.assertEqual(iou([0, 0, 0, 0], [0, 0, 0, 0]), 0)
        self.assertEqual(match_counts([prediction, prediction, wrong_class, low], [target], threshold=.25),
                         {"tp": 1, "fp": 2, "fn": 0})
        self.assertEqual(match_counts([low], [target], threshold=.25), {"tp": 0, "fp": 0, "fn": 1})

    def test_best_weight_must_exist_and_be_nonempty(self):
        with tempfile.TemporaryDirectory() as directory:
            best = Path(directory) / "best.pt"
            with self.assertRaises(FileNotFoundError):
                require_best(best)
            best.touch()
            with self.assertRaises(FileNotFoundError):
                require_best(best)
            best.write_bytes(b"unit-test-only-not-real-weights")
            self.assertEqual(len(require_best(best)), 64)

    def test_before_after_use_identical_val_and_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = demo(root / "dataset", {"train": 40, "val": 10})
            report = validate_dataset(data)
            output = root / "eval"
            output.mkdir()
            model = Mock()
            model.names = CLASS_NAMES
            model.predict.return_value = [SimpleNamespace(boxes=None)] * 10
            model.val.return_value = SimpleNamespace(box=SimpleNamespace(map50=.2, map=.1, mp=.4, mr=.5))
            factory = Mock(return_value=model)
            rows = []
            for stage in ("before", "after"):
                weights = root / f"{stage}.pt"
                weights.write_bytes(b"unit-test-only")
                rows.extend(evaluate(weights, data, output, expected_val=report["val_sha256"],
                                     stage=stage, imgsz=320, device="cpu", factory=factory))
            self.assertEqual(len(rows), 4)
            self.assertEqual([r["conf"] for r in rows], [.001, .25, .001, .25])
            self.assertEqual(len({r["val_sha256"] for r in rows}), 1)
            self.assertTrue(all(r["precision"] == r["recall"] == r["max_conf"] == 0 for r in rows))
            self.assertTrue(all(r["ultralytics_precision_best_f1"] == .4 for r in rows))
            self.assertTrue(all(call.kwargs["data"] == str(data) and call.kwargs["imgsz"] == 320
                                for call in model.val.call_args_list))
            write_csv(root / "compare.csv", rows)
            with (root / "compare.csv").open(encoding="utf-8", newline="") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 4)
            (root / "dataset/labels/val/0.txt").write_text("", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "changed"):
                evaluate(weights, data, output, expected_val=report["val_sha256"], stage="after",
                         imgsz=320, device="cpu", factory=factory)


if __name__ == "__main__":
    unittest.main()
