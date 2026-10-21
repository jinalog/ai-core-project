import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from experiments.f101_yolo_benchmark import (
    THRESHOLDS, benchmark, detections_from, main, summarize, validate_inputs, write_csv,
)


def fake_result():
    box = SimpleNamespace(cls=SimpleNamespace(item=lambda: 2),
                          conf=SimpleNamespace(item=lambda: 0.7),
                          xyxy=[SimpleNamespace(tolist=lambda: [1, 2, 30, 40])])
    return SimpleNamespace(names={2: "car"}, boxes=[box])


class F101Tests(unittest.TestCase):
    def test_statistics(self):
        result = summarize([0.1, 0.2, 0.3, 0.4])
        self.assertAlmostEqual(result["mean_ms"], 250)
        self.assertAlmostEqual(result["median_ms"], 250)
        self.assertAlmostEqual(result["p95_ms"], 400)
        self.assertAlmostEqual(result["fps"], 4)

    def test_invalid_statistics(self):
        for values in ([], [0], [-1], [float("nan")], [float("inf")]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                summarize(values)

    def test_bbox_class_confidence(self):
        self.assertEqual(detections_from([fake_result()]), [
            {"bbox": [1., 2., 30., 40.], "class_id": 2, "class": "car", "confidence": .7}])
        self.assertEqual(detections_from([SimpleNamespace(boxes=None)]), [])
        self.assertEqual(detections_from([SimpleNamespace(boxes=[])]), [])

    def test_sweep_warmup_timing_memory_and_csv(self):
        # Per conf: first 9s, second warmup 7s, measured calls 1s and 3s.
        ticks = iter([0, 9, 10, 17, 20, 21, 30, 33] * 3)
        memory = iter([100, 200, 250, 280] * 3)
        model = Mock()
        model.predict.side_effect = lambda **kw: [fake_result()] if kw["conf"] < .6 else []
        sync = Mock()
        summary, iterations, sweep = benchmark(model, "test-only.jpg", repeats=2, warmup=2,
                                               rss=lambda: next(memory), clock=lambda: next(ticks),
                                               synchronize=sync)
        self.assertEqual([r["conf"] for r in summary], list(THRESHOLDS))
        self.assertEqual(model.predict.call_count, 12)
        self.assertEqual(sync.call_count, 24)
        self.assertEqual(len(iterations), 6)
        self.assertEqual([r["cold_call"] for r in summary], [True, False, False])
        self.assertEqual([r["detection_count"] for r in sweep], [1, 1, 0])
        for row in summary:
            self.assertEqual(row["first_call_ms"], 9000)
            self.assertEqual(row["mean_ms"], 2000)
            self.assertEqual(row["fps"], .5)
            self.assertEqual(row["rss_delta_bytes"], 180)
        for call in model.predict.call_args_list:
            self.assertFalse(call.kwargs["save"])
            self.assertEqual(call.kwargs["imgsz"], 640)
            self.assertEqual(call.kwargs["device"], "cpu")
        with tempfile.TemporaryDirectory() as directory:
            for name, rows in (("measurement.csv", summary), ("iterations.csv", iterations), ("sweep.csv", sweep)):
                path = Path(directory) / name
                write_csv(path, rows)
                with path.open(encoding="utf-8", newline="") as stream:
                    read = list(csv.DictReader(stream))
                self.assertEqual(len(read), len(rows))
                self.assertEqual(set(read[0]), set(rows[0]))
                if name == "sweep.csv":
                    self.assertEqual(json.loads(read[0]["detections_json"])[0]["class"], "car")

    def test_invalid_parameters(self):
        for options in ({"repeats": 0}, {"warmup": 0}, {"imgsz": -1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                benchmark(Mock(), "unused", rss=lambda: 0, **options)

    def test_missing_inputs_do_not_write_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("sys.argv", ["f101", "--weights", str(root / "missing.pt"),
                                    "--image", str(root / "missing.jpg"),
                                    "--output-dir", str(root / "artifacts")]):
                self.assertEqual(main(), 1)
            self.assertFalse((root / "artifacts").exists())

    def test_corrupted_jpeg(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "test.pt").write_bytes(b"unit-test-only")
            (root / "bad.jpg").write_bytes(b"not an image")
            with self.assertRaises(OSError):
                validate_inputs(root / "test.pt", root / "bad.jpg")

    def test_empty_csv_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty.csv"
            with self.assertRaises(ValueError):
                write_csv(path, [])
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
