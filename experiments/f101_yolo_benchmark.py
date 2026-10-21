"""F101 YOLO11n confidence sweep and repeated inference benchmark."""

import argparse
import csv
import hashlib
import json
import math
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from app.config import Settings, project_path

THRESHOLDS = (0.10, 0.25, 0.60)


def summarize(durations: list[float]) -> dict:
    if not durations or any(not math.isfinite(x) or x <= 0 for x in durations):
        raise ValueError("Durations must be finite positive seconds")
    mean = statistics.mean(durations)
    return {"mean_ms": mean * 1000, "min_ms": min(durations) * 1000,
            "median_ms": statistics.median(durations) * 1000,
            "p95_ms": sorted(durations)[math.ceil(len(durations) * .95) - 1] * 1000,
            "max_ms": max(durations) * 1000, "fps": 1 / mean}


def detections_from(results) -> list[dict]:
    detections = []
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            class_id = int(box.cls.item())
            detections.append({"bbox": [float(x) for x in box.xyxy[0].tolist()],
                               "class_id": class_id, "class": result.names[class_id],
                               "confidence": float(box.conf.item())})
    return detections


def benchmark(model, image, *, rss, repeats=10, warmup=1, imgsz=640, device="cpu",
              synchronize=lambda: None, clock=time.perf_counter):
    """First/warmup calls are excluded; RSS samples do not measure peaks or VRAM."""
    if min(repeats, warmup, imgsz) < 1:
        raise ValueError("repeats, warmup and imgsz must be positive")
    summaries, iterations, sweep = [], [], []
    for conf in THRESHOLDS:
        before = rss()

        def infer():
            synchronize()
            start = clock()
            result = model.predict(source=image, conf=conf, imgsz=imgsz, device=device,
                                   verbose=False, save=False)
            synchronize()
            return result, clock() - start

        result, first = infer()
        for _ in range(warmup - 1):
            result, _ = infer()
        after_warmup = rss()
        durations = []
        for iteration in range(1, repeats + 1):
            result, elapsed = infer()
            durations.append(elapsed)
            after = rss()
            iterations.append({"conf": conf, "iteration": iteration,
                               "elapsed_ms": elapsed * 1000, "rss_bytes": after,
                               "rss_delta_bytes": after - before})
        detections = detections_from(result)
        sweep.append({"conf": conf, "detection_count": len(detections),
                      "detections_json": json.dumps(detections, ensure_ascii=False)})
        summaries.append({"conf": conf, "repeats": repeats, "warmup_calls": warmup,
                          "first_call_ms": first * 1000, "cold_call": conf == THRESHOLDS[0],
                          **summarize(durations), "rss_before_bytes": before,
                          "rss_after_warmup_bytes": after_warmup, "rss_after_bytes": after,
                          "rss_delta_bytes": after - before, "detection_count": len(detections)})
    return summaries, iterations, sweep


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("Empty data is not a measurement")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_inputs(weights: Path, image: Path) -> None:
    for path in (weights, image):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing or empty local file: {path}; see README.md")
    if weights.suffix.lower() != ".pt":
        raise ValueError("Expected local YOLO11n .pt weights")
    from PIL import Image
    with Image.open(image) as opened:
        if opened.format != "JPEG":
            raise ValueError("Expected a real JPEG sample")
        opened.verify()


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> int:
    config = Settings.from_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=project_path, default=config.weights)
    parser.add_argument("--image", type=project_path, default=config.image)
    parser.add_argument("--device", default=config.device)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--output-dir", type=project_path, default=project_path("artifacts"))
    args = parser.parse_args()
    try:
        if min(args.repeats, args.warmup, args.imgsz) < 1:
            raise ValueError("repeats, warmup and imgsz must be positive")
        validate_inputs(args.weights, args.image)  # Check before YOLO can auto-download.
        import psutil
        import torch
        import ultralytics
        from PIL import Image
        from ultralytics import YOLO

        if args.device != "cpu" and not (args.device.isdigit() and torch.cuda.is_available()
                                        and int(args.device) < torch.cuda.device_count()):
            raise ValueError("device must be cpu or an available CUDA index, e.g. 0")
        sync = (lambda: torch.cuda.synchronize(int(args.device))) if args.device != "cpu" else (lambda: None)
        process = psutil.Process()
        rss_before_load = process.memory_info().rss
        start = time.perf_counter()
        model = YOLO(str(args.weights), task="detect")
        load_ms = (time.perf_counter() - start) * 1000
        rss_after_load = process.memory_info().rss
        with Image.open(args.image) as opened:
            width, height = opened.size
        metadata = {"run_id": datetime.now(timezone.utc).isoformat(),
                    "python": platform.python_version(), "os": platform.platform(),
                    "device": args.device, "cuda_available": torch.cuda.is_available(),
                    "device_name": torch.cuda.get_device_name(int(args.device)) if args.device != "cpu" else platform.processor(),
                    "torch_version": torch.__version__, "ultralytics_version": ultralytics.__version__,
                    "weights": str(args.weights), "weights_sha256": sha256(args.weights),
                    "image": str(args.image), "image_sha256": sha256(args.image),
                    "image_width": width, "image_height": height, "imgsz": args.imgsz,
                    "model_load_ms": load_ms, "rss_before_load_bytes": rss_before_load,
                    "rss_after_load_bytes": rss_after_load,
                    "model_load_rss_delta_bytes": rss_after_load - rss_before_load}
        summary, iterations, sweep = benchmark(model, str(args.image), rss=lambda: process.memory_info().rss,
                                               repeats=args.repeats, warmup=args.warmup, imgsz=args.imgsz,
                                               device=args.device, synchronize=sync)
        for filename, rows in (("measurement.csv", summary), ("f101_iterations.csv", iterations),
                               ("f101_conf_sweep.csv", sweep)):
            write_csv(args.output_dir / filename, [{**metadata, **row} for row in rows])
        for row in sweep:
            print(json.dumps(row, ensure_ascii=False))
        print(f"F101 completed: {args.output_dir / 'measurement.csv'}")
        return 0
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        print(f"F101 failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
