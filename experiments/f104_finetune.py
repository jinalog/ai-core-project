"""F104: fixed demo training and same-val before/after evaluation, never modifying source data."""

import argparse
import csv
import json
import logging
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from app.config import Settings, project_path
from experiments.f104_dataset_validate import DEFAULT_DATA, file_hash, read_definition, stage_dataset, validate_dataset


def save_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        raise ValueError("No real evaluation data to write")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def iou(a, b):
    intersection = max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(0., min(a[3], b[3]) - max(a[1], b[1]))
    area_a = max(0., a[2] - a[0]) * max(0., a[3] - a[1])
    area_b = max(0., b[2] - b[0]) * max(0., b[3] - b[1])
    return intersection / (area_a + area_b - intersection) if area_a + area_b - intersection else 0.


def match_counts(predictions, targets, *, threshold, match_iou=.5):
    """Confidence-ordered, class-aware greedy matching at a fixed operating threshold."""
    chosen = sorted((p for p in predictions if p["confidence"] >= threshold),
                    key=lambda p: p["confidence"], reverse=True)
    matched = set()
    tp = 0
    for prediction in chosen:
        candidates = [(iou(prediction["bbox"], target["bbox"]), index)
                      for index, target in enumerate(targets)
                      if index not in matched and prediction["class_id"] == target["class_id"]]
        if candidates:
            overlap, index = max(candidates)
            if overlap >= match_iou:
                tp += 1
                matched.add(index)
    return {"tp": tp, "fp": len(chosen) - tp, "fn": len(targets) - tp}


def require_best(path):
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Training did not produce a nonempty best.pt: {path}")
    return file_hash(path)


def val_signature(data):
    report = validate_dataset(data)
    if not report["valid"]:
        raise ValueError(f"Validation dataset is invalid: {report['errors']}")
    return report["val_sha256"]


def evaluate(weights, data, output, *, expected_val, stage, imgsz, device, factory=None):
    if val_signature(data) != expected_val:
        raise ValueError("Validation set changed before evaluation")
    if factory is None:
        from ultralytics import YOLO
        factory = YOLO
    model = factory(str(weights))
    names = model.names
    root = read_definition(data)
    from PIL import Image
    from experiments.f101_yolo_benchmark import detections_from

    images = sorted((root / "images" / "val").glob("*"))
    images = [p for p in images if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    probe_start = time.perf_counter()
    results = model.predict([str(p) for p in images], imgsz=imgsz, conf=.001, iou=.7,
                            max_det=300, device=device, verbose=False, save=False, stream=False)
    probe_ms = (time.perf_counter() - probe_start) * 1000
    if len(results) != len(images):
        raise RuntimeError("Prediction result count does not match val image count")
    predictions, targets = [], []
    max_conf = 0.
    target_max_conf = 0.
    for image, result in zip(images, results):
        detections = detections_from([result])
        predictions.append(detections)
        max_conf = max([max_conf] + [p["confidence"] for p in detections])
        target_max_conf = max([target_max_conf] + [p["confidence"] for p in detections if p["class_id"] in (0, 1)])
        with Image.open(image) as opened:
            width, height = opened.size
        truth = []
        for line in (root / "labels" / "val" / f"{image.stem}.txt").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            cls, x, y, w, h = map(float, line.split())
            truth.append({"class_id": int(cls), "bbox": [(x - w/2)*width, (y - h/2)*height,
                                                         (x + w/2)*width, (y + h/2)*height]})
        targets.append(truth)
    save_json(output / f"{stage}_val_predictions.json", [
        {"image": image.name, "predictions": pred, "targets": truth}
        for image, pred, truth in zip(images, predictions, targets)])
    rows = []
    for conf in (.001, .25):
        start = time.perf_counter()
        metrics = model.val(data=str(data), split="val", imgsz=imgsz, device=device,
                            conf=conf, iou=.7, max_det=300, batch=8, workers=0,
                            plots=False, verbose=False, save_json=False, save_txt=False,
                            project=str(output), name=f"{stage}_conf_{conf}", exist_ok=False,
                            augment=False, rect=True)
        eval_ms = (time.perf_counter() - start) * 1000
        counts = {"tp": 0, "fp": 0, "fn": 0}
        for pred, truth in zip(predictions, targets):
            for key, value in match_counts(pred, truth, threshold=conf).items():
                counts[key] += value
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        rows.append({"stage": stage, "weights": str(weights), "weights_sha256": file_hash(weights),
                     "val_sha256": expected_val, "val_images": len(images), "imgsz": imgsz,
                     "device": device, "conf": conf, "nms_iou": .7, "match_iou": .5,
                     "eval_batch": 8,
                     "mAP50": float(metrics.box.map50), "mAP50_95": float(metrics.box.map),
                     "precision": tp / (tp + fp) if tp + fp else 0.,
                     "recall": tp / (tp + fn) if tp + fn else 0., **counts,
                     "ultralytics_precision_best_f1": float(metrics.box.mp),
                     "ultralytics_recall_best_f1": float(metrics.box.mr),
                     "max_conf": max_conf, "target_id_max_conf": target_max_conf,
                     "max_conf_probe_threshold": .001, "eval_wall_ms": eval_ms,
                     "predict_probe_wall_ms": probe_ms,
                     "model_names_json": json.dumps(names, ensure_ascii=False)})
    if val_signature(data) != expected_val:
        raise ValueError("Validation set changed during evaluation")
    return rows


def run(args):
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    out = args.output_dir or project_path(f"artifacts/f104/{run_id}")
    out.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True,
                        handlers=[logging.FileHandler(out / "execution.log", encoding="utf-8"), logging.StreamHandler()])
    report = {"run_id": run_id, "status": "validating", "data": str(args.data),
              "initial_weights": str(args.weights), "epochs": args.epochs, "imgsz": args.imgsz,
              "batch": args.batch, "device": args.device, "seed": args.seed,
              "python": platform.python_version(), "os": platform.platform()}
    try:
        save_json(out / "run.json", report)
        if min(args.epochs, args.imgsz, args.batch) < 1:
            raise ValueError("epochs, imgsz and batch must be positive")
        if not args.weights.is_file() or args.weights.stat().st_size == 0:
            raise FileNotFoundError("A local pretrained yolo11n.pt file is required; no automatic download")
        original = validate_dataset(args.data)
        save_json(out / "original_validation.json", original)
        if not original["valid"]:
            raise ValueError(f"Dataset validation failed: {original['errors']}")
        data = stage_dataset(original, out / "dataset")
        staged = validate_dataset(data)
        save_json(out / "staged_validation.json", staged)
        if not staged["valid"] or staged["dataset_sha256"] != original["dataset_sha256"]:
            raise ValueError("Working dataset differs from the original")
        # Keep any library settings, plots, labels caches or incidental downloads private.
        os.environ["YOLO_CONFIG_DIR"] = str(out / "ultralytics_config")
        (out / "ultralytics_config").mkdir()
        import torch
        import ultralytics
        from ultralytics import YOLO
        torch.set_num_threads(8)
        report.update(ultralytics_version=ultralytics.__version__, torch_version=torch.__version__,
                      cuda_available=torch.cuda.is_available(), weights_sha256=file_hash(args.weights),
                      dataset_sha256=original["dataset_sha256"], val_sha256=original["val_sha256"])
        eval_dir = out / "evaluation"
        eval_dir.mkdir()
        report["status"] = "evaluating_before"
        save_json(out / "run.json", report)
        rows = evaluate(args.weights, data, eval_dir, expected_val=original["val_sha256"],
                        stage="before", imgsz=args.imgsz, device=args.device)
        write_csv(out / "finetune_compare.csv", rows)
        report["status"] = "training"
        save_json(out / "run.json", report)
        logging.info("Training starts: epochs=%s imgsz=%s batch=%s", args.epochs, args.imgsz, args.batch)
        model = YOLO(str(args.weights))
        start = time.perf_counter()
        model.train(data=str(data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                    device=args.device, workers=0, seed=args.seed, deterministic=True,
                    patience=0, amp=False, cache=False, plots=False, verbose=False,
                    project=str(out), name="train", exist_ok=False, save=True,
                    pretrained=True, optimizer="auto")
        report["train_wall_seconds"] = time.perf_counter() - start
        trainer = model.trainer
        best = Path(trainer.best)
        report.update(best_pt=str(best), best_sha256=require_best(best),
                      optimizer_class=type(trainer.optimizer).__name__)
        with (trainer.save_dir / "results.csv").open(encoding="utf-8", newline="") as stream:
            history = [{key.strip(): value for key, value in row.items()} for row in csv.DictReader(stream)]
        report["completed_epochs"] = len(history)
        if len(history) != args.epochs:
            raise RuntimeError("Training did not complete the requested number of epochs")
        write_csv(out / "training_history.csv", history)
        save_json(out / "training_args.json", vars(trainer.args))
        report["status"] = "evaluating_after"
        save_json(out / "run.json", report)
        rows.extend(evaluate(best, data, eval_dir, expected_val=original["val_sha256"],
                             stage="after", imgsz=args.imgsz, device=args.device))
        write_csv(out / "finetune_compare.csv", rows)
        if validate_dataset(args.data)["dataset_sha256"] != original["dataset_sha256"]:
            raise ValueError("Original dataset contents changed")
        if file_hash(args.weights) != report["weights_sha256"]:
            raise ValueError("Initial weights changed")
        report["status"] = "completed"
        logging.info("F104 completed: %.3f seconds training; best=%s", report["train_wall_seconds"], best)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return 0
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        logging.exception("F104 failed; actual partial outputs retained: %s", out)
        return 1
    finally:
        save_json(out / "run.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=project_path, default=DEFAULT_DATA)
    parser.add_argument("--weights", type=project_path, default=Settings.from_env().weights)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=project_path)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
