"""F103: real HTTP cold, warmed sequential and warmed parallel API timings."""

import argparse
import asyncio
import csv
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import httpx

from app.config import Settings, project_path

ENDPOINTS = (("yolo", "/api/yolo-detect", "summary"),
             ("vlm", "/api/vlm-describe", "description"))


def write_csv(path, rows):
    if rows:
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


async def measure(client, image, out, report):
    rows, groups = [], []
    before = (await client.get("/api/health")).raise_for_status().json()
    report["health_before"] = before
    if any(s["status"] != "unloaded" or s["load_attempts"] for s in before["models"].values()):
        raise ValueError("Cold measurement requires a fresh single-worker server with unloaded models")

    async def call(scenario, model, endpoint, key):
        start = time.perf_counter()
        response = await client.post(endpoint, content=image, headers={"Content-Type": "image/jpeg"})
        client_ms = (time.perf_counter() - start) * 1000
        body = response.json()
        row = {"run_id": report["run_id"], "scenario": scenario, "model": model,
               "endpoint": endpoint, "status_code": response.status_code,
               "client_ms": client_ms, "api_elapsed_ms": body.get("elapsed_ms"),
               "model_inference_ms": response.headers.get("x-model-inference-ms"),
               "model_load_ms": response.headers.get("x-model-load-ms"),
               "model_queue_ms": response.headers.get("x-model-queue-ms"),
               "cold_load": response.headers.get("x-cold-load"),
               "finish_reason": response.headers.get("x-finish-reason"),
               "response_json": json.dumps(body, ensure_ascii=False)}
        rows.append(row)
        write_csv(out / "api_requests.csv", rows)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        response.raise_for_status()
        if set(body) != {key, "elapsed_ms"} or not isinstance(body[key], str) or not body[key].strip():
            raise ValueError(f"Invalid API response schema: {endpoint}")
        return row

    for scenario in ("cold_sequential", "warm_sequential", "warm_parallel"):
        start = time.perf_counter()
        if scenario == "warm_parallel":
            result = await asyncio.gather(*(call(scenario, *endpoint) for endpoint in ENDPOINTS),
                                          return_exceptions=True)
            for item in result:
                if isinstance(item, BaseException):
                    raise item
        else:
            for endpoint in ENDPOINTS:
                await call(scenario, *endpoint)
        group_ms = (time.perf_counter() - start) * 1000
        groups.append({"run_id": report["run_id"], "scenario": scenario,
                       "total_wall_ms": group_ms, "requests": 2})
        write_csv(out / "api_groups.csv", groups)
        report["completed_groups"] = len(groups)
        (out / "run.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    report["health_after"] = (await client.get("/api/health")).raise_for_status().json()
    return rows, groups


async def run(args):
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    out = args.output_dir or project_path(f"artifacts/f103/{run_id}")
    out.mkdir(parents=True, exist_ok=False)
    report = {"run_id": run_id, "status": "started", "base_url": args.base_url,
              "python": platform.python_version(), "os": platform.platform(),
              "timeout_seconds": args.timeout, "image": str(args.image),
              "completed_groups": 0}
    try:
        if args.timeout <= 0:
            raise ValueError("timeout must be positive")
        image = args.image.read_bytes()
        report["image_sha256"] = hashlib.sha256(image).hexdigest()
        async with httpx.AsyncClient(base_url=args.base_url, trust_env=False,
                                     timeout=httpx.Timeout(args.timeout, connect=10)) as client:
            await measure(client, image, out, report)
        report["status"] = "completed"
        return 0
    except Exception as exc:
        import traceback
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        (out / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
        print(f"F103 measurement failed: {exc}", flush=True)
        return 1
    finally:
        (out / "run.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--image", type=project_path, default=Settings.from_env().image)
    parser.add_argument("--output-dir", type=project_path)
    parser.add_argument("--timeout", type=float, default=600)
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
