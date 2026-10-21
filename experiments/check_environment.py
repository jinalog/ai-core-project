"""Check Python, imports, local files and hardware without loading VLM weights."""

import argparse
import importlib
import json
import platform
import sys

from app.config import Settings, project_path

MODULES = ("ultralytics", "llama_cpp", "fastapi", "uvicorn", "huggingface_hub", "torch", "psutil", "PIL")


def check_environment() -> dict:
    config = Settings.from_env()
    imports, loaded = {}, {}
    for name in MODULES:
        try:
            module = importlib.import_module(name)
            loaded[name] = module
            imports[name] = {"ok": True, "version": getattr(module, "__version__", "unknown")}
        except Exception as exc:
            imports[name] = {"ok": False, "error": str(exc)}
    files = {name: {"path": str(path), "exists": path.is_file() and path.stat().st_size > 0}
             for name, path in (("yolo", config.weights), ("sample", config.image),
                                ("vlm", config.vlm_model), ("mmproj", config.mmproj))}
    torch, psutil = loaded.get("torch"), loaded.get("psutil")
    cuda = bool(torch and torch.cuda.is_available())
    valid_device = config.device == "cpu" or (
        cuda and config.device.isdigit() and int(config.device) < torch.cuda.device_count())
    python_ok = sys.version_info[:2] == (3, 12)
    vlm_version_ok = imports.get("llama_cpp", {}).get("version") == "0.3.34"
    return {"passed": python_ok and vlm_version_ok and valid_device
                      and all(x["ok"] for x in imports.values()) and all(x["exists"] for x in files.values()),
            "python": sys.version, "python_ok": python_ok, "executable": sys.executable,
            "os": platform.platform(), "imports": imports, "files": files,
            "llama_cpp_version_ok": vlm_version_ok,
            "device": config.device, "device_valid": valid_device, "cuda_available": cuda,
            "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if cuda else [],
            "ram_total_bytes": psutil.virtual_memory().total if psutil else None,
            "ram_available_bytes": psutil.virtual_memory().available if psutil else None,
            "note": "Model file existence only; VLM loading/inference belongs to F102."}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=project_path, default=project_path("artifacts/environment.json"))
    args = parser.parse_args()
    report = check_environment()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
