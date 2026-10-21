"""Paths resolve against the project, without modifying supplied materials."""

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


@dataclass(frozen=True)
class Settings:
    weights: Path
    image: Path
    device: str
    vlm_model: Path
    mmproj: Path

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            project_path(os.getenv("YOLO_MODEL_PATH", "yolo11n.pt")),
            project_path(os.getenv("SAMPLE_IMAGE_PATH", "source/2회차_day1/samples/sample.jpg")),
            os.getenv("MODEL_DEVICE", "cpu"),
            project_path(os.getenv("VLM_MODEL_PATH", "models/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf")),
            project_path(os.getenv("VLM_MMPROJ_PATH", "models/mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf")),
        )
