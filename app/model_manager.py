"""Thread-safe lazy model loading and serialized inference per model."""

import base64
import logging
import os
import threading
import time
from collections import Counter
from dataclasses import dataclass
from io import BytesIO

from PIL import Image, UnidentifiedImageError

from app.config import Settings
from experiments.f101_yolo_benchmark import detections_from
from experiments.f102_vlm_prompt_compare import PROMPTS, load_vlm, messages_for


class InvalidImage(ValueError):
    pass


class ModelUnavailable(RuntimeError):
    pass


class ModelBusy(RuntimeError):
    pass


@dataclass(frozen=True)
class ApiSettings:
    max_bytes: int = 10 * 1024 * 1024
    max_pixels: int = 20_000_000
    queue_timeout_seconds: float = 600
    vlm_n_ctx: int = 8192
    vlm_threads: int = 8
    vlm_max_tokens: int = 320
    vlm_seed: int = 42

    @classmethod
    def from_env(cls):
        value = cls(
            max_bytes=int(os.getenv("API_MAX_IMAGE_BYTES", str(cls.max_bytes))),
            max_pixels=int(os.getenv("API_MAX_IMAGE_PIXELS", str(cls.max_pixels))),
            queue_timeout_seconds=float(os.getenv("MODEL_QUEUE_TIMEOUT_SECONDS", "600")),
            vlm_n_ctx=int(os.getenv("VLM_N_CTX", "8192")),
            vlm_threads=int(os.getenv("VLM_N_THREADS", "8")),
            vlm_max_tokens=int(os.getenv("VLM_MAX_TOKENS", "320")),
            vlm_seed=int(os.getenv("VLM_SEED", "42")),
        )
        if min(value.max_bytes, value.max_pixels, value.queue_timeout_seconds,
               value.vlm_n_ctx, value.vlm_threads, value.vlm_max_tokens) <= 0:
            raise ValueError("API limits, context, threads, tokens and queue timeout must be positive")
        return value


def decode_jpeg(payload: bytes, settings: ApiSettings):
    if not payload:
        raise InvalidImage("JPEG body must not be empty")
    if len(payload) > settings.max_bytes:
        raise InvalidImage("JPEG body exceeds configured size limit")
    try:
        with Image.open(BytesIO(payload)) as image:
            if image.format != "JPEG":
                raise InvalidImage("Only JPEG images are accepted")
            if image.width * image.height > settings.max_pixels:
                raise InvalidImage("JPEG dimensions exceed configured pixel limit")
            image.verify()
        with Image.open(BytesIO(payload)) as image:
            image.load()  # Verify full JPEG decoding, not just its header.
            return image.convert("RGB")
    except InvalidImage:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise InvalidImage("Invalid or corrupted JPEG image") from exc


class ModelManager:
    def __init__(self, config=None, options=None, *, yolo_loader=None, vlm_loader=None):
        self.config = config or Settings.from_env()
        self.options = options or ApiSettings.from_env()
        self._models = {"yolo": None, "vlm": None}
        self._load_locks = {name: threading.Lock() for name in self._models}
        self._infer_locks = {name: threading.Lock() for name in self._models}
        self._state_lock = threading.Lock()
        self._state = {name: {"status": "unloaded", "load_attempts": 0, "load_count": 0,
                              "active_inferences": 0, "completed_inferences": 0}
                       for name in self._models}
        self._loaders = {"yolo": yolo_loader or self._load_yolo,
                         "vlm": vlm_loader or self._load_vlm}

    def _load_yolo(self):
        if not self.config.weights.is_file() or self.config.weights.stat().st_size == 0:
            raise FileNotFoundError("Local YOLO weight file is missing or empty")
        from ultralytics import YOLO
        return YOLO(str(self.config.weights), task="detect")

    def _load_vlm(self):
        for path in (self.config.vlm_model, self.config.mmproj):
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError("Local VLM weight file is missing or empty")
        llm, handler, timings = load_vlm(self.config.vlm_model, self.config.mmproj,
                                       n_ctx=self.options.vlm_n_ctx,
                                       n_threads=self.options.vlm_threads, seed=self.options.vlm_seed)
        return llm, handler, timings

    def health(self):
        with self._state_lock:
            return {"status": "ok", "models": {name: dict(state) for name, state in self._state.items()},
                    "configuration": {"yolo_device": self.config.device, "yolo_conf": .25,
                                      "yolo_imgsz": 640, "vlm_n_ctx": self.options.vlm_n_ctx,
                                      "vlm_threads": self.options.vlm_threads,
                                      "vlm_max_tokens": self.options.vlm_max_tokens,
                                      "vlm_seed": self.options.vlm_seed, "vlm_temperature": 0.0,
                                      "queue_timeout_seconds": self.options.queue_timeout_seconds}}

    def get_model(self, name):
        with self._state_lock:
            model = self._models[name]
        if model is not None:
            return model, 0.0, False
        if not self._load_locks[name].acquire(timeout=self.options.queue_timeout_seconds):
            raise ModelBusy("Model loading queue timeout")
        try:
            with self._state_lock:
                if self._models[name] is not None:
                    return self._models[name], 0.0, False
                # Failed initialization is cached; retry requires restarting the server.
                if self._state[name]["status"] == "failed":
                    raise ModelUnavailable(f"{name} initialization failed; check server logs")
                self._state[name]["status"] = "loading"
                self._state[name]["load_attempts"] += 1
            start = time.perf_counter()
            try:
                model = self._loaders[name]()
            except Exception as exc:
                logging.exception("%s model initialization failed", name)
                with self._state_lock:
                    self._state[name]["status"] = "failed"
                raise ModelUnavailable(f"{name} initialization failed; check server logs") from exc
            elapsed = (time.perf_counter() - start) * 1000
            with self._state_lock:
                self._models[name] = model
                self._state[name]["status"] = "loaded"
                self._state[name]["load_count"] += 1
            return model, elapsed, True
        finally:
            self._load_locks[name].release()

    def infer(self, name: str, payload: bytes):
        # Decoding runs in the caller's worker thread and precedes all model loading.
        image = decode_jpeg(payload, self.options)
        start_wait = time.perf_counter()
        lock = self._infer_locks[name]
        if not lock.acquire(timeout=self.options.queue_timeout_seconds):
            image.close()
            raise ModelBusy("Model inference queue timeout")
        queue_ms = (time.perf_counter() - start_wait) * 1000
        active = False
        try:
            model, load_ms, cold = self.get_model(name)
            with self._state_lock:
                self._state[name]["active_inferences"] += 1
            active = True
            start = time.perf_counter()
            if name == "yolo":
                if self.config.device != "cpu":
                    import torch
                    torch.cuda.synchronize(int(self.config.device))
                detections = detections_from(model.predict(source=image, conf=0.25, imgsz=640,
                                                           device=self.config.device,
                                                           verbose=False, save=False))
                if self.config.device != "cpu":
                    torch.cuda.synchronize(int(self.config.device))
                counts = Counter(d["class"] for d in detections)
                text = ", ".join(f"{label} {count}개" for label, count in sorted(counts.items())) or "검출된 객체 없음"
                result = {"summary": text}
                finish_reason = "stop"
            else:
                llm, _, _ = model
                data_url = "data:image/jpeg;base64," + base64.b64encode(payload).decode("ascii")
                response = llm.create_chat_completion(
                    messages=messages_for(PROMPTS[0], data_url), temperature=0.0,
                    seed=self.options.vlm_seed, max_tokens=self.options.vlm_max_tokens, stream=False)
                choice = response["choices"][0]
                text = choice["message"]["content"]
                if not isinstance(text, str) or not text.strip():
                    raise RuntimeError("VLM returned no description")
                finish_reason = choice.get("finish_reason", "unknown")
                result = {"description": text}
            infer_ms = (time.perf_counter() - start) * 1000
            with self._state_lock:
                self._state[name]["completed_inferences"] += 1
            return result, {"inference_ms": infer_ms, "load_ms": load_ms,
                            "queue_ms": queue_ms, "cold_load": cold, "finish_reason": finish_reason}
        finally:
            if active:
                with self._state_lock:
                    self._state[name]["active_inferences"] -= 1
            lock.release()
            image.close()

    def close(self):
        # Called after the ASGI server drains requests, outside the event loop.
        for name, model in self._models.items():
            if model is not None and name == "vlm":
                llm, handler, _ = model
                handler._exit_stack.close()
                llm.close()
