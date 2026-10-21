"""UI-enabled entry point wrapping the existing FastAPI app and model manager."""

import logging
import threading
import time
from io import BytesIO
from pathlib import Path

from fastapi import Query, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.main import YoloResponse, create_app as create_api_app
from app.model_manager import InvalidImage, ModelBusy, ModelManager, ModelUnavailable
from experiments.f101_yolo_benchmark import detections_from

STATIC = Path(__file__).parent / "static"


class Detection(BaseModel):
    bbox: list[float] = Field(min_length=4, max_length=4)
    class_id: int
    class_name: str = Field(alias="class")
    confidence: float = Field(ge=0, le=1)


class UIYoloResponse(YoloResponse):
    detections: list[Detection] | None = None
    image_width: int | None = None
    image_height: int | None = None


class CaptureYolo:
    def __init__(self, model, storage):
        self.model, self.storage = model, storage

    def predict(self, *args, **kwargs):
        results = self.model.predict(*args, **kwargs)
        self.storage.detections = detections_from(results)
        return results


class UIModelManager(ModelManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ui_local = threading.local()

    def get_model(self, name):
        model, elapsed, cold = super().get_model(name)
        if name == "yolo":
            model = CaptureYolo(model, self._ui_local)
        return model, elapsed, cold

    def infer(self, name, payload):
        self._ui_local.detections = []
        result, timing = super().infer(name, payload)
        if name == "yolo":
            with Image.open(BytesIO(payload)) as image:
                timing["image_width"], timing["image_height"] = image.size
            timing["detections"] = self._ui_local.detections
        return result, timing


def create_ui_app(manager=None):
    manager = manager or UIModelManager()
    app = create_api_app(manager)
    original = next(route for route in app.routes if getattr(route, "path", None) == "/api/yolo-detect")
    app.router.routes.remove(original)

    @app.post("/api/yolo-detect", response_model=UIYoloResponse, response_model_exclude_none=True)
    async def yolo(request: Request, include_detections: bool = Query(False)):
        if not include_detections:
            # Preserve the original two-key body and the original error behavior.
            return await original.endpoint(request)
        start = time.perf_counter()
        try:
            content_type = request.headers.get("content-type", "").split(";")[0].strip().lower()
            if content_type not in ("image/jpeg", "application/octet-stream", ""):
                raise InvalidImage("Send raw JPEG bytes with Content-Type: image/jpeg")
            payload = bytearray()
            async for chunk in request.stream():
                payload.extend(chunk)
                if len(payload) > manager.options.max_bytes:
                    raise InvalidImage("JPEG body exceeds configured size limit")
            result, timing = await run_in_threadpool(manager.infer, "yolo", bytes(payload))
            body = UIYoloResponse(**result, elapsed_ms=(time.perf_counter() - start) * 1000,
                                  detections=timing["detections"],
                                  image_width=timing["image_width"], image_height=timing["image_height"])
            return JSONResponse(body.model_dump(by_alias=True, exclude_none=True),
                                headers={"X-Model-Inference-Ms": f"{timing['inference_ms']:.6f}",
                                         "X-Model-Load-Ms": f"{timing['load_ms']:.6f}",
                                         "X-Model-Queue-Ms": f"{timing['queue_ms']:.6f}",
                                         "X-Cold-Load": str(timing["cold_load"]).lower(),
                                         "X-Finish-Reason": timing["finish_reason"]})
        except InvalidImage as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        except (ModelBusy, ModelUnavailable) as exc:
            return JSONResponse({"error": str(exc)}, status_code=503)
        except Exception:
            logging.exception("UI YOLO inference failed")
            return JSONResponse({"error": "Model inference failed; check server logs"}, status_code=500)

    app.mount("/ui-assets", StaticFiles(directory=STATIC), name="ui-assets")

    @app.get("/", include_in_schema=False)
    async def index():
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})

    return app


app = create_ui_app()
