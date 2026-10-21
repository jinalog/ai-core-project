"""F103 FastAPI PoC: raw JPEG input, text JSON output."""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.model_manager import InvalidImage, ModelBusy, ModelManager, ModelUnavailable


class YoloResponse(BaseModel):
    summary: str
    elapsed_ms: float = Field(ge=0)


class VlmResponse(BaseModel):
    description: str
    elapsed_ms: float = Field(ge=0)


def create_app(manager=None):
    manager = manager or ModelManager()

    @asynccontextmanager
    async def lifespan(app):
        yield
        await run_in_threadpool(manager.close)

    app = FastAPI(title="AI Core F103 PoC", version="0.1.0", lifespan=lifespan)
    app.state.models = manager

    @app.get("/api/health")
    async def health():
        return manager.health()

    async def recognize(request, name):
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
            result, timing = await run_in_threadpool(manager.infer, name, bytes(payload))
            elapsed = (time.perf_counter() - start) * 1000
            headers = {"X-Model-Inference-Ms": f"{timing['inference_ms']:.6f}",
                       "X-Model-Load-Ms": f"{timing['load_ms']:.6f}",
                       "X-Model-Queue-Ms": f"{timing['queue_ms']:.6f}",
                       "X-Cold-Load": str(timing["cold_load"]).lower(),
                       "X-Finish-Reason": timing["finish_reason"]}
            # Validate the public response while retaining the required two-key schema.
            schema = YoloResponse if name == "yolo" else VlmResponse
            response = schema(**result, elapsed_ms=elapsed)
            return JSONResponse(response.model_dump(), headers=headers)
        except InvalidImage as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        except (ModelBusy, ModelUnavailable) as exc:
            return JSONResponse({"error": str(exc)}, status_code=503)
        except Exception:
            logging.exception("%s inference request failed", name)
            return JSONResponse({"error": "Model inference failed; check server logs"}, status_code=500)

    @app.post("/api/yolo-detect", response_model=YoloResponse)
    async def yolo(request: Request):
        return await recognize(request, "yolo")

    @app.post("/api/vlm-describe", response_model=VlmResponse)
    async def vlm(request: Request):
        return await recognize(request, "vlm")

    return app


app = create_app()
