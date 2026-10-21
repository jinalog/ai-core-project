import asyncio
import csv
import json
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
from fastapi.testclient import TestClient
from PIL import Image

from app.main import create_app
from app.model_manager import ApiSettings, ModelManager
from experiments.f103_api_latency_compare import measure


def jpeg():
    stream = BytesIO()
    Image.new("RGB", (8, 6)).save(stream, format="JPEG")
    return stream.getvalue()


def fake_manager(yolo=None, vlm=None, options=None):
    yolo = yolo or Mock(predict=Mock(return_value=[SimpleNamespace(boxes=[])]))
    llm = vlm or Mock(create_chat_completion=Mock(return_value={
        "choices": [{"message": {"content": "사람과 버스가 보입니다."}, "finish_reason": "stop"}]}))
    manager = ModelManager(yolo_loader=Mock(return_value=yolo),
                           vlm_loader=Mock(return_value=(llm, Mock(), {})), options=options)
    return manager


class F103ApiTests(unittest.TestCase):
    def test_health_is_lazy_and_normal_jpeg_schemas(self):
        manager = fake_manager()
        with TestClient(create_app(manager)) as client:
            health = client.get("/api/health").json()
            self.assertTrue(all(s["status"] == "unloaded" for s in health["models"].values()))
            for endpoint, key in (("yolo-detect", "summary"), ("vlm-describe", "description")):
                response = client.post(f"/api/{endpoint}", content=jpeg(), headers={"Content-Type": "image/jpeg"})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(set(response.json()), {key, "elapsed_ms"})
                self.assertIsInstance(response.json()[key], str)
                self.assertGreater(response.json()["elapsed_ms"], 0)
                self.assertEqual(response.headers["x-cold-load"], "true")
                self.assertGreater(float(response.headers["x-model-inference-ms"]), 0)
            self.assertTrue(all(s["load_count"] == 1 for s in client.get("/api/health").json()["models"].values()))

    def test_invalid_empty_corrupt_non_jpeg_and_oversize(self):
        manager = fake_manager(options=replace(ApiSettings(), max_bytes=1024, max_pixels=100))
        png = BytesIO()
        Image.new("RGB", (2, 2)).save(png, format="PNG")
        with TestClient(create_app(manager)) as client:
            for endpoint in ("yolo-detect", "vlm-describe"):
                for content in (b"", b"broken JPEG", png.getvalue(), jpeg()[:50], b"x" * 1025):
                    with self.subTest(endpoint=endpoint, length=len(content)):
                        response = client.post(f"/api/{endpoint}", content=content)
                        self.assertEqual(response.status_code, 400)
                        self.assertEqual(set(response.json()), {"error"})
                self.assertEqual(client.post(f"/api/{endpoint}", content=jpeg(),
                                             headers={"Content-Type": "application/json"}).status_code, 400)
            manager._loaders["yolo"].assert_not_called()
            manager._loaders["vlm"].assert_not_called()

    def test_loading_failure_cached_and_sanitized(self):
        manager = fake_manager()
        manager._loaders["yolo"].side_effect = RuntimeError("secret path not for response")
        with TestClient(create_app(manager)) as client:
            for _ in range(2):
                response = client.post("/api/yolo-detect", content=jpeg())
                self.assertEqual(response.status_code, 503)
                self.assertNotIn("secret", response.text)
            self.assertEqual(manager._loaders["yolo"].call_count, 1)
            self.assertEqual(client.get("/api/health").json()["models"]["yolo"]["status"], "failed")

    def test_inference_exception_returns_500_and_releases_lock(self):
        llm = Mock()
        llm.create_chat_completion.side_effect = RuntimeError("internal inference failure")
        manager = fake_manager(vlm=llm)
        with TestClient(create_app(manager)) as client:
            response = client.post("/api/vlm-describe", content=jpeg())
            self.assertEqual(response.status_code, 500)
            self.assertEqual(manager.health()["models"]["vlm"]["active_inferences"], 0)
            self.assertTrue(manager._infer_locks["vlm"].acquire(blocking=False))
            manager._infer_locks["vlm"].release()

    def test_queue_timeout_returns_503(self):
        manager = fake_manager(options=replace(ApiSettings(), queue_timeout_seconds=.01))
        with TestClient(create_app(manager)) as client:
            manager._infer_locks["vlm"].acquire()
            try:
                self.assertEqual(client.post("/api/vlm-describe", content=jpeg()).status_code, 503)
            finally:
                manager._infer_locks["vlm"].release()

    def test_double_check_single_initialization_for_both_models(self):
        for name in ("yolo", "vlm"):
            manager = fake_manager()
            barrier = threading.Barrier(8)
            obj = object()

            def load():
                time.sleep(.03)
                return obj

            manager._loaders[name] = Mock(side_effect=load)

            def get():
                barrier.wait(timeout=2)
                return manager.get_model(name)[0]

            with ThreadPoolExecutor(max_workers=8) as executor:
                results = list(executor.map(lambda _: get(), range(8)))
            self.assertTrue(all(result is obj for result in results))
            self.assertEqual(manager._loaders[name].call_count, 1)


class F103ConcurrencyTests(unittest.IsolatedAsyncioTestCase):
    async def test_event_loop_responds_while_native_inference_waits(self):
        started, release = threading.Event(), threading.Event()
        llm = Mock()

        def infer(**kwargs):
            started.set()
            if not release.wait(timeout=5):
                raise TimeoutError("Test worker was not released")
            return {"choices": [{"message": {"content": "한국어 설명"}, "finish_reason": "stop"}]}

        llm.create_chat_completion.side_effect = infer
        manager = fake_manager(vlm=llm)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(manager)), base_url="http://test") as client:
            task = asyncio.create_task(client.post("/api/vlm-describe", content=jpeg()))
            try:
                self.assertTrue(await asyncio.to_thread(started.wait, 2))
                response = await asyncio.wait_for(client.get("/api/health"), timeout=.5)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["models"]["vlm"]["active_inferences"], 1)
            finally:
                release.set()
            self.assertEqual((await task).status_code, 200)

    async def test_concurrent_requests_load_once_and_serialize_vlm(self):
        state = {"active": 0, "peak": 0}
        lock = threading.Lock()
        llm = Mock()

        def infer(**kwargs):
            with lock:
                state["active"] += 1
                state["peak"] = max(state["peak"], state["active"])
            time.sleep(.02)
            with lock:
                state["active"] -= 1
            return {"choices": [{"message": {"content": "한국어 설명"}, "finish_reason": "stop"}]}

        llm.create_chat_completion.side_effect = infer
        manager = fake_manager(vlm=llm)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(manager)), base_url="http://test") as client:
            responses = await asyncio.gather(*(client.post("/api/vlm-describe", content=jpeg()) for _ in range(6)))
        self.assertTrue(all(response.status_code == 200 for response in responses))
        self.assertEqual(manager._loaders["vlm"].call_count, 1)
        self.assertEqual(state["peak"], 1)

    async def test_yolo_and_vlm_can_infer_concurrently(self):
        barrier = threading.Barrier(2)
        yolo, vlm = Mock(), Mock()

        def detect(**kwargs):
            barrier.wait(timeout=2)
            return [SimpleNamespace(boxes=[])]

        def describe(**kwargs):
            barrier.wait(timeout=2)
            return {"choices": [{"message": {"content": "한국어 설명"}, "finish_reason": "stop"}]}

        yolo.predict.side_effect, vlm.create_chat_completion.side_effect = detect, describe
        manager = fake_manager(yolo=yolo, vlm=vlm)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(manager)), base_url="http://test") as client:
            responses = await asyncio.gather(client.post("/api/yolo-detect", content=jpeg()),
                                             client.post("/api/vlm-describe", content=jpeg()))
        self.assertTrue(all(r.status_code == 200 for r in responses))

    async def test_measurement_scenarios_schemas_and_parallel_dispatch(self):
        # Client-side orchestration test, not a hardware performance measurement.
        active, peak = 0, 0

        async def handler(request):
            nonlocal active, peak
            if request.method == "GET":
                return httpx.Response(200, json={"models": {"yolo": {"status": "unloaded", "load_attempts": 0}}})
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(.01)
            active -= 1
            key = "summary" if "yolo" in request.url.path else "description"
            return httpx.Response(200, json={key: "테스트", "elapsed_ms": 1},
                                  headers={"X-Model-Inference-Ms": "0.5", "X-Model-Load-Ms": "0",
                                           "X-Model-Queue-Ms": "0", "X-Cold-Load": "false", "X-Finish-Reason": "stop"})

        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://test") as client:
                rows, groups = await measure(client, jpeg(), out, {"run_id": "unit-test"})
            self.assertEqual(len(rows), 6)
            self.assertEqual([g["scenario"] for g in groups], ["cold_sequential", "warm_sequential", "warm_parallel"])
            self.assertEqual(peak, 2)
            with (out / "api_requests.csv").open(encoding="utf-8", newline="") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 6)


if __name__ == "__main__":
    unittest.main()
