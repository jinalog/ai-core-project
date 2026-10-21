import unittest
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import Mock

from fastapi.testclient import TestClient
from PIL import Image

from ui01.server import UIModelManager, create_ui_app


def jpeg():
    stream = BytesIO()
    Image.new("RGB", (8, 6)).save(stream, format="JPEG")
    return stream.getvalue()


def manager():
    box = SimpleNamespace(cls=SimpleNamespace(item=lambda: 0),
                          conf=SimpleNamespace(item=lambda: .8),
                          xyxy=[SimpleNamespace(tolist=lambda: [1, 2, 6, 5])])
    yolo = Mock(predict=Mock(return_value=[SimpleNamespace(boxes=[box], names={0: "person"})]))
    llm = Mock(create_chat_completion=Mock(return_value={"choices": [
        {"message": {"content": "사람이 보입니다."}, "finish_reason": "stop"}]}))
    return UIModelManager(yolo_loader=Mock(return_value=yolo),
                          vlm_loader=Mock(return_value=(llm, Mock(), {})))


class UI01Tests(unittest.TestCase):
    def test_index_assets_and_lazy_health(self):
        models = manager()
        with TestClient(create_ui_app(models)) as client:
            response = client.get("/")
            self.assertEqual(response.status_code, 200)
            self.assertIn("이미지 선택", response.text)
            self.assertEqual(client.get("/ui-assets/app.js").status_code, 200)
            self.assertEqual(client.get("/ui-assets/style.css").status_code, 200)
            self.assertTrue(all(state["load_count"] == 0 for state in models.health()["models"].values()))

    def test_default_api_compatibility_and_optional_boxes(self):
        models = manager()
        with TestClient(create_ui_app(models)) as client:
            old = client.post("/api/yolo-detect", content=jpeg()).json()
            self.assertEqual(set(old), {"summary", "elapsed_ms"})
            response = client.post("/api/yolo-detect?include_detections=true", content=jpeg())
            self.assertEqual(response.status_code, 200)
            body = response.json()
            self.assertEqual(body["summary"], "person 1개")
            self.assertEqual((body["image_width"], body["image_height"]), (8, 6))
            self.assertEqual(body["detections"], [{"bbox": [1., 2., 6., 5.], "class_id": 0,
                                                  "class": "person", "confidence": .8}])
            self.assertEqual(models.health()["models"]["yolo"]["load_count"], 1)

    def test_vlm_unchanged(self):
        with TestClient(create_ui_app(manager())) as client:
            response = client.post("/api/vlm-describe", content=jpeg())
            self.assertEqual(response.status_code, 200)
            self.assertEqual(set(response.json()), {"description", "elapsed_ms"})
            self.assertEqual(response.json()["description"], "사람이 보입니다.")

    def test_optional_route_errors_do_not_load_model(self):
        models = manager()
        with TestClient(create_ui_app(models)) as client:
            for payload in (b"", b"broken JPEG"):
                response = client.post("/api/yolo-detect?include_detections=true", content=payload)
                self.assertEqual(response.status_code, 400)
                self.assertEqual(set(response.json()), {"error"})
            response = client.post("/api/yolo-detect?include_detections=true", content=jpeg(),
                                   headers={"Content-Type": "application/json"})
            self.assertEqual(response.status_code, 400)
            self.assertEqual(models.health()["models"]["yolo"]["load_count"], 0)


if __name__ == "__main__":
    unittest.main()
