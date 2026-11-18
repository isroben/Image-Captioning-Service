import cv2
import numpy as np
from fastapi.testclient import TestClient

from src.data.preprocessing import decodePreprocessImg
from src.serving.app import app


class DummyCaptioner:
    """Lightweight mock so tests run in milliseconds without loading weights."""
    def predict(self, rgb_image: np.ndarray) -> str:
        return "a test caption"


def _make_sample_jpeg_bytes(width: int = 800, height: int = 600) -> bytes:
    img = np.zeros((height, width, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def test_preprocessing_resizes_and_converts():
    raw_bytes = _make_sample_jpeg_bytes(width=800, height=400)
    processed = decodePreprocessImg(raw_bytes, max_size=384)
    assert processed.shape == (192, 384, 3)
    assert processed.dtype == np.uint8


def test_preprocessing_rejects_invalid_bytes():
    try:
        decodePreprocessImg(b"not-an-image")
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_health_and_caption_endpoints():
    # Inject dummy model directly into app.state
    from src.serving.app import loadConfig

    app.state.config = loadConfig()
    app.state.captioner = DummyCaptioner()

    client = TestClient(app)

    # 1. Check /health
    health_res = client.get("/health")
    assert health_res.status_code == 200
    assert health_res.json() == {"status": "ok"}

    # 2. Check /caption
    img_bytes = _make_sample_jpeg_bytes()
    res = client.post(
        "/caption",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["caption"] == "a test caption"
    assert "latency_ms" in body