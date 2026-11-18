import cv2
import numpy as np
from locust import HttpUser, between, task


def _create_test_image() -> bytes:
    img = np.random.randint(0, 255, (384, 384, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


class CaptionServiceUser(HttpUser):
    wait_time = between(0.5, 1.5)

    def on_start(self):
        self.image_bytes = _create_test_image()

    @task(1)
    def check_health(self):
        self.client.get("/health")

    @task(4)
    def caption_image(self):
        self.client.post(
            "/caption",
            files={"file": ("load_test.jpg", self.image_bytes, "image/jpeg")},
        )