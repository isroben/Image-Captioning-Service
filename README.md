# Image Captioning Service

A production-ready image captioning microservice built with **FastAPI**, **PyTorch**, **Hugging Face Transformers (`Salesforce/blip-image-captioning-base`)**, and **OpenCV**, packaged in **Docker** with unit tests and load-test benchmarks.

## Architecture & Project Structure

```text
image-captioning-service/
├── configs/
│   └── config.yaml             # Model, inference, and preprocessing parameters
├── scripts/
│   └── export_onnx.py          # ONNX export and PyTorch vs. ONNX latency benchmark
├── src/
│   ├── data/
│   │   └── preprocessing.py    # OpenCV byte decoding, validation, resizing, BGR->RGB
│   ├── models/
│   │   └── captioner.py        # Hugging Face BLIP inference wrapper
│   └── serving/
│       └── app.py              # FastAPI app with lifespan loading, Pydantic schemas, JSON logging
├── tests/
│   ├── locustfile.py           # Concurrent load testing script
│   └── test_api.py             # Unit tests for preprocessing and API endpoints
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── requirements.txt
```

## Quickstart

### 1. Run with Docker (Recommended)
```bash
docker-compose up --build
```
The API will be available at `http://localhost:8000` (interactive Swagger docs at `http://localhost:8000/docs`). Model weights are cached in a persistent Docker volume (`hf_cache`).

### 2. Run Locally
```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

uvicorn src.serving.app:app --host 0.0.0.0 --port 8000
```

## API Endpoints

### `GET /health`
Liveness check used by Docker health probes.
```json
{
  "status": "ok"
}
```

### `POST /caption`
Accepts an uploaded image (`multipart/form-data`), validates and resizes it via OpenCV, and returns a generated caption with inference latency.

```bash
curl -X POST "http://localhost:8000/caption" \
  -H "accept: application/json" \
  -F "file=@sample.jpg"
```

**Response:**
```json
{
  "caption": "a person using a laptop with a keyboard",
  "latency_ms": 1043.25
}
```

## Testing

Run unit tests (uses mocked model weights so CI runs in ~5 seconds without downloading checkpoints):
```bash
pytest -v
```

Run a 30-second Locust load test against the running service:
```bash
locust -f tests/locustfile.py --headless -u 5 -r 1 --run-time 30s --host http://localhost:8000
```

## Benchmarks & Engineering Trade-offs

### 1. Runtime Selection: PyTorch vs. ONNX Runtime (CPU / Apple Silicon)
Before choosing the serving runtime, an unquantized FP32 ONNX export was benchmarked against native PyTorch (`384x384` input, `max_new_tokens=30`, 5-run average):

| Runtime | Avg Latency | Relative Speed | Decision |
| :--- | :---: | :---: | :--- |
| **PyTorch (CPU / Accelerate)** | **385.69 ms** | **1.00x (Baseline)** | **Selected for production** |
| ONNX Runtime (FP32, unmerged decoder) | 505.99 ms | 0.76x | Rejected (higher session overhead on ARM64) |

### 2. Load Test Results (Docker CPU Container, 5 Concurrent Users, 30s)

| Endpoint | Requests | Failure Rate | Min Latency | p50 (Median) | p95 Latency | Throughput |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `POST /caption` | 19 | 0.00% | 1,043 ms | 4,000 ms | 11,000 ms | 0.72 req/s |
| `GET /health` | 1 | 0.00% | 1,053 ms | 1,053 ms | 1,100 ms | 0.04 req/s |

**Key Takeaways & Future Scaling:**
* **Zero failures** under concurrent load with stable container memory.
* **Bottleneck identified:** Synchronous CPU-bound token generation causes head-of-line blocking under concurrent requests (evidenced by `/health` waiting ~1.05s behind active inference).
* **Next steps for high-throughput production:** Deploy on CUDA GPU (`T4`/`L4`), enable dynamic request batching, or decouple inference behind an async Redis/Celery worker queue.
