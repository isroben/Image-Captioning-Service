import time
from pathlib import Path
import numpy as np
from optimum.onnxruntime import ORTModelForVision2Seq
from transformers import AutoModelForVision2Seq, AutoTokenizer, ViTImageProcessor

MODEL_NAME = "nlpconnect/vit-gpt2-image-captioning"
ONNX_DIR = Path("models/vit-gpt2-onnx")

# 1. Load the already-exported ONNX model (or export if missing)
if not ONNX_DIR.exists():
    print("Exporting ViT-GPT2 model to ONNX...")
    ort_model = ORTModelForVision2Seq.from_pretrained(MODEL_NAME, export=True)
    ort_model.save_pretrained(ONNX_DIR)
else:
    ort_model = ORTModelForVision2Seq.from_pretrained(ONNX_DIR)

image_processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
pt_model = AutoModelForVision2Seq.from_pretrained(MODEL_NAME).eval()

# Dummy RGB image for benchmarking
dummy_image = np.random.randint(0, 255, (384, 384, 3), dtype=np.uint8)
pixel_values = image_processor(images=dummy_image, return_tensors="pt").pixel_values

def benchmark(model, runs: int = 5) -> float:
    # Warmup run
    _ = model.generate(pixel_values, max_new_tokens=30)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        _ = model.generate(pixel_values, max_new_tokens=30)
        times.append((time.perf_counter() - t0) * 1000)
    return round(sum(times) / len(times), 2)

pt_ms = benchmark(pt_model)
onnx_ms = benchmark(ort_model)

print(f"PyTorch Avg Latency:      {pt_ms} ms")
print(f"ONNX Runtime Avg Latency: {onnx_ms} ms")
print(f"Speedup:                  {round(pt_ms / onnx_ms, 2)}x")