import json
import logging
import time
from pathlib import Path
from contextlib import asynccontextmanager
import yaml
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from pydantic import BaseModel

from src.data.preprocessing import decodePreprocessImg
from src.models.captioner import ImageCaptioner

# configuring structured JSON-style logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("caption_serviec")

class ModelConfig(BaseModel):
    name: str
    max_new_tokens : int
    device : str

class DataConfig(BaseModel):
    max_image_size : int

class AppConfig(BaseModel):
    model : ModelConfig
    data : DataConfig

class CaptionResponse(BaseModel):
    caption : str
    latency_ms : float

def loadConfig(path : str = "configs/config.yaml") -> AppConfig:
    raw = yaml.safe_load(Path(path).read_text())
    return AppConfig(**raw)

@asynccontextmanager
async def lifespan(app : FastAPI):
    config = loadConfig()
    app.state.config = config
    app.state.captioner = ImageCaptioner(
        model_name=config.model.name,
        device=config.model.device,
        max_new_tokens=config.model.max_new_tokens,
    )
    logger.info(json.dumps({"event" : "Model_loaded", "model" : config.model.name}))
    yield

app = FastAPI(title="Image Captioning Service", lifespan=lifespan)

@app.get("/health")
def health_check():
    return {"status" : "ok"}

@app.post("/caption", response_model=CaptionResponse)
async def generateCaption(request : Request, file : UploadFile = File(...)):
    start_time = time.perf_counter()

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    # image_bytes = await file.read()
    config : AppConfig = request.app.state.config
    captioner : ImageCaptioner = request.app.state.captioner

    try:
        rgb_image = decodePreprocessImg(image_bytes, max_size=config.data.max_image_size)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    caption = captioner.predict(rgb_image)
    latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

    logger.info(json.dumps(
        {
            "event" : "inference_completed",
            "filename" : file.filename,
            "latency" : latency_ms
        }
    ))

    return CaptionResponse(caption=caption, latency_ms=latency_ms)