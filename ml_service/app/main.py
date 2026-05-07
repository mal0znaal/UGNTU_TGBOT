from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response

from app.config import get_settings
from app.image_utils import DecodeImageError
from app.pipeline import BackgroundRemovalPipeline, InferenceError, NoDetectionError


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.pipeline = BackgroundRemovalPipeline.from_settings(settings)
    yield


app = FastAPI(title="Wardrobe Background Removal ML Service", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/process")
async def process(file: UploadFile = File(...)) -> Response:
    image_bytes = await file.read()
    pipeline: BackgroundRemovalPipeline = app.state.pipeline

    try:
        png_bytes = pipeline.process(image_bytes)
    except DecodeImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except NoDetectionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except InferenceError as exc:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail="Internal inference error") from exc

    return Response(content=png_bytes, media_type="image/png")
