from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from time import perf_counter

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response

from app.config import get_settings
from app.image_utils import DecodeImageError
from app.pipeline import DetectionPipeline, InferenceError, NoDetectionError


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.pipeline = DetectionPipeline.from_settings(settings)
    yield


app = FastAPI(title="Wardrobe Detection ML Service", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/detect")
async def detect(file: UploadFile = File(...)) -> Response:
    request_start = perf_counter()
    image_bytes = await file.read()
    pipeline: DetectionPipeline = app.state.pipeline

    try:
        png_bytes = pipeline.process(image_bytes)
    except DecodeImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except NoDetectionError as exc:
        return Response(content="Ничего не распозналось", media_type="text/plain; charset=utf-8")
    except InferenceError as exc:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail="Internal inference error") from exc

    total_ms = (perf_counter() - request_start) * 1000
    logger.info(
        "request timings: total=%.2fms input_bytes=%d output_bytes=%d",
        total_ms,
        len(image_bytes),
        len(png_bytes),
    )
    return Response(content=png_bytes, media_type="image/png")


@app.post("/cascade")
async def cascade(file: UploadFile = File(...)) -> JSONResponse:
    request_start = perf_counter()
    image_bytes = await file.read()
    pipeline: DetectionPipeline = app.state.pipeline

    try:
        result = pipeline.cascade(image_bytes)
    except DecodeImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except InferenceError as exc:
        logger.exception("Cascade inference failed")
        raise HTTPException(status_code=500, detail="Internal inference error") from exc

    total_ms = (perf_counter() - request_start) * 1000
    logger.info(
        "cascade request timings: total=%.2fms input_bytes=%d decision=%s reason=%s",
        total_ms,
        len(image_bytes),
        result["decision"],
        result["reason"],
    )
    return JSONResponse(content=result)
