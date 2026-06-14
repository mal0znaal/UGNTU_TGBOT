from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from time import perf_counter

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.image_utils import DecodeImageError
from app.pipeline import DetectionPipeline, InferenceError, NoDetectionError


logger = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pipeline = DetectionPipeline.from_settings(get_settings())
    yield


app = FastAPI(title="Wardrobe Detection ML Service", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/process")
async def process(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
) -> JSONResponse:
    request_start = perf_counter()
    image_bytes = await file.read()
    pipeline: DetectionPipeline = app.state.pipeline

    try:
        result = pipeline.process(image_bytes, background_tasks)
    except DecodeImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except NoDetectionError as exc:
        return JSONResponse(content={"decision": "REJECT", "reason": str(exc)})
    except InferenceError as exc:
        logger.exception("Ошибка обработки изображения")
        raise HTTPException(status_code=500, detail="Internal inference error") from exc

    total_ms = (perf_counter() - request_start) * 1000
    timings = result["timings"]
    logger.info(
        "POST /process: preprocess=%.2fms inference=%.2fms "
        "postprocess=%.2fms total=%.2fms input_bytes=%d",
        timings["preprocess_ms"],
        timings["inference_ms"],
        timings["postprocess_ms"],
        total_ms,
        len(image_bytes),
    )
    result["timings"]["total_ms"] = round(total_ms, 2)
    return JSONResponse(content=result)
