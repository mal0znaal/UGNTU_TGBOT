FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY app /app/app
COPY models /app/models

ENV GARMENT_MODEL_PATH=/app/models/detector.onnx
ENV BAD_CLASSES_DETECTOR_PATH=/app/models/bad_classes_detector.onnx
ENV YOLO_INPUT_SIZE=960
ENV YOLO_IOU_THRESHOLD=0.45
ENV GARMENT_CONF=0.25
ENV BAD_CLASS_CONF=0.35
ENV PERSON_CONF=0.75
ENV SAVE_INFERENCE_RESULTS=true
ENV INFERENCE_OUTPUT_DIR=/app/inference_results

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).read()"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
