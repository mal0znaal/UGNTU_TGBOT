FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY app /app/app
COPY models /app/models

ENV DETECTOR_ONNX_PATH=/app/models/detector.onnx
ENV SEGMENTER_ONNX_PATH=/app/models/segmenter.onnx
ENV YOLO_INPUT_SIZE=960
ENV SEG_INPUT_SIZE=720
ENV YOLO_CONF_THRESHOLD=0.25
ENV YOLO_IOU_THRESHOLD=0.45
ENV CROP_PADDING=0.10
ENV MASK_THRESHOLD=0.5
ENV SAVE_INFERENCE_RESULTS=true
ENV INFERENCE_OUTPUT_DIR=/app/inference_results

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).read()"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
