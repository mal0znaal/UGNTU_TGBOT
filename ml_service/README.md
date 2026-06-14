# ML-сервис анализа одежды

FastAPI-сервис запускает ONNX-модели через ONNX Runtime и обрабатывает фотографии
без `torch`, `transformers` и `ultralytics`.

## Модели

Скачайте обученные модели с
[Google Drive](https://drive.google.com/drive/folders/1Xbcpuj3874qKpg9cb09s58jXelW5-Sig?usp=drive_link)
и поместите их в каталог `ml_service/models`, сохранив структуру каталогов:

```text
models/
  detector.onnx
  segmenter.onnx
  classifier/
    manifest.json
    ...файлы моделей из manifest.json
```

## Запуск

Из корня репозитория:

```powershell
docker compose up --build
```

Проверка сервиса:

```powershell
curl.exe http://localhost:8000/health
```

## API

Сервис предоставляет один рабочий маршрут:

```text
POST /process
Content-Type: multipart/form-data
Поле с изображением: file
```

Пример запроса:

```powershell
curl.exe -X POST `
  -F "file=@C:\path\to\photo.jpg" `
  http://localhost:8000/process
```

Успешный ответ содержит:

```json
{
  "decision": "ACCEPT",
  "image_base64": "...",
  "classification": {
    "category": "...",
    "subcategory": "...",
    "color": "...",
    "seasons": [],
    "styles": []
  },
  "timings": {
    "preprocess_ms": 0.0,
    "inference_ms": 0.0,
    "postprocess_ms": 0.0,
    "total_ms": 0.0
  }
}
```

Те же тайминги выводятся в лог после каждого успешного `POST /process`.

## Пайплайн

1. Изображение декодируется через OpenCV.
2. YOLO находит предметы одежды.
3. Выбирается самый крупный объект с отступом 10% вокруг рамки.
4. SegFormer строит маску и удаляет фон.
5. Классификаторы определяют категорию и подкатегорию.
6. По изображению вычисляются цвет, сезоны и стили.

Если включён `SAVE_INFERENCE_RESULTS`, сервис сохраняет отладочный коллаж в
`INFERENCE_OUTPUT_DIR`.
