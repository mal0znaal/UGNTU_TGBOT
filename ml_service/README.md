# ML service для удаления фона одежды

Отдельный production-like inference service для Telegram-бота. Runtime использует только ONNX Runtime, OpenCV, NumPy и FastAPI: без `torch`, `transformers`, `ultralytics`, `mlflow`, датасетов и training scripts в Docker image.

## Структура

```text
ml_service/
  app/
    main.py
    config.py
    pipeline.py
    yolo.py
    segformer.py
    image_utils.py
  export/
    export_segformer_to_onnx.py
  models/
    .gitkeep
  requirements.txt
  Dockerfile
  docker-compose.example.yml
```

## Что положить в `models/`

```powershell
cd C:\MY_Data\Programming\Work\UGNTU_TGBOT\ml_service

Copy-Item `
  'C:\MY_Data\Programming\Work\fashion\yolo_26n_dynamic_v01.onnx' `
  '.\models\detector.onnx'
```

После экспорта SegFormer также должен появиться:

```text
models/segmenter.onnx
```

## Экспорт SegFormer в ONNX

Экспортный скрипт запускается вне runtime-контейнера, например в conda окружении `research`. Он может использовать `torch` и `transformers`, но эти зависимости не попадают в Docker image сервиса.

```powershell
conda activate research
cd C:\MY_Data\Programming\Work\UGNTU_TGBOT\ml_service

python .\export\export_segformer_to_onnx.py `
  --checkpoint 'C:\MY_Data\Programming\Work\fashion\runs\segmentation\segformer_b0_720_manual_augmented_bce_dice_boundary\best_manual_dice.pt' `
  --config 'C:\MY_Data\Programming\Work\fashion\configs\segmentation\segformer_b0_720_manual_augmented_bce_dice_boundary.yaml' `
  --output '.\models\segmenter.onnx' `
  --input-size 720
```

ONNX экспортирует только logits. Sigmoid, threshold и resize обратно к размеру crop выполняются в runtime.

## Сборка Docker image

```powershell
cd C:\MY_Data\Programming\Work\UGNTU_TGBOT\ml_service
docker build -t wardrobe-ml-service:latest .
```

## Запуск сервиса

```powershell
docker run --rm -p 8000:8000 `
  -v "${PWD}\models:/app/models:ro" `
  wardrobe-ml-service:latest
```

Проверка healthcheck:

```powershell
curl.exe http://localhost:8000/health
```

## Проверка `/process` через curl

```powershell
curl.exe -X POST `
  -F "file=@C:\path\to\photo.jpg" `
  http://localhost:8000/process `
  --output result.png
```

Успешный ответ: `HTTP 200`, `Content-Type: image/png`, тело ответа - PNG crop одежды с alpha-channel.

Каждый успешный инференс сохраняется в `INFERENCE_OUTPUT_DIR`:

```text
source.png
result.png
```

В основном compose проекта эта папка проброшена на хост как `ml_service/inference_results/`.

Ошибки:

- `400` - изображение не декодируется.
- `422` - YOLO detector ничего не нашел.
- `500` - внутренняя ошибка inference.

## Docker Compose пример

```powershell
cd C:\MY_Data\Programming\Work\UGNTU_TGBOT\ml_service
docker compose -f docker-compose.example.yml up --build
```

Внутри общего `docker-compose` Telegram-боту нужен env:

```yaml
environment:
  ML_SERVICE_URL: http://ml-service:8000/process
```

Сервис слушает `0.0.0.0:8000`, endpoint для бота: `POST http://ml-service:8000/process`, multipart field name: `file`.

## Pipeline

1. Декодирование изображения через OpenCV.
2. YOLO ONNX detector: letterbox `960x960`, normalize `/255`, NCHW `[1, 3, 960, 960]`.
3. YOLO postprocess из исходного sample без `ultralytics`, `conf_threshold=0.25`, `iou_threshold=0.45`.
4. После NMS выбирается bbox с самой большой площадью.
5. Bbox переводится в пиксели исходного изображения, добавляется padding `10%`, координаты clip по границам.
6. Crop переводится в RGB и подается в SegFormer ONNX: resize `720x720`, ImageNet normalization, NCHW `[1, 3, 720, 720]`.
7. Logits resize обратно к размеру crop, sigmoid, threshold `0.5`.
8. Возвращается PNG crop: RGB + alpha mask.
