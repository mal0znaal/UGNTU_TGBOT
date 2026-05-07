# UGNTU Telegram bot + ML service

Проект состоит из двух отдельных Docker-контейнеров:

- `ml-service` - FastAPI ML inference service. Он принимает фото, запускает `YOLO ONNX -> crop -> SegFormer ONNX -> PNG без фона`.
- `tg-bot` - Telegram-бот на aiogram. Он принимает фото в Telegram, отправляет его в ML-сервис и возвращает пользователю PNG с удаленным фоном.

Зависимости разделены: ML-зависимости не попадают в контейнер бота, а зависимости бота не попадают в контейнер ML-сервиса.

## Модели

В папке `ml_service/models/` должны лежать:

```text
detector.onnx
segmenter.onnx
```

Файл `ml_service/models/.gitkeep` нужен только для сохранения папки в git. ONNX-модели не коммитятся обычным git.

## Настройка Telegram token

Создайте `.env` в корне проекта из примера:

```powershell
copy .env.example .env
```

Затем впишите настоящий токен:

```text
BOT_TOKEN=123456:your_real_telegram_bot_token
```

Токен берется только из `.env` и не хардкодится в коде.

## Запуск

Из корня проекта:

```powershell
docker compose up --build
```

Остановить:

```powershell
docker compose down
```

Основной запуск теперь через корневой `docker-compose.yml`. Файл `ml_service/docker-compose.example.yml` оставлен как отдельный пример для запуска и отладки только ML-сервиса.

## Проверка ML-сервиса отдельно

Healthcheck:

```powershell
curl.exe http://localhost:8000/health
```

Проверка `/process`:

```powershell
curl.exe -X POST `
  -F "file=@C:\path\to\photo.jpg" `
  http://localhost:8000/process `
  --output result.png
```

Успешный ответ `/process` - `image/png` bytes.

Каждый успешный инференс сохраняется на хосте в:

```text
ml_service/inference_results/
```

Для каждого запроса создается отдельная подпапка с файлами:

```text
source.png
result.png
```

## Важно про адреса внутри Docker Compose

Внутри Docker Compose бот должен обращаться к ML-сервису по адресу:

```text
http://ml-service:8000/process
```

Не используйте `localhost` внутри контейнера бота: там `localhost` будет указывать на сам контейнер `tg-bot`, а не на ML-сервис.
