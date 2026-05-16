import asyncio
import json
import os
import sys

import aiohttp
from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart
from aiogram.types import BufferedInputFile, Message


if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


BOT_TOKEN = os.getenv("BOT_TOKEN")
ML_SERVICE_URL = os.getenv("ML_SERVICE_URL", "http://ml-service:8000/cascade")
TELEGRAM_PHOTO_SIZE_LIMIT = 10 * 1024 * 1024

dp = Dispatcher()


@dp.message(CommandStart())
async def cmd_start(message: Message):
    await message.answer("Привет! Отправь мне фото одежды, и я пришлю её без фона.")


@dp.message(F.photo)
async def handle_photo(message: Message, bot: Bot):
    photo_size = message.photo[-1]
    file_info = await bot.get_file(photo_size.file_id)
    downloaded_file = await bot.download_file(file_info.file_path)

    await message.answer(
        "Фото получил! Отправляю нейросети на удаление фона, подожди немного..."
    )

    try:
        async with aiohttp.ClientSession() as http_session:
            form = aiohttp.FormData()
            form.add_field(
                "file",
                downloaded_file.read(),
                filename="image.jpg",
                content_type="image/jpeg",
            )

            async with http_session.post(ML_SERVICE_URL, data=form) as response:
                if response.status == 200:
                    content_type = response.headers.get("Content-Type", "")
                    if content_type.startswith("text/plain"):
                        await message.answer(await response.text())
                        return
                    if content_type.startswith("application/json"):
                        result_json = await response.json()
                        await message.answer(
                            f"<pre>{json.dumps(result_json, ensure_ascii=False, indent=2)}</pre>",
                            parse_mode="HTML",
                        )
                        return

                    result_image_bytes = await response.read()
                    result_file = BufferedInputFile(
                        result_image_bytes,
                        filename="detections.png",
                    )
                    if len(result_image_bytes) < TELEGRAM_PHOTO_SIZE_LIMIT:
                        await message.answer_photo(
                            photo=result_file,
                            caption="Готово! Вот результат детектора.",
                        )
                    else:
                        await message.answer_document(
                            document=result_file,
                            caption="Готово! Вот результат детектора.",
                        )
                else:
                    await message.answer(f"Ошибка от нейросети. Код: {response.status}")
    except Exception as exc:
        await message.answer(f"Не удалось связаться с сервером обработки: {exc}")


async def main():
    if not BOT_TOKEN:
        raise RuntimeError("BOT_TOKEN is not set. Create .env from .env.example.")

    bot = Bot(token=BOT_TOKEN)
    print("Бот запущен...")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
