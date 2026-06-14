import asyncio
import base64
import os
import sys

import aiohttp
from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart
from aiogram.types import BufferedInputFile, Message


if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


BOT_TOKEN = os.getenv("BOT_TOKEN")
ML_SERVICE_URL = os.getenv("ML_SERVICE_URL", "http://ml-service:8000/process")
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
                    result_json = await response.json()
                    
                    if result_json.get("decision") == "REJECT":
                        reason = result_json.get("reason", "Неизвестная причина")
                        await message.answer(f"❌ Одежда не найдена или отклонена.\nПричина: {reason}")
                        return
                        
                    image_b64 = result_json.get("image_base64")
                    if not image_b64:
                        await message.answer("Ошибка: сервер не вернул картинку.")
                        return
                        
                    image_bytes = base64.b64decode(image_b64)
                    result_file = BufferedInputFile(image_bytes, filename="result.png")
                    
                    cls_info = result_json.get("classification", {})
                    category = cls_info.get("category", "?")
                    subcategory = cls_info.get("subcategory", "?")
                    color = cls_info.get("color", "?")
                    seasons = ", ".join(cls_info.get("seasons", []))
                    styles = ", ".join(cls_info.get("styles", []))
                    
                    caption = (
                        f"✨ <b>Результат обработки:</b>\n\n"
                        f"🧥 <b>Категория:</b> {category}\n"
                        f"👕 <b>Подкатегория:</b> {subcategory}\n"
                        f"🎨 <b>Цвет:</b> {color}\n"
                        f"☀️ <b>Сезон:</b> {seasons}\n"
                        f"🎭 <b>Стиль:</b> {styles}"
                    )
                    
                    await message.answer_photo(
                        photo=result_file,
                        caption=caption,
                        parse_mode="HTML"
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
