"""Handler bot Telegram (aiogram v3) dan entrypoint `python -m autoclip.bot`.

Menerima link YouTube langsung (tanpa command) dari user ber-whitelist
(`TELEGRAM_ALLOWED_USERS`), membuat job baru di database, dan membalas
konfirmasi + job_id. Menyediakan perintah `/status <job_id>` dan
`/cancel <job_id>`. Progress job dipantau lewat task terpisah yang
polling `db.py` dan mengedit pesan konfirmasi (bukan mengirim pesan
baru) setiap status berubah; saat DONE, klip dikirim via
`telegram_sender.py`. Menjalankan bot bersamaan dengan worker (lihat
worker.py) dalam satu proses async.
"""

from __future__ import annotations

import asyncio
import logging
import re
import uuid

from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command
from aiogram.types import Message

from autoclip.config import get_settings
from autoclip.db import JobNotFoundError, get_job_store
from autoclip.models import ClipPlan, JobStatus
from autoclip.pipeline.render import clip_output_path
from autoclip.telegram_sender import format_status_message, send_clips, update_status_message
from autoclip.worker import run_forever

logger = logging.getLogger(__name__)

_YOUTUBE_URL_RE = re.compile(
    r"https?://(?:www\.|m\.)?(?:youtube\.com/(?:watch\?v=|shorts/)|youtu\.be/)[\w-]+",
    re.IGNORECASE,
)

_WATCH_POLL_INTERVAL_SECONDS = 3.0


def _is_user_allowed(user_id: int) -> bool:
    return user_id in get_settings().telegram_allowed_users


def _extract_youtube_url(text: str) -> str | None:
    match = _YOUTUBE_URL_RE.search(text)
    return match.group(0) if match else None


def _parse_command_arg(text: str) -> str | None:
    parts = text.strip().split(maxsplit=1)
    if len(parts) < 2 or not parts[1].strip():
        return None
    return parts[1].strip()


def _generate_job_id() -> str:
    return uuid.uuid4().hex[:10]


async def handle_youtube_link(message: Message) -> None:
    """Terima link YouTube langsung (bukan command), buat job baru."""
    user = message.from_user
    if user is None or not _is_user_allowed(user.id):
        await message.answer("Kamu tidak diizinkan menggunakan bot ini.")
        return

    url = _extract_youtube_url(message.text or "")
    if url is None:
        return

    job_id = _generate_job_id()
    store = get_job_store()
    store.create_job(job_id, url, chat_id=message.chat.id, user_id=user.id)
    logger.info("job_id=%s dibuat oleh user_id=%s untuk url=%s", job_id, user.id, url)

    sent = await message.answer(format_status_message(job_id, JobStatus.QUEUED))
    asyncio.create_task(_watch_job(message.bot, message.chat.id, sent.message_id, job_id))


async def cmd_status(message: Message) -> None:
    """`/status <job_id>` — laporkan status job saat ini."""
    user = message.from_user
    if user is None or not _is_user_allowed(user.id):
        await message.answer("Kamu tidak diizinkan menggunakan bot ini.")
        return

    job_id = _parse_command_arg(message.text or "")
    if job_id is None:
        await message.answer("Pakai format: /status <job_id>")
        return

    store = get_job_store()
    try:
        job = store.get_job(job_id)
    except JobNotFoundError:
        await message.answer(f"Job `{job_id}` tidak ditemukan.")
        return

    await message.answer(format_status_message(job_id, job.status, job.error_message))


async def cmd_cancel(message: Message) -> None:
    """`/cancel <job_id>` — batalkan job yang masih QUEUED (belum diproses worker)."""
    user = message.from_user
    if user is None or not _is_user_allowed(user.id):
        await message.answer("Kamu tidak diizinkan menggunakan bot ini.")
        return

    job_id = _parse_command_arg(message.text or "")
    if job_id is None:
        await message.answer("Pakai format: /cancel <job_id>")
        return

    store = get_job_store()
    try:
        job = store.get_job(job_id)
    except JobNotFoundError:
        await message.answer(f"Job `{job_id}` tidak ditemukan.")
        return

    if job.status is not JobStatus.QUEUED:
        await message.answer(
            f"Job `{job_id}` sudah {job.status.value}, tidak bisa dibatalkan lagi."
        )
        return

    store.update_status(job_id, JobStatus.FAILED, error_message="Dibatalkan oleh user")
    await message.answer(f"Job `{job_id}` dibatalkan.")


async def _watch_job(bot: Bot, chat_id: int, message_id: int, job_id: str) -> None:
    """Polling status job, edit pesan konfirmasi tiap berubah, kirim klip saat DONE."""
    store = get_job_store()
    last_status: JobStatus | None = None

    while True:
        job = store.get_job(job_id)
        if job.status != last_status:
            await update_status_message(
                bot, chat_id, message_id, job_id, job.status, job.error_message
            )
            last_status = job.status

        if job.status is JobStatus.DONE:
            settings = get_settings()
            clip_plan = ClipPlan.model_validate_json(settings.plan_path(job_id).read_text())
            clip_paths = [
                clip_output_path(job_id, index, clip) for index, clip in enumerate(clip_plan.clips)
            ]
            await send_clips(bot, chat_id, clip_paths, clip_plan)
            return

        if job.status is JobStatus.FAILED:
            return

        await asyncio.sleep(_WATCH_POLL_INTERVAL_SECONDS)


def create_dispatcher() -> Dispatcher:
    dp = Dispatcher()
    dp.message.register(cmd_status, Command("status"))
    dp.message.register(cmd_cancel, Command("cancel"))
    dp.message.register(handle_youtube_link, F.text.regexp(_YOUTUBE_URL_RE))
    return dp


async def main() -> None:
    logging.basicConfig(level=logging.INFO)
    settings = get_settings()
    bot = Bot(token=settings.telegram_bot_token)
    dp = create_dispatcher()

    # aiogram menangani SIGINT sendiri dan berhenti polling dengan rapi, tapi
    # run_forever() adalah loop `while True` yang tidak tahu soal shutdown itu
    # sama sekali. Kalau dua-duanya di-gather() begitu saja, gather() akan
    # menunggu KEDUANYA selesai — padahal run_forever() tidak pernah selesai
    # sendiri, jadi proses menggantung sampai Ctrl+C ditekan lagi. Di sini,
    # begitu salah satu selesai (biasanya polling, karena SIGINT), yang lain
    # langsung di-cancel supaya shutdown-nya cepat.
    polling_task = asyncio.create_task(dp.start_polling(bot))
    worker_task = asyncio.create_task(run_forever())
    _done, pending = await asyncio.wait(
        [polling_task, worker_task], return_when=asyncio.FIRST_COMPLETED
    )
    for task in pending:
        task.cancel()
    await asyncio.gather(*pending, return_exceptions=True)


if __name__ == "__main__":
    asyncio.run(main())
