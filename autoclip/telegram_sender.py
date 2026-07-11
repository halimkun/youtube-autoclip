"""Pengiriman hasil job (klip + metadata) kembali ke chat Telegram.

Mengirim tiap file klip sebagai video dengan caption berisi judul,
deskripsi, dan hashtags siap copy-paste (lihat BRIEF.md bagian 5.4).
Juga menangani update progress status job dengan mengedit pesan yang
sudah ada (bukan mengirim pesan baru), serta batas ukuran upload Bot
API (50 MB) — kompres ulang atau kirim sebagai dokumen bila melebihi.
"""

from __future__ import annotations

import logging
from pathlib import Path

from aiogram import Bot
from aiogram.types import FSInputFile

from autoclip.models import Clip, ClipPlan, JobStatus

logger = logging.getLogger(__name__)

_TELEGRAM_VIDEO_LIMIT_BYTES = 50 * 1024 * 1024

_STATUS_LABELS: dict[JobStatus, str] = {
    JobStatus.QUEUED: "Menunggu diproses...",
    JobStatus.DOWNLOADING: "Mengunduh video...",
    JobStatus.TRANSCRIBING: "Mentranskripsi audio...",
    JobStatus.ANALYZING: "Menganalisis momen viral...",
    JobStatus.RENDERING: "Merender klip...",
    JobStatus.DONE: "Selesai!",
    JobStatus.FAILED: "Gagal.",
}


def format_status_message(job_id: str, status: JobStatus, error_message: str | None = None) -> str:
    """Teks progress job untuk diedit ke pesan Telegram yang sama."""
    lines = [f"Job `{job_id}`: {_STATUS_LABELS[status]}"]
    if status is JobStatus.FAILED and error_message:
        lines.append(f"Error: {error_message}")
    return "\n".join(lines)


def format_clip_caption(clip: Clip) -> str:
    """Caption siap copy-paste: judul, deskripsi, tags, hashtags."""
    tags = ", ".join(clip.tags)
    hashtags = " ".join(clip.hashtags)
    return f"*{clip.title}*\n\n{clip.description}\n\nTags: {tags}\n{hashtags}"


async def update_status_message(
    bot: Bot,
    chat_id: int,
    message_id: int,
    job_id: str,
    status: JobStatus,
    error_message: str | None = None,
) -> None:
    """Edit pesan konfirmasi yang sama dengan status terbaru (bukan pesan baru)."""
    text = format_status_message(job_id, status, error_message)
    try:
        await bot.edit_message_text(chat_id=chat_id, message_id=message_id, text=text)
    except Exception:
        logger.warning("job_id=%s gagal edit pesan status ke '%s'", job_id, status.value)


async def send_clip(bot: Bot, chat_id: int, clip_path: Path, clip: Clip) -> None:
    """Kirim satu klip: video kalau <=50MB, dokumen kalau lebih besar."""
    caption = format_clip_caption(clip)
    size_bytes = clip_path.stat().st_size

    if size_bytes > _TELEGRAM_VIDEO_LIMIT_BYTES:
        logger.info(
            "clip=%s melebihi batas 50MB (%d bytes), kirim sebagai dokumen", clip_path, size_bytes
        )
        await bot.send_document(chat_id=chat_id, document=FSInputFile(clip_path), caption=caption)
    else:
        await bot.send_video(chat_id=chat_id, video=FSInputFile(clip_path), caption=caption)


async def send_clips(bot: Bot, chat_id: int, clip_paths: list[Path], clip_plan: ClipPlan) -> None:
    """Kirim semua klip hasil render, urut sesuai `clip_plan.clips`."""
    for path, clip in zip(clip_paths, clip_plan.clips, strict=True):
        await send_clip(bot, chat_id, path, clip)
