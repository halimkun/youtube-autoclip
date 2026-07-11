"""Pengiriman hasil job (klip + metadata) kembali ke chat Telegram.

Mengirim tiap file klip sebagai video dengan caption berisi judul,
deskripsi, dan hashtags siap copy-paste (lihat BRIEF.md bagian 5.4).
Juga menangani update progress job dengan mengedit pesan yang sudah ada
(bukan mengirim pesan baru) — pesan progress berupa timeline semua
tahap yang sudah dilalui beserta durasinya (lihat `format_status_message`),
supaya user tahu persis job sedang di tahap mana dan sudah berapa lama.
Serta batas ukuran upload Bot API (50 MB) — kompres ulang atau kirim
sebagai dokumen bila melebihi.

Semua pesan pakai parse_mode HTML alih-alih Markdown (lebih aman: cuma
perlu escape &/</> lewat `html.escape`, tidak serumit aturan escape
MarkdownV2). Teks dinamis dari luar (judul/deskripsi LLM, pesan error)
SELALU di-escape sebelum disisipkan ke HTML.
"""

from __future__ import annotations

import html
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from aiogram import Bot
from aiogram.types import FSInputFile

from autoclip.models import Clip, ClipPlan, JobStatus

logger = logging.getLogger(__name__)

_TELEGRAM_VIDEO_LIMIT_BYTES = 50 * 1024 * 1024
# Batas panjang teks error yang ditampilkan di pesan Telegram (pesan Telegram
# sendiri dibatasi 4096 karakter; sisakan ruang untuk timeline di atasnya).
_MAX_ERROR_CHARS = 1200


@dataclass
class StageEvent:
    """Satu transisi status job yang sudah diamati — dipakai `format_status_message`
    membangun timeline. `at` adalah `Job.updated_at` (waktu lokal) saat transisi
    itu terjadi di database, bukan waktu polling mendeteksinya."""

    status: JobStatus
    at: datetime


_STAGE_LABELS: dict[JobStatus, str] = {
    JobStatus.QUEUED: "Antre",
    JobStatus.DOWNLOADING: "Mengunduh video",
    JobStatus.TRANSCRIBING: "Mentranskripsi audio",
    JobStatus.ANALYZING: "Menganalisis momen viral",
    JobStatus.RENDERING: "Merender klip",
    JobStatus.DONE: "Selesai",
    JobStatus.FAILED: "Gagal",
}


def _format_duration(delta: timedelta) -> str:
    total_seconds = max(0, int(delta.total_seconds()))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    parts = []
    if hours:
        parts.append(f"{hours}j")
    if minutes:
        parts.append(f"{minutes}m")
    if seconds or not parts:
        parts.append(f"{seconds}d")
    return " ".join(parts)


def format_status_message(
    job_id: str, history: list[StageEvent], error_message: str | None = None
) -> str:
    """Bangun teks timeline progress job (HTML) untuk diedit ke pesan Telegram yang sama.

    Tiap tahap yang sudah lewat ditandai selesai (✅) berikut durasinya;
    tahap terakhir yang belum DONE/FAILED ditandai sedang berjalan (🔄).
    Kalau job gagal, pesan errornya SELALU ditampilkan penuh (dipotong
    kalau kepanjangan) supaya bisa dipakai untuk debugging.
    """
    lines = [f"🎬 <b>Job</b> <code>{html.escape(job_id)}</code>", ""]

    for i, event in enumerate(history):
        if event.status is JobStatus.FAILED:
            # Biasanya bukan baris timeline tersendiri — tahap SEBELUM ini
            # yang gagal (ditandai ❌ di bawah oleh iterasi entry itu
            # sendiri), dan teks errornya tampil di blok "Error" terpisah
            # supaya tidak duplikat. KECUALI kalau FAILED adalah entry
            # PERTAMA (gagal sebelum sempat masuk tahap apa pun yang
            # teramati, mis. /cancel) — di situ tidak ada tahap sebelumnya
            # untuk menempelkan ❌, jadi tampilkan langsung di sini.
            if i == 0:
                label = html.escape(_STAGE_LABELS[JobStatus.FAILED])
                lines.append(f"❌ <b>{label}</b> — {event.at.strftime('%H:%M:%S')}")
            continue

        label = html.escape(_STAGE_LABELS.get(event.status, event.status.value))
        time_str = event.at.strftime("%H:%M:%S")
        next_event = history[i + 1] if i + 1 < len(history) else None

        if event.status is JobStatus.DONE:
            total = _format_duration(event.at - history[0].at)
            lines.append(f"🎉 <b>{label}!</b> — {time_str} (total {total})")
        elif next_event is not None and next_event.status is JobStatus.FAILED:
            duration = _format_duration(next_event.at - event.at)
            lines.append(f"❌ {label} — {time_str} (gagal setelah {duration})")
        elif next_event is not None:
            lines.append(f"✅ {label} — {time_str} ({_format_duration(next_event.at - event.at)})")
        else:
            lines.append(f"🔄 {label} — {time_str} (sedang berjalan...)")

    if error_message:
        truncated = error_message
        if len(truncated) > _MAX_ERROR_CHARS:
            truncated = truncated[:_MAX_ERROR_CHARS] + "… (dipotong)"
        lines.append("")
        lines.append("⚠️ <b>Error:</b>")
        lines.append(f"<code>{html.escape(truncated)}</code>")

    return "\n".join(lines)


def format_clip_caption(clip: Clip) -> str:
    """Caption siap copy-paste (HTML): judul, deskripsi, tags, hashtags."""
    title = html.escape(clip.title)
    description = html.escape(clip.description)
    tags = html.escape(", ".join(clip.tags))
    hashtags = html.escape(" ".join(clip.hashtags))
    return f"<b>{title}</b>\n\n{description}\n\nTags: {tags}\n{hashtags}"


async def update_status_message(
    bot: Bot,
    chat_id: int,
    message_id: int,
    job_id: str,
    history: list[StageEvent],
    error_message: str | None = None,
) -> None:
    """Edit pesan konfirmasi yang sama dengan timeline terbaru (bukan pesan baru)."""
    text = format_status_message(job_id, history, error_message)
    try:
        await bot.edit_message_text(
            chat_id=chat_id, message_id=message_id, text=text, parse_mode="HTML"
        )
    except Exception:
        logger.warning("job_id=%s gagal edit pesan status", job_id)


async def send_clip(bot: Bot, chat_id: int, clip_path: Path, clip: Clip) -> None:
    """Kirim satu klip: video kalau <=50MB, dokumen kalau lebih besar."""
    caption = format_clip_caption(clip)
    size_bytes = clip_path.stat().st_size

    if size_bytes > _TELEGRAM_VIDEO_LIMIT_BYTES:
        logger.info(
            "clip=%s melebihi batas 50MB (%d bytes), kirim sebagai dokumen", clip_path, size_bytes
        )
        await bot.send_document(
            chat_id=chat_id,
            document=FSInputFile(clip_path),
            caption=caption,
            parse_mode="HTML",
        )
    else:
        await bot.send_video(
            chat_id=chat_id,
            video=FSInputFile(clip_path),
            caption=caption,
            parse_mode="HTML",
        )


async def send_clips(bot: Bot, chat_id: int, clip_paths: list[Path], clip_plan: ClipPlan) -> None:
    """Kirim semua klip hasil render, urut sesuai `clip_plan.clips`."""
    for path, clip in zip(clip_paths, clip_plan.clips, strict=True):
        await send_clip(bot, chat_id, path, clip)
