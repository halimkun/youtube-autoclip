"""Test untuk telegram_sender.py: format pesan/caption & pengiriman (Bot di-mock)."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from autoclip.models import Clip, ClipPlan, JobStatus
from autoclip.telegram_sender import (
    StageEvent,
    format_clip_caption,
    format_status_message,
    send_clip,
    send_clips,
    update_status_message,
)


class _FakeBot:
    def __init__(self) -> None:
        self.edit_calls: list[dict[str, Any]] = []
        self.video_calls: list[dict[str, Any]] = []
        self.document_calls: list[dict[str, Any]] = []
        self.raise_on_edit: Exception | None = None

    async def edit_message_text(self, **kwargs: Any) -> None:
        if self.raise_on_edit is not None:
            raise self.raise_on_edit
        self.edit_calls.append(kwargs)

    async def send_video(self, **kwargs: Any) -> None:
        self.video_calls.append(kwargs)

    async def send_document(self, **kwargs: Any) -> None:
        self.document_calls.append(kwargs)


def _clip(**overrides: Any) -> Clip:
    data: dict[str, Any] = {
        "start": "00:00:00",
        "end": "00:00:40",
        "hook_reason": "momen test",
        "virality_score": 7,
        "title": "Judul Hook",
        "description": "Deskripsi klip.",
        "tags": ["tag1", "tag2"],
        "hashtags": ["#shorts", "#viral"],
    }
    data.update(overrides)
    return Clip(**data)


def _at(seconds_from_epoch: int) -> datetime:
    return datetime(2026, 1, 1, 8, 0, 0) + timedelta(seconds=seconds_from_epoch)


def test_format_status_message_shows_in_progress_stage() -> None:
    history = [StageEvent(JobStatus.RENDERING, _at(0))]

    text = format_status_message("job-1", history)

    assert "job-1" in text
    assert "Merender" in text
    assert "🔄" in text
    assert "sedang berjalan" in text


def test_format_status_message_shows_completed_stages_with_duration() -> None:
    history = [
        StageEvent(JobStatus.QUEUED, _at(0)),
        StageEvent(JobStatus.DOWNLOADING, _at(2)),
        StageEvent(JobStatus.TRANSCRIBING, _at(1922)),  # +32m
    ]

    text = format_status_message("job-1", history)

    assert "✅ Antre" in text
    assert "(2d)" in text
    assert "✅ Mengunduh video" in text
    assert "(32m)" in text
    assert "🔄 Mentranskripsi audio" in text


def test_format_status_message_shows_done_with_total_duration() -> None:
    history = [
        StageEvent(JobStatus.QUEUED, _at(0)),
        StageEvent(JobStatus.DOWNLOADING, _at(5)),
        StageEvent(JobStatus.DONE, _at(65)),
    ]

    text = format_status_message("job-1", history)

    assert "🎉" in text
    assert "total 1m 5d" in text


def test_format_status_message_includes_error_when_failed() -> None:
    history = [
        StageEvent(JobStatus.DOWNLOADING, _at(0)),
        StageEvent(JobStatus.FAILED, _at(10)),
    ]

    text = format_status_message("job-1", history, error_message="yt-dlp gagal")

    # Stage yang gagal (Mengunduh video) ditandai ❌ langsung — bukan baris
    # "Gagal" generik terpisah, supaya jelas TAHAP MANA yang gagal.
    assert "❌ Mengunduh video" in text
    assert "gagal setelah 10d" in text
    assert "yt-dlp gagal" in text


def test_format_status_message_shows_generic_failed_when_no_prior_stage() -> None:
    # Kasus /cancel: FAILED tanpa tahap sebelumnya yang teramati sama sekali.
    history = [StageEvent(JobStatus.FAILED, _at(0))]

    text = format_status_message("job-1", history, error_message="Dibatalkan oleh user")

    assert "❌" in text
    assert "Gagal" in text
    assert "Dibatalkan oleh user" in text


def test_format_status_message_escapes_html_in_error() -> None:
    history = [StageEvent(JobStatus.FAILED, _at(0))]

    text = format_status_message("job-1", history, error_message="<script>alert(1)</script>")

    assert "<script>" not in text
    assert "&lt;script&gt;" in text


def test_format_status_message_truncates_long_error() -> None:
    history = [StageEvent(JobStatus.FAILED, _at(0))]
    long_error = "x" * 5000

    text = format_status_message("job-1", history, error_message=long_error)

    assert len(text) < 2000
    assert "dipotong" in text


def test_format_clip_caption_includes_all_metadata() -> None:
    caption = format_clip_caption(_clip())
    assert "<b>Judul Hook</b>" in caption
    assert "Deskripsi klip." in caption
    assert "tag1, tag2" in caption
    assert "#shorts #viral" in caption


def test_format_clip_caption_escapes_html_special_chars() -> None:
    caption = format_clip_caption(_clip(title="A & B <tag>"))
    assert "<tag>" not in caption
    assert "A &amp; B &lt;tag&gt;" in caption


async def test_update_status_message_edits_existing_message() -> None:
    bot = _FakeBot()
    history = [StageEvent(JobStatus.DONE, _at(0))]

    await update_status_message(bot, chat_id=1, message_id=42, job_id="job-1", history=history)

    assert len(bot.edit_calls) == 1
    assert bot.edit_calls[0]["chat_id"] == 1
    assert bot.edit_calls[0]["message_id"] == 42
    assert bot.edit_calls[0]["parse_mode"] == "HTML"
    assert "Selesai" in bot.edit_calls[0]["text"]


async def test_update_status_message_swallows_edit_failure() -> None:
    bot = _FakeBot()
    bot.raise_on_edit = RuntimeError("message not modified")
    history = [StageEvent(JobStatus.DONE, _at(0))]

    # Tidak boleh melempar - kegagalan edit pesan tidak seharusnya menjatuhkan pipeline.
    await update_status_message(bot, chat_id=1, message_id=42, job_id="job-1", history=history)


async def test_send_clip_sends_as_video_when_under_limit(tmp_path: Path) -> None:
    bot = _FakeBot()
    clip_path = tmp_path / "clip.mp4"
    clip_path.write_bytes(b"x" * 1024)

    await send_clip(bot, chat_id=1, clip_path=clip_path, clip=_clip())

    assert len(bot.video_calls) == 1
    assert len(bot.document_calls) == 0
    assert "Judul Hook" in bot.video_calls[0]["caption"]
    assert bot.video_calls[0]["parse_mode"] == "HTML"


async def test_send_clip_sends_as_document_when_over_limit(tmp_path: Path) -> None:
    bot = _FakeBot()
    clip_path = tmp_path / "clip.mp4"
    clip_path.write_bytes(b"x" * (51 * 1024 * 1024))

    await send_clip(bot, chat_id=1, clip_path=clip_path, clip=_clip())

    assert len(bot.document_calls) == 1
    assert len(bot.video_calls) == 0


async def test_send_clips_sends_each_clip_matched_by_order(tmp_path: Path) -> None:
    bot = _FakeBot()
    path_a = tmp_path / "a.mp4"
    path_b = tmp_path / "b.mp4"
    path_a.write_bytes(b"a")
    path_b.write_bytes(b"b")

    plan = ClipPlan(
        video_summary="ringkasan",
        clips=[
            _clip(title="Klip A", start="00:00:00", end="00:00:40"),
            _clip(title="Klip B", start="00:01:00", end="00:01:40"),
        ],
    )

    await send_clips(bot, chat_id=1, clip_paths=[path_a, path_b], clip_plan=plan)

    assert len(bot.video_calls) == 2
    assert "Klip A" in bot.video_calls[0]["caption"]
    assert "Klip B" in bot.video_calls[1]["caption"]


async def test_send_clips_raises_when_paths_and_clips_mismatch(tmp_path: Path) -> None:
    bot = _FakeBot()
    path_a = tmp_path / "a.mp4"
    path_a.write_bytes(b"a")

    plan = ClipPlan(
        video_summary="ringkasan",
        clips=[
            _clip(start="00:00:00", end="00:00:40"),
            _clip(title="Klip B", start="00:01:00", end="00:01:40"),
        ],
    )

    with pytest.raises(ValueError, match="zip"):
        await send_clips(bot, chat_id=1, clip_paths=[path_a], clip_plan=plan)
