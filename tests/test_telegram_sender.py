"""Test untuk telegram_sender.py: format pesan/caption & pengiriman (Bot di-mock)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from autoclip.models import Clip, ClipPlan, JobStatus
from autoclip.telegram_sender import (
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


def test_format_status_message_without_error() -> None:
    text = format_status_message("job-1", JobStatus.RENDERING)
    assert "job-1" in text
    assert "Merender" in text


def test_format_status_message_includes_error_when_failed() -> None:
    text = format_status_message("job-1", JobStatus.FAILED, error_message="yt-dlp gagal")
    assert "Gagal" in text
    assert "yt-dlp gagal" in text


def test_format_clip_caption_includes_all_metadata() -> None:
    caption = format_clip_caption(_clip())
    assert "Judul Hook" in caption
    assert "Deskripsi klip." in caption
    assert "tag1, tag2" in caption
    assert "#shorts #viral" in caption


async def test_update_status_message_edits_existing_message() -> None:
    bot = _FakeBot()

    await update_status_message(
        bot, chat_id=1, message_id=42, job_id="job-1", status=JobStatus.DONE
    )

    assert len(bot.edit_calls) == 1
    assert bot.edit_calls[0]["chat_id"] == 1
    assert bot.edit_calls[0]["message_id"] == 42
    assert "Selesai" in bot.edit_calls[0]["text"]


async def test_update_status_message_swallows_edit_failure() -> None:
    bot = _FakeBot()
    bot.raise_on_edit = RuntimeError("message not modified")

    # Tidak boleh melempar - kegagalan edit pesan tidak seharusnya menjatuhkan pipeline.
    await update_status_message(
        bot, chat_id=1, message_id=42, job_id="job-1", status=JobStatus.DONE
    )


async def test_send_clip_sends_as_video_when_under_limit(tmp_path: Path) -> None:
    bot = _FakeBot()
    clip_path = tmp_path / "clip.mp4"
    clip_path.write_bytes(b"x" * 1024)

    await send_clip(bot, chat_id=1, clip_path=clip_path, clip=_clip())

    assert len(bot.video_calls) == 1
    assert len(bot.document_calls) == 0
    assert "Judul Hook" in bot.video_calls[0]["caption"]


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
