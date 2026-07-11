"""Test untuk bot.py: handler aiogram (Message & Bot di-mock), whitelist, watcher."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

import autoclip.bot as bot_module
from autoclip.bot import (
    _extract_youtube_url,
    _generate_job_id,
    _is_user_allowed,
    _parse_command_arg,
    _watch_job,
    cmd_cancel,
    cmd_status,
    handle_youtube_link,
)
from autoclip.config import get_settings
from autoclip.db import JobStore, get_job_store
from autoclip.models import Clip, ClipPlan, JobStatus


class _FakeUser:
    def __init__(self, user_id: int) -> None:
        self.id = user_id


class _FakeChat:
    def __init__(self, chat_id: int) -> None:
        self.id = chat_id


class _FakeSentMessage:
    def __init__(self, message_id: int = 999) -> None:
        self.message_id = message_id


class _FakeMessage:
    def __init__(self, text: str, user_id: int | None, chat_id: int = 555) -> None:
        self.text = text
        self.from_user = None if user_id is None else _FakeUser(user_id)
        self.chat = _FakeChat(chat_id)
        self.bot = None
        self.answers: list[str] = []
        self.answer_kwargs: list[dict[str, Any]] = []

    async def answer(self, text: str, **kwargs: Any) -> _FakeSentMessage:
        self.answers.append(text)
        self.answer_kwargs.append(kwargs)
        return _FakeSentMessage()


class _FakeBot:
    def __init__(self) -> None:
        self.edit_calls: list[dict[str, Any]] = []
        self.video_calls: list[dict[str, Any]] = []

    async def edit_message_text(self, **kwargs: Any) -> None:
        self.edit_calls.append(kwargs)

    async def send_video(self, **kwargs: Any) -> None:
        self.video_calls.append(kwargs)

    async def send_document(self, **kwargs: Any) -> None:
        self.video_calls.append(kwargs)


@pytest.fixture
def store() -> JobStore:
    return get_job_store()


def test_is_user_allowed_matches_env_whitelist() -> None:
    # conftest._base_env set TELEGRAM_ALLOWED_USERS="111,222"
    assert _is_user_allowed(111) is True
    assert _is_user_allowed(222) is True
    assert _is_user_allowed(999) is False


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("cek video ini https://youtu.be/abc123 keren", "https://youtu.be/abc123"),
        ("https://www.youtube.com/watch?v=xyz", "https://www.youtube.com/watch?v=xyz"),
        ("https://youtube.com/shorts/abc", "https://youtube.com/shorts/abc"),
        ("halo, apa kabar?", None),
    ],
)
def test_extract_youtube_url(text: str, expected: str | None) -> None:
    assert _extract_youtube_url(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("/status abc123", "abc123"),
        ("/status   abc123  ", "abc123"),
        ("/status", None),
        ("/status   ", None),
    ],
)
def test_parse_command_arg(text: str, expected: str | None) -> None:
    assert _parse_command_arg(text) == expected


def test_generate_job_id_is_short_hex() -> None:
    job_id = _generate_job_id()
    assert len(job_id) == 10
    int(job_id, 16)  # tidak raise -> valid hex


async def test_handle_youtube_link_rejects_disallowed_user(store: JobStore) -> None:
    message = _FakeMessage("https://youtu.be/abc123", user_id=999)

    await handle_youtube_link(message)

    assert "tidak diizinkan" in message.answers[0]
    assert store.next_queued_job() is None


async def test_handle_youtube_link_creates_job_and_replies(
    monkeypatch: pytest.MonkeyPatch, store: JobStore
) -> None:
    async def _noop_watch(bot: Any, chat_id: int, message_id: int, job_id: str) -> None:
        return None

    monkeypatch.setattr("autoclip.bot._watch_job", _noop_watch)

    message = _FakeMessage("tonton https://youtu.be/abc123 dong", user_id=111)

    await handle_youtube_link(message)
    await asyncio.sleep(0)  # flush task yang dibuat via asyncio.create_task

    job = store.next_queued_job()
    assert job is not None
    assert job.url == "https://youtu.be/abc123"
    assert job.chat_id == 555
    assert job.user_id == 111
    assert job.job_id in message.answers[0]
    assert "diterima" in message.answers[0]
    assert message.answer_kwargs[0]["parse_mode"] == "HTML"


async def test_handle_youtube_link_keeps_strong_reference_to_watch_task(
    monkeypatch: pytest.MonkeyPatch, store: JobStore
) -> None:
    # asyncio.create_task() cuma menyimpan weak reference — tanpa disimpan di
    # _background_tasks, task ini bisa di-GC di tengah jalan (bug asli yang
    # bikin pesan Telegram berhenti ter-update walau job jalan terus).
    started = asyncio.Event()
    release = asyncio.Event()

    async def _blocking_watch(bot: Any, chat_id: int, message_id: int, job_id: str) -> None:
        started.set()
        await release.wait()

    monkeypatch.setattr("autoclip.bot._watch_job", _blocking_watch)

    message = _FakeMessage("tonton https://youtu.be/abc123 dong", user_id=111)

    await handle_youtube_link(message)
    await started.wait()

    assert len(bot_module._background_tasks) == 1
    task = next(iter(bot_module._background_tasks))

    release.set()
    await task  # tunggu task selesai betulan, baru done_callback membersihkannya

    assert len(bot_module._background_tasks) == 0


async def test_handle_youtube_link_ignores_message_without_url(store: JobStore) -> None:
    message = _FakeMessage("halo, apa kabar?", user_id=111)

    await handle_youtube_link(message)

    assert message.answers == []
    assert store.next_queued_job() is None


async def test_cmd_status_reports_not_found(store: JobStore) -> None:
    message = _FakeMessage("/status doesnotexist", user_id=111)

    await cmd_status(message)

    assert "tidak ditemukan" in message.answers[0]


async def test_cmd_status_reports_current_status(store: JobStore) -> None:
    store.create_job("job-1", "https://youtu.be/abc", chat_id=1, user_id=111)
    message = _FakeMessage("/status job-1", user_id=111)

    await cmd_status(message)

    assert "job-1" in message.answers[0]
    assert "Antre" in message.answers[0]


async def test_cmd_status_missing_arg_prompts_usage(store: JobStore) -> None:
    message = _FakeMessage("/status", user_id=111)

    await cmd_status(message)

    assert "format" in message.answers[0].lower()


async def test_cmd_cancel_marks_queued_job_as_failed(store: JobStore) -> None:
    store.create_job("job-1", "https://youtu.be/abc", chat_id=1, user_id=111)
    message = _FakeMessage("/cancel job-1", user_id=111)

    await cmd_cancel(message)

    job = store.get_job("job-1")
    assert job.status is JobStatus.FAILED
    assert job.error_message == "Dibatalkan oleh user"
    assert "dibatalkan" in message.answers[0]


async def test_cmd_cancel_rejects_when_already_past_queued(store: JobStore) -> None:
    store.create_job("job-1", "https://youtu.be/abc", chat_id=1, user_id=111)
    store.update_status("job-1", JobStatus.DOWNLOADING)
    message = _FakeMessage("/cancel job-1", user_id=111)

    await cmd_cancel(message)

    job = store.get_job("job-1")
    assert job.status is JobStatus.DOWNLOADING  # tidak berubah
    assert "tidak bisa dibatalkan" in message.answers[0]


async def test_cmd_cancel_reports_not_found(store: JobStore) -> None:
    message = _FakeMessage("/cancel doesnotexist", user_id=111)

    await cmd_cancel(message)

    assert "tidak ditemukan" in message.answers[0]


def _clip(**overrides: Any) -> Clip:
    data: dict[str, Any] = {
        "start": "00:00:00",
        "end": "00:00:40",
        "hook_reason": "momen test",
        "virality_score": 7,
        "title": "Judul Hook",
        "description": "deskripsi",
        "tags": ["tag1"],
        "hashtags": ["#shorts"],
    }
    data.update(overrides)
    return Clip(**data)


async def test_watch_job_sends_clips_when_done(store: JobStore) -> None:
    store.create_job("job-1", "https://youtu.be/abc", chat_id=1, user_id=111)
    for target in (
        JobStatus.DOWNLOADING,
        JobStatus.TRANSCRIBING,
        JobStatus.ANALYZING,
        JobStatus.RENDERING,
        JobStatus.DONE,
    ):
        store.update_status("job-1", target)

    settings = get_settings()
    settings.ensure_job_dir("job-1")
    plan = ClipPlan(video_summary="ringkasan", clips=[_clip()])
    settings.plan_path("job-1").write_text(plan.model_dump_json())
    clip_path = settings.clips_dir("job-1") / "judul-hook-0.mp4"
    clip_path.write_bytes(b"video bytes")

    bot = _FakeBot()
    await _watch_job(bot, chat_id=1, message_id=42, job_id="job-1")

    assert len(bot.edit_calls) == 1
    assert "Selesai" in bot.edit_calls[0]["text"]
    assert len(bot.video_calls) == 1


async def test_watch_job_stops_and_edits_message_when_failed(store: JobStore) -> None:
    store.create_job("job-1", "https://youtu.be/abc", chat_id=1, user_id=111)
    store.update_status("job-1", JobStatus.FAILED, error_message="yt-dlp gagal")

    bot = _FakeBot()
    await _watch_job(bot, chat_id=1, message_id=42, job_id="job-1")

    assert len(bot.edit_calls) == 1
    assert "Gagal" in bot.edit_calls[0]["text"]
    assert "yt-dlp gagal" in bot.edit_calls[0]["text"]
    assert bot.video_calls == []


async def test_watch_job_polls_until_status_changes(
    monkeypatch: pytest.MonkeyPatch, store: JobStore
) -> None:
    store.create_job("job-1", "https://youtu.be/abc", chat_id=1, user_id=111)

    async def _advance_then_sleep(seconds: float) -> None:
        # Simulasikan waktu berlalu: job baru selesai setelah satu kali poll.
        store.update_status("job-1", JobStatus.FAILED, error_message="gagal setelah nunggu")

    monkeypatch.setattr("autoclip.bot.asyncio.sleep", _advance_then_sleep)

    bot = _FakeBot()
    await _watch_job(bot, chat_id=1, message_id=42, job_id="job-1")

    statuses_seen = [call["text"] for call in bot.edit_calls]
    assert any("🔄 Antre" in text for text in statuses_seen)
    # Stage yang gagal (Antre, satu-satunya yang teramati) ditandai ❌
    # langsung, bukan baris "Gagal" generik terpisah.
    assert any("❌ Antre" in text and "gagal setelah nunggu" in text for text in statuses_seen)


async def test_watch_job_does_not_reedit_on_unchanged_status(
    monkeypatch: pytest.MonkeyPatch, store: JobStore
) -> None:
    store.create_job("job-1", "https://youtu.be/abc", chat_id=1, user_id=111)

    call_count = {"n": 0}

    async def _fake_sleep(seconds: float) -> None:
        call_count["n"] += 1
        if call_count["n"] >= 2:
            store.update_status("job-1", JobStatus.FAILED, error_message="selesai nunggu")

    monkeypatch.setattr("autoclip.bot.asyncio.sleep", _fake_sleep)

    bot = _FakeBot()
    await _watch_job(bot, chat_id=1, message_id=42, job_id="job-1")

    # Timeline persisten di tiap edit (jadi "Antre" tetap muncul di pesan
    # akhir juga) — yang penting jumlah edit-nya cuma sebanyak transisi
    # NYATA (QUEUED -> FAILED = 2x), bukan bertambah tiap iterasi poll yang
    # statusnya belum berubah.
    assert len(bot.edit_calls) == 2
