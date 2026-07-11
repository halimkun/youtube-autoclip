"""Test untuk worker.py: orkestrasi pipeline penuh, mock semua fungsi tahap."""

from __future__ import annotations

from pathlib import Path

import pytest

from autoclip.db import JobStore, get_job_store
from autoclip.models import Job, JobStatus
from autoclip.pipeline.analyze import AnalyzeError
from autoclip.pipeline.download import DownloadError
from autoclip.pipeline.render import RenderError
from autoclip.pipeline.transcribe import TranscribeError
from autoclip.worker import process_job, run_forever, run_once


@pytest.fixture
def store() -> JobStore:
    return get_job_store()


def _create_job(store: JobStore, job_id: str = "job-1") -> Job:
    return store.create_job(job_id, "https://youtu.be/xyz", chat_id=1, user_id=2)


def _install_success_mocks(monkeypatch: pytest.MonkeyPatch, call_order: list[str]) -> None:
    async def _download(job_id: str, url: str) -> Path:
        call_order.append("download")
        return Path("/tmp/raw.mp4")

    async def _extract(job_id: str) -> Path:
        call_order.append("extract_audio")
        return Path("/tmp/audio.m4a")

    async def _transcribe(job_id: str) -> str:
        call_order.append("transcribe")
        return "FAKE_TRANSCRIPT"

    async def _analyze(job_id: str, transcript: str) -> str:
        call_order.append("analyze")
        assert transcript == "FAKE_TRANSCRIPT"
        return "FAKE_PLAN"

    async def _render(job_id: str, clip_plan: str, transcript: str) -> list[Path]:
        call_order.append("render")
        assert clip_plan == "FAKE_PLAN"
        assert transcript == "FAKE_TRANSCRIPT"
        return [Path("/tmp/clip-0.mp4")]

    monkeypatch.setattr("autoclip.worker.download_video", _download)
    monkeypatch.setattr("autoclip.worker.extract_audio", _extract)
    monkeypatch.setattr("autoclip.worker.transcribe_audio", _transcribe)
    monkeypatch.setattr("autoclip.worker.analyze_transcript", _analyze)
    monkeypatch.setattr("autoclip.worker.render_all_clips", _render)


async def test_process_job_runs_full_pipeline_in_order(
    monkeypatch: pytest.MonkeyPatch, store: JobStore
) -> None:
    _create_job(store)
    call_order: list[str] = []
    _install_success_mocks(monkeypatch, call_order)

    job = await process_job("job-1")

    assert job.status is JobStatus.DONE
    assert call_order == ["download", "extract_audio", "transcribe", "analyze", "render"]


async def test_process_job_marks_failed_on_download_error(
    monkeypatch: pytest.MonkeyPatch, store: JobStore
) -> None:
    _create_job(store)

    async def _boom(job_id: str, url: str) -> Path:
        raise DownloadError("yt-dlp gagal total")

    monkeypatch.setattr("autoclip.worker.download_video", _boom)

    job = await process_job("job-1")

    assert job.status is JobStatus.FAILED
    assert job.error_message == "yt-dlp gagal total"


async def test_process_job_marks_failed_on_transcribe_error(
    monkeypatch: pytest.MonkeyPatch, store: JobStore
) -> None:
    _create_job(store)
    call_order: list[str] = []
    _install_success_mocks(monkeypatch, call_order)

    async def _boom(job_id: str) -> str:
        raise TranscribeError("whisper meledak")

    monkeypatch.setattr("autoclip.worker.transcribe_audio", _boom)

    job = await process_job("job-1")

    assert job.status is JobStatus.FAILED
    assert job.error_message == "whisper meledak"
    assert call_order == ["download", "extract_audio"]  # tidak lanjut ke analyze/render


async def test_process_job_marks_failed_on_analyze_error(
    monkeypatch: pytest.MonkeyPatch, store: JobStore
) -> None:
    _create_job(store)
    call_order: list[str] = []
    _install_success_mocks(monkeypatch, call_order)

    async def _boom(job_id: str, transcript: str) -> str:
        raise AnalyzeError("LLM gagal 3 percobaan")

    monkeypatch.setattr("autoclip.worker.analyze_transcript", _boom)

    job = await process_job("job-1")

    assert job.status is JobStatus.FAILED
    assert job.error_message == "LLM gagal 3 percobaan"


async def test_process_job_marks_failed_on_render_error(
    monkeypatch: pytest.MonkeyPatch, store: JobStore
) -> None:
    _create_job(store)
    call_order: list[str] = []
    _install_success_mocks(monkeypatch, call_order)

    async def _boom(job_id: str, clip_plan: str, transcript: str) -> list[Path]:
        raise RenderError("ffmpeg gagal")

    monkeypatch.setattr("autoclip.worker.render_all_clips", _boom)

    job = await process_job("job-1")

    assert job.status is JobStatus.FAILED
    assert job.error_message == "ffmpeg gagal"


async def test_process_job_retries_from_failed_status(
    monkeypatch: pytest.MonkeyPatch, store: JobStore
) -> None:
    _create_job(store)
    store.update_status("job-1", JobStatus.DOWNLOADING)
    store.update_status("job-1", JobStatus.FAILED, error_message="percobaan pertama gagal")

    call_order: list[str] = []
    _install_success_mocks(monkeypatch, call_order)

    job = await process_job("job-1")

    assert job.status is JobStatus.DONE
    assert job.error_message is None


async def test_process_job_resumes_from_mid_flight_status_without_error(
    monkeypatch: pytest.MonkeyPatch, store: JobStore
) -> None:
    # Simulasikan worker mati di tengah TRANSCRIBING (belum sempat menandai FAILED).
    _create_job(store)
    store.update_status("job-1", JobStatus.DOWNLOADING)
    store.update_status("job-1", JobStatus.TRANSCRIBING)

    call_order: list[str] = []
    _install_success_mocks(monkeypatch, call_order)

    job = await process_job("job-1")

    assert job.status is JobStatus.DONE


async def test_process_job_returns_immediately_when_already_done(
    monkeypatch: pytest.MonkeyPatch, store: JobStore
) -> None:
    _create_job(store)
    for target in (
        JobStatus.DOWNLOADING,
        JobStatus.TRANSCRIBING,
        JobStatus.ANALYZING,
        JobStatus.RENDERING,
        JobStatus.DONE,
    ):
        store.update_status("job-1", target)

    def _boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("stage function should not be called for a DONE job")

    monkeypatch.setattr("autoclip.worker.download_video", _boom)

    job = await process_job("job-1")

    assert job.status is JobStatus.DONE


async def test_run_once_returns_none_when_no_queued_job(store: JobStore) -> None:
    assert await run_once() is None


async def test_run_once_processes_next_queued_job(
    monkeypatch: pytest.MonkeyPatch, store: JobStore
) -> None:
    _create_job(store)
    call_order: list[str] = []
    _install_success_mocks(monkeypatch, call_order)

    job = await run_once()

    assert job is not None
    assert job.job_id == "job-1"
    assert job.status is JobStatus.DONE


async def test_run_forever_polls_and_sleeps_when_idle(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"run_once": 0, "sleep_seconds": None}

    class _StopLoop(Exception):
        pass

    async def _fake_run_once() -> None:
        calls["run_once"] += 1
        return None

    async def _fake_sleep(seconds: float) -> None:
        calls["sleep_seconds"] = seconds
        raise _StopLoop

    monkeypatch.setattr("autoclip.worker.run_once", _fake_run_once)
    monkeypatch.setattr("autoclip.worker.asyncio.sleep", _fake_sleep)

    with pytest.raises(_StopLoop):
        await run_forever(poll_interval_seconds=2.5)

    assert calls["run_once"] == 1
    assert calls["sleep_seconds"] == 2.5
