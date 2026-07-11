"""Fixture bersama: environment dasar, isolasi Settings/JobStore, & sample video."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from autoclip.config import get_settings
from autoclip.db import get_job_store


@pytest.fixture(autouse=True)
def _base_env(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """Set env minimal yang valid agar Settings() tidak bergantung pada .env lokal."""
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "111,222")
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("LLM_API_KEY", "dummy-key")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("CLIP_MIN_SECONDS", "30")
    monkeypatch.setenv("CLIP_MAX_SECONDS", "180")
    monkeypatch.setenv("MAX_CLIPS_PER_VIDEO", "8")
    monkeypatch.setenv("WHISPER_MODEL", "small")
    monkeypatch.setenv("WHISPER_FALLBACK_ENABLED", "true")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    get_job_store.cache_clear()
    yield
    get_job_store.cache_clear()
    get_settings.cache_clear()


@pytest.fixture
def sample_video(tmp_path: Path) -> Path:
    """Video pendek (3 detik, 640x360) dengan audio, digenerate via ffmpeg lavfi.

    Dipakai lintas test pipeline supaya tidak perlu commit file video ke repo.
    """
    path = tmp_path / "sample.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=3:size=640x360:rate=10",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=3",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-shortest",
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path
