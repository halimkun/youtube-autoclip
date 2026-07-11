"""Test untuk Settings (config.py): parsing env & helper path artefak."""

from __future__ import annotations

from autoclip.config import get_settings


def test_settings_loads_required_fields() -> None:
    settings = get_settings()
    assert settings.telegram_bot_token == "test-token"
    assert settings.telegram_allowed_users == [111, 222]
    assert settings.llm_base_url == "http://localhost:11434/v1"
    assert settings.llm_api_key == "dummy-key"
    assert settings.llm_model == "test-model"
    assert settings.clip_min_seconds == 30
    assert settings.clip_max_seconds == 180
    assert settings.max_clips_per_video == 8
    assert settings.whisper_model == "small"


def test_path_helpers_are_scoped_under_data_dir() -> None:
    settings = get_settings()
    job_id = "abc123"

    assert settings.db_path == settings.data_dir / "jobs.db"
    assert settings.job_dir(job_id) == settings.data_dir / "media" / job_id
    assert settings.raw_video_path(job_id) == settings.job_dir(job_id) / "raw.mp4"
    assert settings.audio_path(job_id) == settings.job_dir(job_id) / "audio.m4a"
    assert settings.transcript_path(job_id) == settings.job_dir(job_id) / "transcript.json"
    assert settings.plan_path(job_id) == settings.job_dir(job_id) / "plan.json"
    assert settings.clips_dir(job_id) == settings.job_dir(job_id) / "clips"


def test_ensure_job_dir_creates_directories() -> None:
    settings = get_settings()
    job_id = "job-xyz"

    path = settings.ensure_job_dir(job_id)

    assert path.is_dir()
    assert (path / "clips").is_dir()
