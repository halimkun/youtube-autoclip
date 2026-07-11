"""Konfigurasi aplikasi via pydantic-settings.

Memuat semua nilai environment yang didefinisikan di BRIEF.md bagian
"Environment (.env)": kredensial Telegram, endpoint/kredensial LLM
(provider-agnostic), aturan durasi & jumlah klip, model whisper, dan
direktori data. Juga menyediakan helper path artefak (mis. direktori
media per job_id) yang dipakai oleh db.py dan modul pipeline lain agar
path tidak pernah di-hardcode di tempat lain.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    """Nilai konfigurasi runtime, seluruhnya dari environment/.env."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    telegram_bot_token: str
    telegram_allowed_users: Annotated[list[int], NoDecode]

    llm_base_url: str
    llm_api_key: str
    llm_model: str

    clip_min_seconds: int = 30
    clip_max_seconds: int = 180
    max_clips_per_video: int = 8

    whisper_model: str = "small"
    # Kalau caption YouTube (data/media/<job_id>/raw.id.vtt) tidak ada:
    # True (default) -> transkripsi pakai whisper seperti biasa.
    # False -> job langsung FAILED di tahap TRANSCRIBING, tidak menunggu
    # whisper (yang lambat untuk video panjang) dan tidak lanjut ke ANALYZING.
    whisper_fallback_enabled: bool = True

    data_dir: Path = Path("./data")

    @field_validator("telegram_allowed_users", mode="before")
    @classmethod
    def _split_allowed_users(cls, value: object) -> object:
        """Terima "111,222" (format .env) maupun list int langsung."""
        if isinstance(value, str):
            return [int(part.strip()) for part in value.split(",") if part.strip()]
        return value

    @property
    def db_path(self) -> Path:
        return self.data_dir / "jobs.db"

    @property
    def media_root(self) -> Path:
        return self.data_dir / "media"

    def job_dir(self, job_id: str) -> Path:
        return self.media_root / job_id

    def raw_video_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "raw.mp4"

    def audio_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "audio.m4a"

    def transcript_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "transcript.json"

    def subtitle_path(self, job_id: str) -> Path:
        """Path caption YouTube (kalau ada) — lihat `download._download_subtitles_best_effort`."""
        return self.job_dir(job_id) / "raw.id.vtt"

    def plan_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "plan.json"

    def clips_dir(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "clips"

    def ensure_job_dir(self, job_id: str) -> Path:
        """Buat direktori media job (termasuk subfolder clips/) bila belum ada."""
        path = self.job_dir(job_id)
        self.clips_dir(job_id).mkdir(parents=True, exist_ok=True)
        return path


@lru_cache
def get_settings() -> Settings:
    """Singleton Settings — panggil ini, jangan instansiasi Settings() langsung."""
    return Settings()
