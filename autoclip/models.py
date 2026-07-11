"""Model data Pydantic yang dipakai lintas tahap pipeline.

Mendefinisikan kontrak data BRIEF.md bagian 4 (skema JSON output LLM)
dan status job: `Clip`, `ClipPlan`, `Job`, serta enum status
(QUEUED, DOWNLOADING, TRANSCRIBING, ANALYZING, RENDERING, DONE, FAILED)
dan `OutputFormat` (VERTICAL/ORIGINAL, dipilih user lewat tombol Telegram
sebelum job dibuat).
Termasuk validator Pydantic untuk aturan bisnis klip (format timestamp
HH:MM:SS, end > start, durasi min/maks, virality_score 1-10, jumlah
klip maksimal, dan larangan overlap > 10 detik antar klip).
"""

from __future__ import annotations

import re
from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field, field_validator, model_validator

from autoclip.config import get_settings

_TIMESTAMP_RE = re.compile(r"^(\d{1,2}):(\d{2}):(\d{2})$")
_MMSS_RE = re.compile(r"^(\d{1,2}):(\d{2})$")

_MAX_OVERLAP_SECONDS = 10


class TimestampFormatError(ValueError):
    """Dilempar bila string timestamp tidak berformat HH:MM:SS/MM:SS yang valid."""


def normalize_timestamp(value: str) -> str:
    """Normalisasi timestamp jadi HH:MM:SS baku (dipakai untuk field `Clip.start/end`).

    LLM kadang mengembalikan format "MM:SS" (mis. "08:17") untuk video di bawah
    satu jam alih-alih "HH:MM:SS" penuh — dan ketika diminta memperbaiki lewat
    retry, sering justru salah menyisipkan ":00" jadi "08:17:00" (yang berarti
    8 JAM 17 MENIT, bukan 8 menit 17 detik) alih-alih "00:08:17". Daripada
    bergantung pada LLM memperbaikinya dengan benar, "MM:SS" diterima langsung
    di sini dan dinormalisasi (jam=0) — hasilnya tetap HH:MM:SS baku sesuai
    kontrak BRIEF.md bagian 4.
    """
    match = _TIMESTAMP_RE.match(value)
    if match is not None:
        hours, minutes, seconds = (int(part) for part in match.groups())
    else:
        match = _MMSS_RE.match(value)
        if match is None:
            raise TimestampFormatError(
                f"Timestamp harus berformat HH:MM:SS atau MM:SS, dapat: {value!r}"
            )
        hours = 0
        minutes, seconds = (int(part) for part in match.groups())

    if minutes >= 60 or seconds >= 60:
        raise TimestampFormatError(f"Menit/detik tidak valid pada timestamp: {value!r}")
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def parse_timestamp(value: str) -> int:
    """Ubah timestamp ("HH:MM:SS" atau "MM:SS") menjadi total detik."""
    hours, minutes, seconds = (int(part) for part in normalize_timestamp(value).split(":"))
    return hours * 3600 + minutes * 60 + seconds


class JobStatus(StrEnum):
    QUEUED = "QUEUED"
    DOWNLOADING = "DOWNLOADING"
    TRANSCRIBING = "TRANSCRIBING"
    ANALYZING = "ANALYZING"
    RENDERING = "RENDERING"
    DONE = "DONE"
    FAILED = "FAILED"


class OutputFormat(StrEnum):
    """Format output klip yang dikonfirmasi user lewat tombol Telegram
    sebelum job dibuat -- lihat bot.py `handle_format_choice`."""

    VERTICAL = "vertical"
    ORIGINAL = "original"


class Clip(BaseModel):
    """Satu klip hasil analisis LLM, sesuai skema BRIEF.md bagian 4."""

    start: str
    end: str
    hook_reason: str
    virality_score: int = Field(ge=1, le=10)
    title: str = Field(max_length=100)
    description: str
    tags: list[str]
    hashtags: list[str]

    @field_validator("start", "end", mode="before")
    @classmethod
    def _normalize_timestamp_format(cls, value: object) -> object:
        if isinstance(value, str):
            return normalize_timestamp(value)
        return value

    @model_validator(mode="after")
    def _validate_duration(self) -> Clip:
        start_seconds = parse_timestamp(self.start)
        end_seconds = parse_timestamp(self.end)
        if end_seconds <= start_seconds:
            raise ValueError(f"end ({self.end}) harus lebih besar dari start ({self.start})")

        settings = get_settings()
        duration = end_seconds - start_seconds
        if not (settings.clip_min_seconds <= duration <= settings.clip_max_seconds):
            raise ValueError(
                f"Durasi klip {duration}s di luar batas "
                f"[{settings.clip_min_seconds}, {settings.clip_max_seconds}]s"
            )
        return self

    @property
    def start_seconds(self) -> int:
        return parse_timestamp(self.start)

    @property
    def end_seconds(self) -> int:
        return parse_timestamp(self.end)


class ClipPlan(BaseModel):
    """Output lengkap LLM: ringkasan video + daftar klip kandidat."""

    video_summary: str
    clips: list[Clip]

    @model_validator(mode="after")
    def _validate_clip_count(self) -> ClipPlan:
        if len(self.clips) == 0:
            raise ValueError("ClipPlan harus punya minimal 1 klip")

        settings = get_settings()
        if len(self.clips) > settings.max_clips_per_video:
            raise ValueError(
                f"Jumlah klip ({len(self.clips)}) melebihi batas {settings.max_clips_per_video}"
            )
        return self

    @model_validator(mode="after")
    def _validate_no_excess_overlap(self) -> ClipPlan:
        sorted_clips = sorted(self.clips, key=lambda clip: clip.start_seconds)
        for prev, curr in zip(sorted_clips, sorted_clips[1:], strict=False):
            overlap = prev.end_seconds - curr.start_seconds
            if overlap > _MAX_OVERLAP_SECONDS:
                raise ValueError(
                    f"Klip {prev.start}-{prev.end} dan {curr.start}-{curr.end} "
                    f"overlap {overlap}s (maks {_MAX_OVERLAP_SECONDS}s)"
                )
        return self

    def validate_within_video_duration(self, video_duration_seconds: int) -> None:
        """Pastikan tidak ada klip yang `end`-nya melewati durasi video sumber.

        Dipanggil terpisah oleh analyze.py karena durasi video baru diketahui
        setelah tahap download/transcribe, bukan pada saat parsing JSON LLM.
        """
        for clip in self.clips:
            if clip.end_seconds > video_duration_seconds:
                raise ValueError(
                    f"Klip end {clip.end} melebihi durasi video sumber ({video_duration_seconds}s)"
                )


class Job(BaseModel):
    """Representasi satu baris tabel `jobs` di db.py."""

    job_id: str
    url: str
    chat_id: int
    user_id: int
    status: JobStatus = JobStatus.QUEUED
    output_format: OutputFormat = OutputFormat.VERTICAL
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime
