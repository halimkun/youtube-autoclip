"""Lapisan penyimpanan SQLite untuk antrean job.

Mengelola tabel `jobs` (job_id, url, status, error_message, timestamp,
dsb.) beserta transisi status yang valid antar tahap pipeline. Path
artefak per job (`data/media/<job_id>/...`) sendiri disediakan oleh
helper di `config.py` (`Settings.job_dir` dkk.), dipakai konsisten oleh
seluruh modul pipeline.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

from autoclip.config import get_settings
from autoclip.models import Job, JobStatus

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    url TEXT NOT NULL,
    chat_id INTEGER NOT NULL,
    user_id INTEGER NOT NULL,
    status TEXT NOT NULL,
    error_message TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""

# State machine pipeline (BRIEF.md bagian 2). FAILED bisa retry langsung ke
# tahap yang gagal (artefak sebelumnya tidak dihapus, lihat CLAUDE.md aturan #3).
# DONE adalah status terminal.
_VALID_TRANSITIONS: dict[JobStatus, frozenset[JobStatus]] = {
    JobStatus.QUEUED: frozenset({JobStatus.DOWNLOADING, JobStatus.FAILED}),
    JobStatus.DOWNLOADING: frozenset({JobStatus.TRANSCRIBING, JobStatus.FAILED}),
    JobStatus.TRANSCRIBING: frozenset({JobStatus.ANALYZING, JobStatus.FAILED}),
    JobStatus.ANALYZING: frozenset({JobStatus.RENDERING, JobStatus.FAILED}),
    JobStatus.RENDERING: frozenset({JobStatus.DONE, JobStatus.FAILED}),
    JobStatus.FAILED: frozenset(
        {
            JobStatus.DOWNLOADING,
            JobStatus.TRANSCRIBING,
            JobStatus.ANALYZING,
            JobStatus.RENDERING,
        }
    ),
    JobStatus.DONE: frozenset(),
}


class JobNotFoundError(Exception):
    """Dilempar bila job_id tidak ditemukan di database."""


class InvalidTransitionError(Exception):
    """Dilempar bila perpindahan status job melanggar state machine pipeline."""

    def __init__(self, current: JobStatus, target: JobStatus) -> None:
        self.current = current
        self.target = target
        super().__init__(f"Transisi status {current} -> {target} tidak valid")


def is_valid_transition(current: JobStatus, target: JobStatus) -> bool:
    """Cek apakah perpindahan status current -> target diperbolehkan."""
    return target in _VALID_TRANSITIONS[current]


class JobStore:
    """Akses tabel `jobs` di SQLite. Satu koneksi baru dibuka per operasi
    (sesuai prinsip "satu worker dulu, serial" di BRIEF.md bagian 7)."""

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path or get_settings().db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def create_job(self, job_id: str, url: str, chat_id: int, user_id: int) -> Job:
        """Buat job baru dengan status awal QUEUED."""
        now = datetime.now(UTC)
        job = Job(
            job_id=job_id,
            url=url,
            chat_id=chat_id,
            user_id=user_id,
            status=JobStatus.QUEUED,
            created_at=now,
            updated_at=now,
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO jobs
                    (job_id, url, chat_id, user_id, status, error_message, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job.job_id,
                    job.url,
                    job.chat_id,
                    job.user_id,
                    job.status.value,
                    job.error_message,
                    job.created_at.isoformat(),
                    job.updated_at.isoformat(),
                ),
            )
        return job

    def get_job(self, job_id: str) -> Job:
        """Ambil satu job. Melempar `JobNotFoundError` bila tidak ada."""
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        if row is None:
            raise JobNotFoundError(f"Job {job_id!r} tidak ditemukan")
        return _row_to_job(row)

    def update_status(
        self,
        job_id: str,
        target: JobStatus,
        error_message: str | None = None,
    ) -> Job:
        """Pindahkan status job. Melempar `InvalidTransitionError` bila
        transisi current -> target tidak diperbolehkan state machine."""
        job = self.get_job(job_id)
        if not is_valid_transition(job.status, target):
            raise InvalidTransitionError(job.status, target)

        now = datetime.now(UTC)
        with self._connect() as conn:
            conn.execute(
                "UPDATE jobs SET status = ?, error_message = ?, updated_at = ? WHERE job_id = ?",
                (target.value, error_message, now.isoformat(), job_id),
            )
        return self.get_job(job_id)

    def next_queued_job(self) -> Job | None:
        """Ambil job QUEUED tertua (FIFO) untuk diproses worker, atau None."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE status = ? ORDER BY created_at ASC LIMIT 1",
                (JobStatus.QUEUED.value,),
            ).fetchone()
        return _row_to_job(row) if row is not None else None


def _row_to_job(row: sqlite3.Row) -> Job:
    return Job(
        job_id=row["job_id"],
        url=row["url"],
        chat_id=row["chat_id"],
        user_id=row["user_id"],
        status=JobStatus(row["status"]),
        error_message=row["error_message"],
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
    )


@lru_cache
def get_job_store() -> JobStore:
    """Singleton JobStore — panggil ini, jangan instansiasi JobStore() langsung."""
    return JobStore()
