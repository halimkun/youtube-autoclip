"""Test untuk db.py: penyimpanan job & state machine transisi status."""

from __future__ import annotations

import pytest

from autoclip.config import get_settings
from autoclip.db import (
    InvalidTransitionError,
    JobNotFoundError,
    JobStore,
    get_job_store,
    is_valid_transition,
)
from autoclip.models import JobStatus


@pytest.fixture
def store(tmp_path) -> JobStore:
    return JobStore(db_path=tmp_path / "jobs.db")


def test_create_job_starts_as_queued(store: JobStore) -> None:
    job = store.create_job("job-1", "https://youtu.be/xyz", chat_id=1, user_id=2)

    assert job.status is JobStatus.QUEUED
    assert job.error_message is None
    assert job.url == "https://youtu.be/xyz"


def test_get_job_returns_persisted_job(store: JobStore) -> None:
    store.create_job("job-1", "https://youtu.be/xyz", chat_id=1, user_id=2)

    job = store.get_job("job-1")

    assert job.job_id == "job-1"
    assert job.chat_id == 1
    assert job.user_id == 2


def test_get_job_raises_when_missing(store: JobStore) -> None:
    with pytest.raises(JobNotFoundError):
        store.get_job("does-not-exist")


def test_update_status_follows_pipeline_order(store: JobStore) -> None:
    store.create_job("job-1", "https://youtu.be/xyz", chat_id=1, user_id=2)

    for target in (
        JobStatus.DOWNLOADING,
        JobStatus.TRANSCRIBING,
        JobStatus.ANALYZING,
        JobStatus.RENDERING,
        JobStatus.DONE,
    ):
        job = store.update_status("job-1", target)
        assert job.status is target


def test_update_status_rejects_skipping_stages(store: JobStore) -> None:
    store.create_job("job-1", "https://youtu.be/xyz", chat_id=1, user_id=2)

    with pytest.raises(InvalidTransitionError):
        store.update_status("job-1", JobStatus.ANALYZING)


def test_update_status_persists_error_message_on_failed(store: JobStore) -> None:
    store.create_job("job-1", "https://youtu.be/xyz", chat_id=1, user_id=2)
    store.update_status("job-1", JobStatus.DOWNLOADING)

    job = store.update_status("job-1", JobStatus.FAILED, error_message="yt-dlp gagal")

    assert job.status is JobStatus.FAILED
    assert job.error_message == "yt-dlp gagal"


def test_failed_job_can_retry_from_failed_stage(store: JobStore) -> None:
    store.create_job("job-1", "https://youtu.be/xyz", chat_id=1, user_id=2)
    store.update_status("job-1", JobStatus.DOWNLOADING)
    store.update_status("job-1", JobStatus.TRANSCRIBING)
    store.update_status("job-1", JobStatus.ANALYZING, error_message="LLM timeout")
    store.update_status("job-1", JobStatus.FAILED, error_message="LLM timeout")

    job = store.update_status("job-1", JobStatus.ANALYZING)

    assert job.status is JobStatus.ANALYZING


def test_done_is_terminal(store: JobStore) -> None:
    store.create_job("job-1", "https://youtu.be/xyz", chat_id=1, user_id=2)
    store.update_status("job-1", JobStatus.DOWNLOADING)
    store.update_status("job-1", JobStatus.TRANSCRIBING)
    store.update_status("job-1", JobStatus.ANALYZING)
    store.update_status("job-1", JobStatus.RENDERING)
    store.update_status("job-1", JobStatus.DONE)

    with pytest.raises(InvalidTransitionError):
        store.update_status("job-1", JobStatus.FAILED)


def test_next_queued_job_returns_oldest_first(store: JobStore) -> None:
    store.create_job("job-1", "https://youtu.be/1", chat_id=1, user_id=2)
    store.create_job("job-2", "https://youtu.be/2", chat_id=1, user_id=2)

    job = store.next_queued_job()

    assert job is not None
    assert job.job_id == "job-1"


def test_next_queued_job_skips_non_queued(store: JobStore) -> None:
    store.create_job("job-1", "https://youtu.be/1", chat_id=1, user_id=2)
    store.update_status("job-1", JobStatus.DOWNLOADING)
    store.create_job("job-2", "https://youtu.be/2", chat_id=1, user_id=2)

    job = store.next_queued_job()

    assert job is not None
    assert job.job_id == "job-2"


def test_next_queued_job_returns_none_when_empty(store: JobStore) -> None:
    assert store.next_queued_job() is None


@pytest.mark.parametrize(
    ("current", "target", "expected"),
    [
        (JobStatus.QUEUED, JobStatus.DOWNLOADING, True),
        (JobStatus.QUEUED, JobStatus.ANALYZING, False),
        (JobStatus.FAILED, JobStatus.RENDERING, True),
        (JobStatus.DONE, JobStatus.QUEUED, False),
    ],
)
def test_is_valid_transition(current: JobStatus, target: JobStatus, expected: bool) -> None:
    assert is_valid_transition(current, target) is expected


def test_get_job_store_uses_settings_db_path() -> None:
    job_store = get_job_store()

    assert job_store is get_job_store()  # singleton
    assert job_store._db_path == get_settings().db_path
    assert job_store._db_path.exists()
