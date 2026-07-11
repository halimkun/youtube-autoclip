"""Test untuk models.py: parser timestamp, validasi Clip & ClipPlan."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError

from autoclip.models import (
    Clip,
    ClipPlan,
    Job,
    JobStatus,
    TimestampFormatError,
    normalize_timestamp,
    parse_timestamp,
)


def test_parse_timestamp_valid() -> None:
    assert parse_timestamp("00:04:12") == 4 * 60 + 12
    assert parse_timestamp("01:00:00") == 3600
    assert parse_timestamp("00:00:00") == 0


def test_parse_timestamp_accepts_mmss_format() -> None:
    # LLM kadang mengirim "MM:SS" (tanpa jam) untuk video di bawah satu jam.
    assert parse_timestamp("04:12") == 4 * 60 + 12
    assert parse_timestamp("4:12") == 4 * 60 + 12
    assert parse_timestamp("59:59") == 59 * 60 + 59


@pytest.mark.parametrize("value", ["00:60:00", "00:00:60", "abc", "00:04:12:00", "", "60:00"])
def test_parse_timestamp_invalid(value: str) -> None:
    with pytest.raises(TimestampFormatError):
        parse_timestamp(value)


def test_normalize_timestamp_converts_mmss_to_hhmmss() -> None:
    # Kasus nyata: LLM mengirim "08:17" (8 menit 17 detik) — HARUS jadi
    # "00:08:17", BUKAN "08:17:00" (yang berarti 8 jam 17 menit, kesalahan
    # yang pernah terjadi ketika LLM "memperbaiki" formatnya sendiri).
    assert normalize_timestamp("08:17") == "00:08:17"
    assert normalize_timestamp("00:08:17") == "00:08:17"
    assert normalize_timestamp("1:05:00") == "01:05:00"


def _clip_data(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "start": "00:04:12",
        "end": "00:05:28",
        "hook_reason": "momen lucu",
        "virality_score": 8,
        "title": "Judul hook",
        "description": "Deskripsi singkat.",
        "tags": ["tag1", "tag2"],
        "hashtags": ["#shorts"],
    }
    data.update(overrides)
    return data


def _plan_data(clips: list[dict[str, Any]]) -> dict[str, Any]:
    return {"video_summary": "Ringkasan video", "clips": clips}


def test_clip_valid() -> None:
    clip = Clip(**_clip_data())
    assert clip.start_seconds == 4 * 60 + 12
    assert clip.end_seconds == 5 * 60 + 28


def test_clip_rejects_invalid_timestamp_format() -> None:
    with pytest.raises(ValidationError):
        Clip(**_clip_data(start="not-a-timestamp"))


def test_clip_normalizes_mmss_start_to_hhmmss() -> None:
    # Skenario nyata dari LLM: start/end dikirim sebagai "MM:SS".
    clip = Clip(**_clip_data(start="08:17", end="09:59"))

    assert clip.start == "00:08:17"
    assert clip.end == "00:09:59"
    assert clip.start_seconds == 8 * 60 + 17
    assert clip.end_seconds == 9 * 60 + 59


def test_clip_rejects_end_before_start() -> None:
    with pytest.raises(ValidationError):
        Clip(**_clip_data(start="00:05:00", end="00:04:00"))


def test_clip_rejects_duration_too_short() -> None:
    with pytest.raises(ValidationError):
        Clip(**_clip_data(start="00:00:00", end="00:00:10"))


def test_clip_rejects_duration_too_long() -> None:
    with pytest.raises(ValidationError):
        Clip(**_clip_data(start="00:00:00", end="00:10:00"))


def test_clip_rejects_virality_score_out_of_range() -> None:
    with pytest.raises(ValidationError):
        Clip(**_clip_data(virality_score=11))


def test_clip_plan_valid() -> None:
    plan = ClipPlan(**_plan_data([_clip_data()]))
    assert len(plan.clips) == 1


def test_clip_plan_rejects_empty_clips() -> None:
    with pytest.raises(ValidationError):
        ClipPlan(**_plan_data([]))


def test_clip_plan_rejects_too_many_clips() -> None:
    clips = [
        _clip_data(start=f"00:{i * 2:02d}:00", end=f"00:{i * 2:02d}:40")
        for i in range(9)  # melebihi MAX_CLIPS_PER_VIDEO=8 dari env test
    ]
    with pytest.raises(ValidationError):
        ClipPlan(**_plan_data(clips))


def test_clip_plan_allows_overlap_within_limit() -> None:
    clip_a = _clip_data(start="00:00:00", end="00:01:00")
    clip_b = _clip_data(start="00:00:55", end="00:02:00")  # overlap 5s, dalam batas

    plan = ClipPlan(**_plan_data([clip_a, clip_b]))

    assert len(plan.clips) == 2


def test_clip_plan_rejects_excess_overlap() -> None:
    clip_a = _clip_data(start="00:00:00", end="00:01:00")
    clip_b = _clip_data(start="00:00:30", end="00:02:00")  # overlap 30s, melebihi batas 10s

    with pytest.raises(ValidationError):
        ClipPlan(**_plan_data([clip_a, clip_b]))


def test_clip_plan_validate_within_video_duration_ok() -> None:
    plan = ClipPlan(**_plan_data([_clip_data()]))
    plan.validate_within_video_duration(video_duration_seconds=600)


def test_clip_plan_validate_within_video_duration_raises() -> None:
    plan = ClipPlan(**_plan_data([_clip_data()]))
    with pytest.raises(ValueError, match="melebihi durasi video sumber"):
        plan.validate_within_video_duration(video_duration_seconds=300)


def test_job_defaults_to_queued_status() -> None:
    now = datetime.now(UTC)
    job = Job(
        job_id="abc123",
        url="https://youtu.be/xyz",
        chat_id=1,
        user_id=2,
        created_at=now,
        updated_at=now,
    )
    assert job.status is JobStatus.QUEUED
    assert job.error_message is None


def test_job_status_values() -> None:
    assert JobStatus.QUEUED == "QUEUED"
    assert JobStatus.FAILED == "FAILED"
