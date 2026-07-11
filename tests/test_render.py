"""Test untuk pipeline/render.py: orkestrasi ffmpeg cut+crop+burn subtitle (ffmpeg asli)."""

from __future__ import annotations

from pathlib import Path

import pytest

from autoclip.config import get_settings
from autoclip.models import Clip, ClipPlan
from autoclip.pipeline.facecrop import CropWindow, FaceBox
from autoclip.pipeline.render import (
    RenderError,
    _escape_filter_path,
    _slugify,
    render_all_clips,
    render_clip,
)
from autoclip.pipeline.transcribe import Transcript, TranscriptSegment, TranscriptWord


class _FixedFaceDetector:
    def __init__(self, center_x: float = 320.0) -> None:
        self._center_x = center_x

    def detect(self, frame) -> list[FaceBox]:
        return [FaceBox(x=self._center_x - 50, y=0, width=100, height=100)]


@pytest.fixture(autouse=True)
def _short_clip_bounds(monkeypatch: pytest.MonkeyPatch):
    """Sample video di conftest cuma 3 detik; longgarkan batas durasi Clip untuk test ini."""
    monkeypatch.setenv("CLIP_MIN_SECONDS", "1")
    monkeypatch.setenv("CLIP_MAX_SECONDS", "10")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _clip(**overrides) -> Clip:
    data = {
        "start": "00:00:00",
        "end": "00:00:02",
        "hook_reason": "momen test",
        "virality_score": 5,
        "title": "Judul Klip Keren!",
        "description": "deskripsi",
        "tags": ["a"],
        "hashtags": ["#a"],
    }
    data.update(overrides)
    return Clip(**data)


@pytest.fixture
def transcript() -> Transcript:
    return Transcript(
        language="id",
        duration=3.0,
        segments=[
            TranscriptSegment(
                start=0.0,
                end=2.0,
                text="halo dunia",
                words=[
                    TranscriptWord(word="halo", start=0.0, end=0.5),
                    TranscriptWord(word="dunia", start=0.5, end=1.0),
                ],
            )
        ],
    )


async def test_render_clip_produces_vertical_mp4(
    sample_video: Path, transcript: Transcript
) -> None:
    output_path = await render_clip(
        "job-1", 0, _clip(), transcript, video_path=sample_video, detector=_FixedFaceDetector()
    )

    assert output_path.exists()
    assert output_path.name == "judul-klip-keren-0.mp4"
    assert output_path.parent == get_settings().clips_dir("job-1")
    assert output_path.stat().st_size > 0


async def test_render_clip_skips_when_already_rendered(
    sample_video: Path, transcript: Transcript
) -> None:
    settings = get_settings()
    settings.ensure_job_dir("job-1")
    existing_path = settings.clips_dir("job-1") / "judul-klip-keren-0.mp4"
    existing_path.write_bytes(b"already rendered")

    output_path = await render_clip(
        "job-1", 0, _clip(), transcript, video_path=sample_video, detector=_FixedFaceDetector()
    )

    assert output_path.read_bytes() == b"already rendered"


async def test_render_clip_raises_when_source_missing(transcript: Transcript) -> None:
    with pytest.raises(RenderError, match="tidak ditemukan"):
        await render_clip(
            "job-1",
            0,
            _clip(),
            transcript,
            video_path=Path("/does/not/exist.mp4"),
            detector=_FixedFaceDetector(),
        )


async def test_render_clip_wraps_facecrop_failure(
    sample_video: Path, transcript: Transcript
) -> None:
    class _BoomDetector:
        def detect(self, frame):
            raise RuntimeError("model rusak")

    with pytest.raises(RenderError, match="Deteksi wajah gagal"):
        await render_clip(
            "job-1", 0, _clip(), transcript, video_path=sample_video, detector=_BoomDetector()
        )


async def test_render_clip_raises_on_ffmpeg_failure(
    monkeypatch: pytest.MonkeyPatch, sample_video: Path, transcript: Transcript
) -> None:
    # Wajah terdeteksi normal, tapi window crop lebar=0 bikin filter ffmpeg gagal beneran.
    def _invalid_crop_window(*args: object, **kwargs: object) -> CropWindow:
        return CropWindow(x=0, y=0, width=0, height=0)

    monkeypatch.setattr("autoclip.pipeline.render.compute_crop_window", _invalid_crop_window)

    with pytest.raises(RenderError, match="ffmpeg gagal"):
        await render_clip(
            "job-1", 0, _clip(), transcript, video_path=sample_video, detector=_FixedFaceDetector()
        )


async def test_render_all_clips_renders_each_clip_in_order(
    sample_video: Path, transcript: Transcript
) -> None:
    plan = ClipPlan(
        video_summary="ringkasan",
        clips=[
            _clip(title="Klip Pertama", start="00:00:00", end="00:00:01"),
            _clip(title="Klip Kedua", start="00:00:01", end="00:00:02"),
        ],
    )

    paths = await render_all_clips(
        "job-1", plan, transcript, video_path=sample_video, detector=_FixedFaceDetector()
    )

    assert [p.name for p in paths] == ["klip-pertama-0.mp4", "klip-kedua-1.mp4"]
    assert all(p.exists() for p in paths)


def test_slugify_normalizes_and_truncates() -> None:
    assert _slugify("Judul Klip Keren!") == "judul-klip-keren"
    assert _slugify("   ") == "clip"
    assert _slugify("A" * 100) == "a" * 60


def test_escape_filter_path_escapes_colon_and_backslash() -> None:
    assert _escape_filter_path(Path("/tmp/a:b.ass")) == "/tmp/a\\:b.ass"
