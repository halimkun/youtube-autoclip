"""Test untuk pipeline/download.py: download.py (yt-dlp di-mock) & extract_audio (ffmpeg asli)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yt_dlp

from autoclip.config import get_settings
from autoclip.pipeline.download import DownloadError, download_video, extract_audio


class _FakeYDL:
    """Pengganti yt_dlp.YoutubeDL: menulis file dummy alih-alih download asli."""

    def __init__(self, opts: dict) -> None:
        self.opts = opts

    def __enter__(self) -> _FakeYDL:
        return self

    def __exit__(self, *exc_info: object) -> bool:
        return False

    def download(self, urls: list[str]) -> None:
        Path(self.opts["outtmpl"]).write_bytes(b"fake video bytes")


class _FailingYDL(_FakeYDL):
    def download(self, urls: list[str]) -> None:
        raise yt_dlp.utils.DownloadError("video tidak tersedia")


class _SilentYDL(_FakeYDL):
    """Simulasi yt-dlp yang "sukses" tanpa error tapi tidak menulis file apa pun."""

    def download(self, urls: list[str]) -> None:
        pass


class _VideoOkSubtitleRateLimitedYDL(_FakeYDL):
    """Video utama sukses; panggilan subtitle terpisah (skip_download) kena 429."""

    def download(self, urls: list[str]) -> None:
        if self.opts.get("skip_download"):
            raise yt_dlp.utils.DownloadError(
                "Unable to download video subtitles for 'en': HTTP Error 429: Too Many Requests"
            )
        Path(self.opts["outtmpl"]).write_bytes(b"fake video bytes")


async def test_download_video_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("autoclip.pipeline.download.yt_dlp.YoutubeDL", _FakeYDL)

    output_path = await download_video("job-1", "https://youtu.be/xyz")

    assert output_path == get_settings().raw_video_path("job-1")
    assert output_path.read_bytes() == b"fake video bytes"


async def test_download_video_skips_when_already_downloaded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = get_settings()
    settings.ensure_job_dir("job-1")
    settings.raw_video_path("job-1").write_bytes(b"already there")

    def _boom(opts: dict) -> None:
        raise AssertionError("YoutubeDL should not be called when raw.mp4 already exists")

    monkeypatch.setattr("autoclip.pipeline.download.yt_dlp.YoutubeDL", _boom)

    output_path = await download_video("job-1", "https://youtu.be/xyz")

    assert output_path.read_bytes() == b"already there"


async def test_download_video_ignores_subtitle_download_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "autoclip.pipeline.download.yt_dlp.YoutubeDL", _VideoOkSubtitleRateLimitedYDL
    )

    output_path = await download_video("job-1", "https://youtu.be/xyz")

    assert output_path.read_bytes() == b"fake video bytes"


async def test_download_video_wraps_ytdlp_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("autoclip.pipeline.download.yt_dlp.YoutubeDL", _FailingYDL)

    with pytest.raises(DownloadError, match="yt-dlp gagal"):
        await download_video("job-1", "https://youtu.be/xyz")


async def test_download_video_raises_when_file_missing_after_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("autoclip.pipeline.download.yt_dlp.YoutubeDL", _SilentYDL)

    with pytest.raises(DownloadError, match="tidak ditemukan"):
        await download_video("job-1", "https://youtu.be/xyz")


async def test_extract_audio_creates_m4a(sample_video: Path) -> None:
    output_path = await extract_audio("job-1", video_path=sample_video)

    assert output_path == get_settings().audio_path("job-1")
    assert output_path.exists()
    assert output_path.stat().st_size > 0


async def test_extract_audio_skips_when_already_extracted(sample_video: Path) -> None:
    settings = get_settings()
    settings.ensure_job_dir("job-1")
    settings.audio_path("job-1").write_bytes(b"already extracted")

    output_path = await extract_audio("job-1", video_path=sample_video)

    assert output_path.read_bytes() == b"already extracted"


async def test_extract_audio_raises_when_source_missing() -> None:
    with pytest.raises(DownloadError, match="tidak ditemukan"):
        await extract_audio("job-1", video_path=Path("/does/not/exist.mp4"))


async def test_extract_audio_raises_on_ffmpeg_failure(tmp_path: Path) -> None:
    bogus_video = tmp_path / "not-a-video.mp4"
    bogus_video.write_text("this is not a real video file")

    with pytest.raises(DownloadError, match="ffmpeg gagal"):
        await extract_audio("job-1", video_path=bogus_video)
