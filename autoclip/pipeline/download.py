"""Unduh video + audio sumber via yt-dlp.

Mengunduh video mentah (`raw.mp4`) ke `data/media/<job_id>/` via yt-dlp,
lalu mengekstrak audio (`audio.m4a`) dari file tersebut via FFmpeg
untuk kebutuhan transkripsi (BRIEF.md bagian 5.1). Kedua tahap
idempotent: jika artefak sudah ada dari percobaan sebelumnya, tahap
terkait dilewati agar retry tidak mengulang dari awal. Melempar
`DownloadError` bila yt-dlp atau ffmpeg gagal (link invalid, video
privat/geo-blok, dsb.) agar worker bisa menandai job FAILED dengan
pesan jelas.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import yt_dlp

from autoclip.config import get_settings

logger = logging.getLogger(__name__)


class DownloadError(Exception):
    """Dilempar bila proses download (yt-dlp) atau ekstraksi audio (ffmpeg) gagal."""


async def download_video(job_id: str, url: str) -> Path:
    """Unduh video sumber ke `data/media/<job_id>/raw.mp4` via yt-dlp.

    Idempotent: bila `raw.mp4` sudah ada (retry setelah tahap
    berikutnya gagal), download tidak diulang.
    """
    settings = get_settings()
    settings.ensure_job_dir(job_id)
    output_path = settings.raw_video_path(job_id)

    if output_path.exists():
        logger.info("job_id=%s raw video sudah ada, skip download", job_id)
        return output_path

    logger.info("job_id=%s mulai download %s", job_id, url)
    try:
        await asyncio.to_thread(_run_ytdlp, url, output_path)
    except yt_dlp.utils.DownloadError as exc:
        raise DownloadError(f"yt-dlp gagal mengunduh {url}: {exc}") from exc

    if not output_path.exists():
        raise DownloadError(f"yt-dlp selesai tanpa error tapi file tidak ditemukan: {output_path}")

    logger.info("job_id=%s download selesai -> %s", job_id, output_path)
    return output_path


def _run_ytdlp(url: str, output_path: Path) -> None:
    """Panggil yt_dlp.YoutubeDL secara sinkron (dijalankan di thread terpisah)."""
    ydl_opts = {
        # Prioritaskan H.264 (avc1): OpenCV (facecrop.py) memakai FFmpeg bundel-nya
        # sendiri yang di banyak environment tidak bisa decode AV1 sama sekali
        # (cv2.VideoCapture.read() gagal total, tanpa error eksplisit) — akibatnya
        # crop diam-diam fallback ke tengah frame, bukan berpusat wajah. AV1 tetap
        # jadi fallback terakhir kalau video itu memang cuma tersedia dalam AV1.
        "format": (
            "bestvideo[ext=mp4][vcodec^=avc1]+bestaudio[ext=m4a]"
            "/best[ext=mp4][vcodec^=avc1]"
            "/bestvideo[ext=mp4]+bestaudio[ext=m4a]"
            "/best[ext=mp4]/best"
        ),
        "outtmpl": str(output_path),
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    _download_subtitles_best_effort(url, output_path)


def _download_subtitles_best_effort(url: str, output_path: Path) -> None:
    """Unduh subtitle/caption YouTube (manual + auto-generated) sebagai bonus.

    Dilakukan sebagai panggilan yt-dlp TERPISAH setelah video utama sukses,
    dan kegagalannya (mis. rate-limit 429 di endpoint timedtext YouTube —
    endpoint ini jauh lebih ketat rate-limit-nya daripada endpoint video)
    sengaja HANYA di-log sebagai warning, tidak pernah melempar DownloadError.
    Tidak ada tahap pipeline saat ini yang membaca file subtitle ini; ini
    murni artefak tambahan, jadi kegagalannya tidak boleh menjatuhkan job.
    """
    subtitle_opts = {
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        # Default yt-dlp adalah 'en' kalau tidak di-set — konten proyek ini
        # podcast berbahasa Indonesia (lihat BRIEF.md), jadi minta 'id'.
        "subtitleslangs": ["id"],
        "outtmpl": str(output_path.with_suffix("")),
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
    }
    try:
        with yt_dlp.YoutubeDL(subtitle_opts) as ydl:
            ydl.download([url])
    except Exception as exc:
        logger.warning("Gagal mengunduh subtitle/caption YouTube (diabaikan): %s", exc)
    else:
        logger.info("Subtitle/caption YouTube berhasil diunduh (kalau tersedia)")


async def extract_audio(job_id: str, video_path: Path | None = None) -> Path:
    """Ekstrak audio dari `raw.mp4` ke `audio.m4a` via ffmpeg (untuk faster-whisper).

    Idempotent seperti `download_video`.
    """
    settings = get_settings()
    settings.ensure_job_dir(job_id)
    source = video_path if video_path is not None else settings.raw_video_path(job_id)
    output_path = settings.audio_path(job_id)

    if output_path.exists():
        logger.info("job_id=%s audio sudah ada, skip ekstraksi", job_id)
        return output_path

    if not source.exists():
        raise DownloadError(f"Video sumber tidak ditemukan untuk ekstraksi audio: {source}")

    # Tulis ke path sementara lalu rename setelah sukses (atomic), supaya proses
    # yang terputus di tengah jalan (crash/kill/OOM) tidak meninggalkan audio.m4a
    # yang rusak tapi tetap dianggap "sudah ada" oleh pengecekan idempotency di atas.
    # Ekstensi asli (.m4a) sengaja dipertahankan di akhir nama file sementara
    # karena ffmpeg menebak muxer output dari ekstensi (".part" saja tidak dikenali).
    tmp_path = output_path.with_name(f"{output_path.stem}.part{output_path.suffix}")
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(source),
        "-vn",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        str(tmp_path),
    ]
    logger.info("job_id=%s ekstrak audio -> %s", job_id, output_path)
    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await process.communicate()
    if process.returncode != 0:
        tmp_path.unlink(missing_ok=True)
        raise DownloadError(
            f"ffmpeg gagal mengekstrak audio dari {source}: "
            f"{stderr.decode(errors='replace').strip()}"
        )

    tmp_path.rename(output_path)
    logger.info("job_id=%s ekstraksi audio selesai -> %s", job_id, output_path)
    return output_path


if __name__ == "__main__":
    import sys

    async def _main() -> None:
        if len(sys.argv) != 3:
            print("Pemakaian: python -m autoclip.pipeline.download <job_id> <url>")
            raise SystemExit(1)
        job_id, url = sys.argv[1], sys.argv[2]

        video_path = await download_video(job_id, url)
        print(f"raw video -> {video_path} ({video_path.stat().st_size:,} bytes)")

        audio_path = await extract_audio(job_id)
        print(f"audio     -> {audio_path} ({audio_path.stat().st_size:,} bytes)")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(_main())
