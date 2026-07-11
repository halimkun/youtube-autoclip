"""Cutting, crop 9:16, dan burn subtitle per klip via FFmpeg subprocess.

Menjalankan FFmpeg (dipanggil sebagai list argumen, bukan string shell)
melalui `asyncio.create_subprocess_exec`, satu pass per klip, urutan
sesuai BRIEF.md bagian 5.3: (1) cut dengan re-encode frame-accurate
(bukan stream copy) via filter `trim`/`atrim` + `setpts`/`asetpts`
(bukan `-ss`/`-to` sebagai opsi output biasa — itu membuat filter `ass`
menerima PTS absolut asli, bukan relatif ke awal klip, sehingga subtitle
tidak pernah muncul; lihat komentar di `_build_ffmpeg_command`). TIDAK
memakai fast-seek input (`-ss` sebelum `-i`) — pernah dicoba untuk
mempercepat klip yang jauh di tengah video panjang, tapi perilaku PTS
setelah seek input ternyata TIDAK konsisten antar video (kadang tetap
absolut, kadang di-reset ke 0), yang bikin `trim=start=<detik absolut>`
kadang tidak menemukan frame sama sekali dan FFmpeg diam-diam menulis
file MP4 kosong (exit code 0, tapi nb_streams=0) — bukan error yang
kelihatan, hanya klip corrupt/tidak bisa dibuka. Decode dari awal file
lebih lambat tapi selalu benar; (2) crop 9:16 berpusat wajah (dari
`facecrop.py`) + scale ke 1080x1920, (3) burn subtitle `.ass` (dari
`subtitles.py`). Output H.264 CRF 21 preset veryfast, AAC 128k, nama
file `<slug-judul>-<n>.mp4`. Melempar `RenderError` bila FFmpeg (atau
deteksi wajah) gagal di tahap mana pun.
"""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path

from autoclip.config import get_settings
from autoclip.models import Clip, ClipPlan
from autoclip.pipeline.facecrop import (
    CropWindow,
    FaceDetector,
    SplitCropWindow,
    compute_crop_window,
)
from autoclip.pipeline.subtitles import write_ass_file
from autoclip.pipeline.transcribe import Transcript

logger = logging.getLogger(__name__)

_SLUG_RE = re.compile(r"[^a-z0-9]+")
_MAX_SLUG_LENGTH = 60
# Klip video 9:16 sungguhan (bahkan beberapa detik) selalu jauh di atas ini;
# file di bawah ambang ini adalah kontainer MP4 kosong (lihat komentar di
# _build_ffmpeg_command), bukan video yang cuma kecil.
_MIN_VALID_CLIP_BYTES = 10_000


class RenderError(Exception):
    """Dilempar bila proses FFmpeg (cut/crop/burn) atau deteksi wajah gagal."""


async def render_clip(
    job_id: str,
    clip_index: int,
    clip: Clip,
    transcript: Transcript,
    video_path: Path | None = None,
    detector: FaceDetector | None = None,
) -> Path:
    """Render satu klip: cut + crop 9:16 + burn subtitle dalam satu pass ffmpeg.

    Idempotent: kalau file output sudah ada, tidak di-render ulang.
    """
    settings = get_settings()
    settings.ensure_job_dir(job_id)
    source = video_path if video_path is not None else settings.raw_video_path(job_id)
    output_path = clip_output_path(job_id, clip_index, clip)

    if output_path.exists():
        logger.info("job_id=%s klip %s sudah ada, skip render", job_id, clip_index)
        return output_path

    if not source.exists():
        raise RenderError(f"Video sumber tidak ditemukan untuk render: {source}")

    start_seconds = clip.start_seconds
    end_seconds = clip.end_seconds

    try:
        crop_window = await asyncio.to_thread(
            compute_crop_window, source, start_seconds, end_seconds, detector
        )
    except Exception as exc:
        raise RenderError(f"Deteksi wajah gagal untuk klip {clip_index}: {exc}") from exc

    ass_path = write_ass_file(job_id, clip_index, transcript, start_seconds, end_seconds)

    # Tulis ke path sementara lalu rename setelah sukses (atomic), supaya proses
    # yang terputus di tengah jalan (crash/kill/OOM) tidak meninggalkan file klip
    # rusak yang tetap dianggap "sudah ada" oleh pengecekan idempotency di atas.
    # Ekstensi asli (.mp4) sengaja dipertahankan di akhir nama file sementara
    # karena ffmpeg menebak muxer output dari ekstensi (".part" saja tidak dikenali).
    tmp_path = output_path.with_name(f"{output_path.stem}.part{output_path.suffix}")
    command = _build_ffmpeg_command(
        source, start_seconds, end_seconds, crop_window, ass_path, tmp_path
    )

    logger.info("job_id=%s render klip %s -> %s", job_id, clip_index, output_path)
    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await process.communicate()
    stderr_text = stderr.decode(errors="replace").strip()
    if process.returncode != 0:
        tmp_path.unlink(missing_ok=True)
        raise RenderError(f"ffmpeg gagal merender klip {clip_index}: {stderr_text}")

    # ffmpeg bisa exit 0 tapi menghasilkan MP4 kosong tanpa stream sama sekali
    # (mis. filter trim tidak menemukan frame yang cocok) — "sukses" itu palsu.
    # Jangan percaya returncode saja; tolak file yang jelas-jelas terlalu kecil
    # untuk berisi video sungguhan, supaya tidak ada klip corrupt yang lolos.
    output_size = tmp_path.stat().st_size
    if output_size < _MIN_VALID_CLIP_BYTES:
        tmp_path.unlink(missing_ok=True)
        raise RenderError(
            f"ffmpeg klip {clip_index} 'berhasil' tapi outputnya kosong/nyaris kosong "
            f"({output_size} bytes) — kemungkinan filter trim tidak menemukan frame yang "
            f"cocok. stderr: {stderr_text}"
        )

    tmp_path.rename(output_path)
    logger.info("job_id=%s render klip %s selesai -> %s", job_id, clip_index, output_path)
    return output_path


async def render_all_clips(
    job_id: str,
    clip_plan: ClipPlan,
    transcript: Transcript,
    video_path: Path | None = None,
    detector: FaceDetector | None = None,
) -> list[Path]:
    """Render semua klip di `ClipPlan` secara berurutan (satu worker, serial)."""
    return [
        await render_clip(job_id, index, clip, transcript, video_path=video_path, detector=detector)
        for index, clip in enumerate(clip_plan.clips)
    ]


def clip_output_path(job_id: str, clip_index: int, clip: Clip) -> Path:
    """Path deterministik output klip — satu-satunya sumber kebenaran nama file,
    dipakai render.py maupun bot.py (untuk mengirim hasil tanpa perlu glob folder).
    """
    return get_settings().clips_dir(job_id) / f"{_slugify(clip.title)}-{clip_index}.mp4"


def _build_ffmpeg_command(
    source: Path,
    start_seconds: int,
    end_seconds: int,
    crop_window: CropWindow | SplitCropWindow,
    ass_path: Path,
    output_path: Path,
) -> list[str]:
    # SENGAJA TIDAK pakai fast-seek input (-ss sebelum -i) — lihat penjelasan
    # panjang di docstring modul. Decode selalu dari awal file (lambat untuk
    # klip yang jauh di tengah video panjang) supaya PTS yang dilihat filter
    # `trim`/`atrim` SELALU absolut & konsisten, tidak pernah menghasilkan
    # klip kosong secara diam-diam.
    #
    # Pemotongan presisi & frame-accurate dilakukan oleh filter `trim`/`atrim`
    # (bukan `-ss`/`-to` sebagai opsi output), lalu `setpts`/`asetpts`
    # me-reset PTS ke 0 SETELAH trim. Ini penting: kalau `-ss`/`-to` dipakai
    # sebagai opsi output biasa, filter `ass` (subtitle) akan menerima PTS
    # absolut asli video (bukan relatif ke awal klip), sehingga event
    # Dialogue di file .ass (yang relatif ke awal klip) tidak pernah cocok
    # dan subtitle tidak pernah muncul sama sekali.
    if isinstance(crop_window, SplitCropWindow):
        video_args = [
            "-filter_complex",
            _build_split_filter_complex(start_seconds, end_seconds, crop_window, ass_path),
            "-map",
            "[outv]",
            "-map",
            "0:a:0",
        ]
    else:
        filter_chain = (
            f"trim=start={start_seconds}:end={end_seconds},"
            "setpts=PTS-STARTPTS,"
            f"crop={crop_window.width}:{crop_window.height}:{crop_window.x}:{crop_window.y},"
            "scale=1080:1920,"
            f"ass={_escape_filter_path(ass_path)}"
        )
        video_args = ["-vf", filter_chain]

    audio_filter = f"atrim=start={start_seconds}:end={end_seconds},asetpts=PTS-STARTPTS"
    return [
        "ffmpeg",
        "-y",
        "-i",
        str(source),
        *video_args,
        "-af",
        audio_filter,
        "-c:v",
        "libx264",
        "-crf",
        "21",
        "-preset",
        "veryfast",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        str(output_path),
    ]


def _build_split_filter_complex(
    start_seconds: int,
    end_seconds: int,
    crop_window: SplitCropWindow,
    ass_path: Path,
) -> str:
    """Filtergraph split-screen: crop dua strip 9:8 (atas=top, bawah=bottom)
    dari sumber yang sama, lalu tumpuk (`vstack`) jadi 1080x1920 sebelum
    subtitle di-burn di atas hasil gabungan (bukan per-strip)."""
    top, bottom = crop_window.top, crop_window.bottom
    ass_arg = _escape_filter_path(ass_path)
    return (
        f"[0:v]trim=start={start_seconds}:end={end_seconds},"
        "setpts=PTS-STARTPTS,split=2[vtop][vbot];"
        f"[vtop]crop={top.width}:{top.height}:{top.x}:{top.y},scale=1080:960[ctop];"
        f"[vbot]crop={bottom.width}:{bottom.height}:{bottom.x}:{bottom.y},scale=1080:960[cbottom];"
        f"[ctop][cbottom]vstack=inputs=2,ass={ass_arg}[outv]"
    )


def _escape_filter_path(path: Path) -> str:
    # ffmpeg filtergraph memperlakukan '\\' dan ':' secara khusus di argumen filter.
    return str(path).replace("\\", "\\\\").replace(":", "\\:")


def _slugify(text: str) -> str:
    slug = _SLUG_RE.sub("-", text.lower()).strip("-")
    return slug[:_MAX_SLUG_LENGTH].strip("-") or "clip"


if __name__ == "__main__":
    import json
    import sys

    async def _main() -> None:
        if len(sys.argv) != 5:
            print(
                "Pemakaian: python -m autoclip.pipeline.render "
                "<job_id> <video_path> <transcript_json_path> <clip_json_path>"
            )
            raise SystemExit(1)
        job_id = sys.argv[1]
        video_path, transcript_path, clip_path = (Path(a) for a in sys.argv[2:5])

        transcript = Transcript.model_validate_json(transcript_path.read_text())
        clip = Clip.model_validate_json(clip_path.read_text())

        output_path = await render_clip(job_id, 0, clip, transcript, video_path=video_path)
        print(f"clip -> {output_path} ({output_path.stat().st_size:,} bytes)")
        print(json.dumps(clip.model_dump(), indent=2, ensure_ascii=False))

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(_main())
