"""Loop worker yang mengorkestrasi seluruh tahap pipeline per job.

Mengambil job berikutnya dari `db.py` (status QUEUED), lalu menjalankan
tahap secara berurutan: download -> transcribe -> analyze -> render
(subtitle & face-crop ada di dalam render.py), sambil memperbarui
status job di setiap perpindahan tahap. Menangkap exception spesifik
tiap tahap (`DownloadError`, `TranscribeError`, `AnalyzeError`,
`RenderError`) untuk menandai job FAILED dengan pesan jelas tanpa
menghapus artefak intermediate, agar retry bisa melanjutkan dari tahap
yang gagal tanpa mengulang dari awal.

Retry tidak butuh API terpisah: panggil `process_job(job_id)` lagi pada
job yang FAILED. Setiap fungsi tahap pipeline sudah idempotent (baca
ulang artefak yang sudah ada alih-alih mengulang kerja), dan
`_advance_status` hanya memindahkan status maju (tidak pernah mundur),
jadi urutan tahap yang sama aman dijalankan ulang baik untuk retry dari
FAILED maupun untuk melanjutkan job yang tersangkut di tengah jalan
akibat worker mati sebelum sempat menandai FAILED. Satu worker berjalan
serial (lihat BRIEF.md bagian 7).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from autoclip.db import JobStore, get_job_store
from autoclip.models import Job, JobStatus, OutputFormat
from autoclip.pipeline.analyze import AnalyzeError, analyze_transcript
from autoclip.pipeline.download import DownloadError, download_video, extract_audio
from autoclip.pipeline.render import RenderError, render_all_clips
from autoclip.pipeline.transcribe import TranscribeError, transcribe_audio

logger = logging.getLogger(__name__)

# Urutan tahap dalam pipeline; dipakai untuk menentukan apakah suatu
# transisi status maju (perlu dijalankan) atau sudah terlewati (skip).
_STAGE_ORDER: dict[JobStatus, int] = {
    JobStatus.QUEUED: 0,
    JobStatus.FAILED: 0,  # retry: mulai lagi dari awal, idempotency yang skip kerja selesai
    JobStatus.DOWNLOADING: 1,
    JobStatus.TRANSCRIBING: 2,
    JobStatus.ANALYZING: 3,
    JobStatus.RENDERING: 4,
    JobStatus.DONE: 5,
}

_StageFn = Callable[[str, Job, dict[str, Any]], Awaitable[None]]


async def _stage_download(job_id: str, job: Job, ctx: dict[str, Any]) -> None:
    ctx["video_path"] = await download_video(job_id, job.url)
    await extract_audio(job_id)


async def _stage_transcribe(job_id: str, job: Job, ctx: dict[str, Any]) -> None:
    ctx["transcript"] = await transcribe_audio(job_id)


async def _stage_analyze(job_id: str, job: Job, ctx: dict[str, Any]) -> None:
    ctx["clip_plan"] = await analyze_transcript(job_id, ctx["transcript"])


async def _stage_render(job_id: str, job: Job, ctx: dict[str, Any]) -> None:
    await render_all_clips(
        job_id, ctx["clip_plan"], ctx["transcript"], output_format=job.output_format
    )


_STAGES: list[tuple[JobStatus, _StageFn, type[Exception]]] = [
    (JobStatus.DOWNLOADING, _stage_download, DownloadError),
    (JobStatus.TRANSCRIBING, _stage_transcribe, TranscribeError),
    (JobStatus.ANALYZING, _stage_analyze, AnalyzeError),
    (JobStatus.RENDERING, _stage_render, RenderError),
]


def _advance_status(store: JobStore, job: Job, target: JobStatus) -> Job:
    """Pindah ke `target` kalau job belum sampai/lewat tahap itu; else no-op."""
    if _STAGE_ORDER[job.status] >= _STAGE_ORDER[target]:
        return job
    return store.update_status(job.job_id, target)


async def process_job(job_id: str) -> Job:
    """Jalankan seluruh pipeline untuk satu job sampai DONE atau FAILED.

    Aman dipanggil berulang pada job_id yang sama (retry dari FAILED,
    atau melanjutkan job yang tersangkut di tengah jalan).
    """
    store = get_job_store()
    job = store.get_job(job_id)

    if job.status is JobStatus.DONE:
        logger.info("job_id=%s sudah DONE, tidak ada yang perlu dikerjakan", job_id)
        return job

    ctx: dict[str, Any] = {}
    for target_status, stage_fn, error_type in _STAGES:
        job = _advance_status(store, job, target_status)
        try:
            await stage_fn(job_id, job, ctx)
        except error_type as exc:
            logger.error("job_id=%s gagal di tahap %s: %s", job_id, target_status.value, exc)
            return store.update_status(job_id, JobStatus.FAILED, error_message=str(exc))

    logger.info("job_id=%s pipeline selesai", job_id)
    return store.update_status(job_id, JobStatus.DONE)


async def run_once() -> Job | None:
    """Ambil satu job QUEUED (kalau ada) dan proses sampai selesai atau FAILED."""
    store = get_job_store()
    job = store.next_queued_job()
    if job is None:
        return None
    return await process_job(job.job_id)


async def run_forever(poll_interval_seconds: float = 5.0) -> None:
    """Loop worker: polling job QUEUED baru selama proses bot berjalan."""
    logger.info("worker dimulai, poll_interval=%.1fs", poll_interval_seconds)
    while True:
        job = await run_once()
        if job is None:
            await asyncio.sleep(poll_interval_seconds)


if __name__ == "__main__":
    import sys
    import uuid

    async def _main() -> None:
        if len(sys.argv) not in (2, 3):
            print("Pemakaian: python -m autoclip.worker <url> [vertical|original]")
            raise SystemExit(1)
        url = sys.argv[1]
        output_format = OutputFormat(sys.argv[2]) if len(sys.argv) == 3 else OutputFormat.VERTICAL
        job_id = uuid.uuid4().hex[:10]

        store = get_job_store()
        job = store.create_job(job_id, url, chat_id=0, user_id=0, output_format=output_format)
        print(f"job_id={job_id} dibuat, status={job.status.value}", flush=True)

        result = await process_job(job_id)
        print(f"job_id={job_id} selesai, status={result.status.value}", flush=True)
        if result.error_message:
            print(f"job_id={job_id} error: {result.error_message}", flush=True)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(_main())
