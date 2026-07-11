"""Panggilan LLM (endpoint OpenAI-compatible) untuk memilih momen viral.

Mengirim transkrip ringkas (`Transcript.to_prompt_text()`) + durasi total
video ke LLM yang dikonfigurasi via env generik `LLM_BASE_URL`/
`LLM_API_KEY`/`LLM_MODEL` (OpenRouter, HF Inference Providers, atau
Ollama lokal — lihat BRIEF.md bagian 5.2). Modul ini TIDAK BOLEH
mengandung nama provider secara hardcoded — hanya bergantung pada
endpoint OpenAI-compatible generik.

Memvalidasi hasil JSON dengan Pydantic (`models.ClipPlan`); jika tidak
valid, retry permintaan LLM maksimal 2x dengan pesan error disertakan,
lalu melempar `AnalyzeError`.
"""

from __future__ import annotations

import asyncio
import logging

from openai import AsyncOpenAI

from autoclip.config import Settings, get_settings
from autoclip.models import ClipPlan
from autoclip.pipeline.transcribe import Transcript

logger = logging.getLogger(__name__)

# Percobaan pertama + retry maksimal 2x sesuai BRIEF.md bagian 4.
_MAX_ATTEMPTS = 3

# Headroom besar: reasoning tokens (model tertentu) + JSON ClipPlan penuh.
_MAX_OUTPUT_TOKENS = 8000

_SYSTEM_PROMPT = """\
Kamu adalah seorang viral clip analyst yang ahli menemukan momen paling
menarik dari podcast berbahasa Indonesia untuk dijadikan klip pendek
YouTube Shorts/TikTok/Reels.

Balas HANYA dengan JSON valid, tanpa markdown fence dan tanpa teks
tambahan apa pun, persis mengikuti skema berikut:
{
  "video_summary": "ringkasan 1-2 kalimat isi video",
  "clips": [
    {
      "start": "HH:MM:SS",
      "end": "HH:MM:SS",
      "hook_reason": "alasan singkat kenapa momen ini berpotensi viral",
      "virality_score": 1-10,
      "title": "judul hook maksimal 100 karakter",
      "description": "deskripsi 2-4 kalimat, boleh mengandung CTA",
      "tags": ["tag1", "tag2"],
      "hashtags": ["#shorts", "#podcast"]
    }
  ]
}

Tuliskan "hook_reason" dan "virality_score" SEBELUM "title"/"description"
pada tiap objek klip, supaya kamu bernalar dulu sebelum menulis metadata
final. Timestamp "end" tidak boleh melebihi durasi total video.

PENTING soal format "start"/"end": WAJIB tiga bagian jam:menit:detik,
SELALU diawali "00:" untuk video di bawah satu jam. Contoh: menit ke-8
detik ke-17 ditulis "00:08:17" — BUKAN "08:17" (kurang bagian jam) dan
BUKAN "08:17:00" (itu artinya 8 JAM 17 menit, bukan 8 menit 17 detik).\
"""


class AnalyzeError(Exception):
    """Dilempar bila LLM gagal mengembalikan JSON valid setelah retry habis."""


def _build_client(settings: Settings) -> AsyncOpenAI:
    return AsyncOpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key)


def _build_user_prompt(
    video_duration_seconds: float,
    transcript_text: str,
    previous_error: str | None,
) -> str:
    parts = [
        f"Durasi total video: {int(video_duration_seconds)} detik.",
        "Transkrip (format [MM:SS] kalimat):",
        transcript_text,
    ]
    if previous_error is not None:
        parts.append(
            "Percobaan sebelumnya gagal divalidasi dengan error berikut, perbaiki:\n"
            f"{previous_error}"
        )
    return "\n\n".join(parts)


async def _call_llm(client: AsyncOpenAI, model: str, user_prompt: str) -> str:
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        # Beberapa model (mis. Gemini 2.5 Flash) memakai "reasoning tokens"
        # tersembunyi yang ikut memotong max_tokens sebelum output JSON
        # sempat ditulis. Nilai kecil bikin content jadi None walau request
        # sukses (bukan error) — beri headroom besar untuk reasoning + JSON
        # ClipPlan penuh (sampai MAX_CLIPS_PER_VIDEO klip).
        max_tokens=_MAX_OUTPUT_TOKENS,
    )
    content = response.choices[0].message.content
    if not content:
        raise ValueError("LLM mengembalikan konten kosong (kemungkinan terpotong max_tokens)")
    return content


async def analyze_transcript(
    job_id: str,
    transcript: Transcript,
    video_duration_seconds: float | None = None,
) -> ClipPlan:
    """Minta LLM memilih momen viral dari transkrip, lalu validasi jadi `ClipPlan`.

    Idempotent: kalau `plan.json` sudah ada, dibaca ulang tanpa
    memanggil LLM lagi.
    """
    settings = get_settings()
    settings.ensure_job_dir(job_id)
    output_path = settings.plan_path(job_id)

    if output_path.exists():
        logger.info("job_id=%s plan sudah ada, skip analisis LLM", job_id)
        return ClipPlan.model_validate_json(output_path.read_text())

    duration = video_duration_seconds if video_duration_seconds is not None else transcript.duration
    client = _build_client(settings)
    transcript_text = transcript.to_prompt_text()

    last_error: str | None = None
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        user_prompt = _build_user_prompt(duration, transcript_text, last_error)
        try:
            raw_json = await _call_llm(client, settings.llm_model, user_prompt)
            plan = ClipPlan.model_validate_json(raw_json)
            plan.validate_within_video_duration(int(duration))
        except Exception as exc:
            logger.warning(
                "job_id=%s percobaan LLM %s/%s gagal: %s", job_id, attempt, _MAX_ATTEMPTS, exc
            )
            last_error = str(exc)
            continue

        output_path.write_text(plan.model_dump_json(indent=2))
        logger.info("job_id=%s analisis LLM selesai -> %s", job_id, output_path)
        return plan

    raise AnalyzeError(
        f"LLM gagal mengembalikan ClipPlan valid setelah {_MAX_ATTEMPTS} percobaan: {last_error}"
    )


if __name__ == "__main__":
    # Cek cepat: apakah LLM_BASE_URL/LLM_API_KEY/LLM_MODEL di .env valid?
    # Tidak menyentuh ClipPlan/transcript sama sekali — cuma satu chat
    # completion murah untuk verifikasi endpoint & kredensial.
    async def _main() -> None:
        settings = get_settings()
        print(f"LLM_BASE_URL = {settings.llm_base_url}")
        print(f"LLM_MODEL    = {settings.llm_model}")
        print("Mengirim test request ke LLM...")

        client = _build_client(settings)
        try:
            response = await client.chat.completions.create(
                model=settings.llm_model,
                messages=[{"role": "user", "content": "Balas dengan satu kata: OK"}],
                # Sebagian model (mis. Gemini 2.5 Flash) memakai reasoning
                # tokens tersembunyi yang ikut motong max_tokens — nilai
                # kecil bikin content jadi None walau auth sukses.
                max_tokens=50,
            )
        except Exception as exc:
            print(f"GAGAL — {type(exc).__name__}: {exc}")
            raise SystemExit(1) from exc

        content = response.choices[0].message.content
        print(f"OK — API key & endpoint valid. Respons model: {content!r}")

    asyncio.run(_main())
