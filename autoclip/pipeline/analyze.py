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
# Video panjang (transkrip 1-2 jam bisa >18K token) butuh lebih banyak
# headroom lagi di percobaan retry (prompt-nya makin panjang tiap retry
# karena menyertakan pesan error percobaan sebelumnya) -- angka kecil
# terbukti memotong output JSON di tengah string pada video panjang.
_MAX_OUTPUT_TOKENS = 16000


def _build_system_prompt(settings: Settings) -> str:
    return f"""\
Kamu adalah seorang viral clip analyst yang ahli menemukan momen paling
menarik dari podcast berbahasa Indonesia untuk dijadikan klip pendek
YouTube Shorts/TikTok/Reels.

Balas HANYA dengan JSON valid, tanpa markdown fence dan tanpa teks
tambahan apa pun, persis mengikuti skema berikut:
{{
  "video_summary": "ringkasan 1-2 kalimat isi video",
  "clips": [
    {{
      "start": "HH:MM:SS",
      "end": "HH:MM:SS",
      "hook_reason": "alasan singkat kenapa momen ini berpotensi viral",
      "virality_score": 1-10,
      "title": "judul hook maksimal 100 karakter",
      "description": "deskripsi 2-4 kalimat, boleh mengandung CTA",
      "tags": ["tag1", "tag2"],
      "hashtags": ["#shorts", "#podcast"]
    }}
  ]
}}

Tuliskan "hook_reason" dan "virality_score" SEBELUM "title"/"description"
pada tiap objek klip, supaya kamu bernalar dulu sebelum menulis metadata
final.

ATURAN KETAT soal durasi & jumlah klip (response DITOLAK otomatis oleh
validator kalau dilanggar, jadi WAJIB dipatuhi persis):
- Durasi tiap klip ("end" dikurangi "start") WAJIB antara
  {settings.clip_min_seconds} dan {settings.clip_max_seconds} detik.
  Kalau sebuah momen menarik berlangsung lebih lama dari itu, potong ke
  bagian paling inti/viral saja -- JANGAN mengambil seluruh rentang topik.
- Jumlah klip dalam "clips" TIDAK BOLEH lebih dari
  {settings.max_clips_per_video}. Pilih yang paling berpotensi viral saja,
  bukan mencantumkan semua momen menarik yang ditemukan.
- Timestamp "start" dan "end" WAJIB berada dalam rentang durasi video
  yang diberikan di awal pesan user (0 sampai durasi total). JANGAN
  PERNAH menulis timestamp yang melebihi durasi video, dan JANGAN
  menebak/mengarang timestamp -- pakai HANYA posisi yang benar-benar ada
  di transkrip yang diberikan.

PENTING soal format "start"/"end": WAJIB tiga bagian jam:menit:detik,
SELALU diawali "00:" untuk video di bawah satu jam. Contoh: menit ke-8
detik ke-17 ditulis "00:08:17" — BUKAN "08:17" (kurang bagian jam) dan
BUKAN "08:17:00" (itu artinya 8 JAM 17 menit, bukan 8 menit 17 detik).

Transkrip di bawah berformat "[menit:detik] kalimat" APA ADANYA dari mesin
transkripsi — untuk video di ATAS satu jam, bagian "menit" BISA lebih dari
59 (mis. "[61:49]"), BUKAN berarti timestamp tidak valid. Konversi ke
HH:MM:SS dengan jam = menit // 60, menit_sisa = menit % 60, detik tetap.
Contoh: "[61:49]" pada transkrip berarti 1 jam 1 menit 49 detik, ditulis
"01:01:49" — BUKAN "00:61:49" (menit tidak valid) dan BUKAN "61:49:00".\
"""


class AnalyzeError(Exception):
    """Dilempar bila LLM gagal mengembalikan JSON valid setelah retry habis."""


def _save_failed_attempt(job_id: str, attempt: int, raw_json: str) -> None:
    path = get_settings().llm_attempt_path(job_id, attempt)
    try:
        path.write_text(raw_json)
    except OSError:
        logger.warning("job_id=%s gagal menyimpan raw LLM attempt %s ke disk", job_id, attempt)


def _build_client(settings: Settings) -> AsyncOpenAI:
    return AsyncOpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key)


def _build_user_prompt(
    video_duration_seconds: float,
    transcript_text: str,
    previous_error: str | None,
) -> str:
    duration_int = int(video_duration_seconds)
    parts = [
        f"Durasi total video: {duration_int} detik ({_format_hhmmss(duration_int)}). "
        "Timestamp klip TIDAK BOLEH melebihi ini.",
        "Transkrip (format [MM:SS] kalimat):",
        transcript_text,
    ]
    if previous_error is not None:
        parts.append(
            "Percobaan sebelumnya gagal divalidasi dengan error berikut, perbaiki:\n"
            f"{previous_error}"
        )
    return "\n\n".join(parts)


def _format_hhmmss(total_seconds: int) -> str:
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


async def _call_llm(client: AsyncOpenAI, model: str, system_prompt: str, user_prompt: str) -> str:
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
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
    system_prompt = _build_system_prompt(settings)
    transcript_text = transcript.to_prompt_text()

    last_error: str | None = None
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        user_prompt = _build_user_prompt(duration, transcript_text, last_error)
        raw_json: str | None = None
        try:
            raw_json = await _call_llm(client, settings.llm_model, system_prompt, user_prompt)
            plan = ClipPlan.model_validate_json(raw_json)
            plan.validate_within_video_duration(int(duration))
        except Exception as exc:
            logger.warning(
                "job_id=%s percobaan LLM %s/%s gagal: %s", job_id, attempt, _MAX_ATTEMPTS, exc
            )
            # Simpan raw respons yang gagal validasi supaya bisa didiagnosis
            # kalau retry habis tanpa hasil -- tanpa ini, isinya hilang begitu
            # saja begitu attempt berikutnya jalan (lihat kasus job_id nyata
            # yang gagal 3x tanpa jejak apa pun sampai kita menambahkan ini).
            if raw_json is not None:
                _save_failed_attempt(job_id, attempt, raw_json)
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
