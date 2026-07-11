"""Transkripsi audio dengan faster-whisper (word-level timestamps).

Menghasilkan `Transcript` (word-level, disimpan sebagai `transcript.json`
untuk kebutuhan generate subtitle) yang juga bisa diformat ringkas
`[MM:SS] kalimat` via `Transcript.to_prompt_text()` untuk dikirim ke LLM
(lihat BRIEF.md bagian 5.1). Jangan pernah kirim transkrip word-level
penuh ke LLM — hanya format ringkas ini.

Kalau caption YouTube sudah terunduh (`download._download_subtitles_best_effort`
-> `data/media/<job_id>/raw.id.vtt`), dipakai langsung via `parse_vtt_captions`
alih-alih menjalankan whisper (jauh lebih cepat untuk video panjang). Kalau
tidak ada dan `WHISPER_FALLBACK_ENABLED=false`, job langsung FAILED di sini
tanpa menunggu whisper. Melempar `TranscribeError` bila keduanya gagal.
"""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path

from faster_whisper import WhisperModel
from pydantic import BaseModel

from autoclip.config import get_settings

logger = logging.getLogger(__name__)


class TranscribeError(Exception):
    """Dilempar bila proses transkripsi faster-whisper gagal."""


class TranscriptWord(BaseModel):
    word: str
    start: float
    end: float


class TranscriptSegment(BaseModel):
    start: float
    end: float
    text: str
    words: list[TranscriptWord]


class Transcript(BaseModel):
    language: str
    duration: float
    segments: list[TranscriptSegment]

    def to_prompt_text(self) -> str:
        """Format ringkas `[MM:SS] kalimat` per segmen, untuk dikirim ke LLM.

        Word-level timestamps sengaja tidak diikutkan (lihat CLAUDE.md
        aturan #1) — hanya dipakai internal oleh subtitles.py.
        """
        lines = [
            f"[{_format_mmss(segment.start)}] {segment.text.strip()}" for segment in self.segments
        ]
        return "\n".join(lines)


def _format_mmss(seconds: float) -> str:
    total_seconds = int(seconds)
    minutes, secs = divmod(total_seconds, 60)
    return f"{minutes:02d}:{secs:02d}"


_VTT_TIMESTAMP_RE = re.compile(r"(\d\d:\d\d:\d\d\.\d\d\d)\s*-->\s*(\d\d:\d\d:\d\d\.\d\d\d)")
_VTT_WORD_TAG_RE = re.compile(r"<(\d\d:\d\d:\d\d\.\d\d\d)><c>([^<]*)</c>")


def _parse_vtt_timestamp(value: str) -> float:
    hours, minutes, rest = value.split(":")
    seconds, millis = rest.split(".")
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(millis) / 1000


def parse_vtt_captions(vtt_path: Path) -> Transcript:
    """Parse caption auto-generate YouTube ("rolling caption" VTT) jadi `Transcript`.

    Format ini menampilkan tiap baris secara progresif lewat beberapa cue
    berurutan: satu cue "tumbuh" dengan tag per-kata berwaktu
    `<HH:MM:SS.mmm><c> kata</c>`, diikuti cue "freeze" berdurasi nyaris nol
    yang mengulang baris itu sebagai teks polos (duplikat, sudah tertangkap
    cue sebelumnya). Cue tanpa tag dilewati; hanya baris TERAKHIR tiap cue
    yang diproses (baris sebelumnya adalah konteks yang sudah terungkap dari
    cue lampau).

    Raises `ValueError` kalau tidak ada satupun cue valid yang bisa
    diekstrak — caller (`transcribe_audio`) treat ini sebagai "caption tidak
    bisa dipakai" dan fallback ke whisper (kalau diizinkan).
    """
    # Cari cue dengan menyisir baris timestamp langsung (bukan split blok pakai
    # baris kosong) — lebih tahan terhadap variasi baris "kosong"/spasi-tunggal
    # yang dipakai YouTube sebagai placeholder baris sebelumnya.
    raw_lines = vtt_path.read_text(encoding="utf-8").replace("\r\n", "\n").split("\n")

    cues: list[tuple[float, float, list[str]]] = []
    current_cue: tuple[float, float, list[str]] | None = None
    for line in raw_lines:
        timestamp_match = _VTT_TIMESTAMP_RE.search(line)
        if timestamp_match is not None:
            current_cue = (
                _parse_vtt_timestamp(timestamp_match.group(1)),
                _parse_vtt_timestamp(timestamp_match.group(2)),
                [],
            )
            cues.append(current_cue)
        elif current_cue is not None and line.strip():
            current_cue[2].append(line)

    segments: list[TranscriptSegment] = []
    for cue_start, cue_end, text_lines in cues:
        if not text_lines or "<" not in text_lines[-1]:
            continue  # cue tanpa tag = duplikat "freeze", lewati

        last_line = text_lines[-1]
        prefix = last_line.split("<", 1)[0]
        tag_matches = list(_VTT_WORD_TAG_RE.finditer(last_line))

        words: list[TranscriptWord] = []
        first_tag_time = _parse_vtt_timestamp(tag_matches[0].group(1)) if tag_matches else cue_end
        words.extend(
            TranscriptWord(word=token, start=cue_start, end=first_tag_time)
            for token in prefix.split()
        )
        for i, tag_match in enumerate(tag_matches):
            word_text = tag_match.group(2).strip()
            if not word_text:
                continue
            word_start = _parse_vtt_timestamp(tag_match.group(1))
            word_end = (
                _parse_vtt_timestamp(tag_matches[i + 1].group(1))
                if i + 1 < len(tag_matches)
                else cue_end
            )
            words.append(TranscriptWord(word=word_text, start=word_start, end=word_end))

        if not words:
            continue

        segments.append(
            TranscriptSegment(
                start=words[0].start,
                end=words[-1].end,
                text=" ".join(word.word for word in words),
                words=words,
            )
        )

    if not segments:
        raise ValueError(f"Tidak ada cue caption yang bisa di-parse dari {vtt_path}")

    return Transcript(language="id", duration=segments[-1].end, segments=segments)


async def transcribe_audio(job_id: str, audio_path: Path | None = None) -> Transcript:
    """Transkripsi audio: pakai caption YouTube kalau ada (jauh lebih cepat),
    else faster-whisper (word_timestamps=True).

    Idempotent: kalau `transcript.json` sudah ada, dibaca ulang tanpa
    transkripsi ulang (baik via caption maupun whisper).
    """
    settings = get_settings()
    settings.ensure_job_dir(job_id)
    output_path = settings.transcript_path(job_id)

    if output_path.exists():
        logger.info("job_id=%s transcript sudah ada, skip transkripsi", job_id)
        return Transcript.model_validate_json(output_path.read_text())

    subtitle_path = settings.subtitle_path(job_id)
    if subtitle_path.exists():
        try:
            transcript = parse_vtt_captions(subtitle_path)
        except Exception as exc:
            logger.warning(
                "job_id=%s caption YouTube di %s tidak bisa di-parse (%s), fallback ke whisper",
                job_id,
                subtitle_path,
                exc,
            )
        else:
            logger.info(
                "job_id=%s pakai caption YouTube -> %s, skip whisper", job_id, subtitle_path
            )
            _write_transcript(output_path, transcript)
            return transcript

    if not settings.whisper_fallback_enabled:
        raise TranscribeError(
            f"Caption YouTube tidak ditemukan/tidak valid di {subtitle_path} dan "
            "WHISPER_FALLBACK_ENABLED=false — tidak transkripsi."
        )

    source = audio_path if audio_path is not None else settings.audio_path(job_id)
    if not source.exists():
        raise TranscribeError(f"Audio sumber tidak ditemukan untuk transkripsi: {source}")

    logger.info("job_id=%s mulai transkripsi %s", job_id, source)
    try:
        transcript = await asyncio.to_thread(_run_whisper, source, settings.whisper_model)
    except Exception as exc:
        raise TranscribeError(f"faster-whisper gagal mentranskripsi {source}: {exc}") from exc

    _write_transcript(output_path, transcript)
    logger.info("job_id=%s transkripsi selesai -> %s", job_id, output_path)
    return transcript


def _write_transcript(output_path: Path, transcript: Transcript) -> None:
    """Simpan `transcript.json`. Kegagalan tulis (mis. direktori job hilang
    di tengah jalan) dibungkus jadi `TranscribeError`, bukan crash proses
    worker tanpa tertangani — supaya job ini saja yang FAILED, bukan seluruh
    worker (lihat CLAUDE.md: tiap tahap harus dibungkus exception spesifik).
    """
    try:
        output_path.write_text(transcript.model_dump_json(indent=2))
    except OSError as exc:
        raise TranscribeError(f"Gagal menyimpan transcript.json ke {output_path}: {exc}") from exc


def _run_whisper(audio_path: Path, model_name: str) -> Transcript:
    """Panggil faster-whisper secara sinkron (dijalankan di thread terpisah)."""
    model = WhisperModel(model_name, compute_type="int8", cpu_threads=8)
    segments_iter, info = model.transcribe(str(audio_path), word_timestamps=True)

    segments = [
        TranscriptSegment(
            start=segment.start,
            end=segment.end,
            text=segment.text,
            words=[
                TranscriptWord(word=word.word.strip(), start=word.start, end=word.end)
                for word in (segment.words or [])
            ],
        )
        for segment in segments_iter
    ]

    return Transcript(language=info.language, duration=info.duration, segments=segments)


if __name__ == "__main__":
    import sys

    async def _main() -> None:
        if len(sys.argv) != 3:
            print("Pemakaian: python -m autoclip.pipeline.transcribe <job_id> <audio_path>")
            raise SystemExit(1)
        job_id, audio_path = sys.argv[1], Path(sys.argv[2])

        transcript = await transcribe_audio(job_id, audio_path=audio_path)
        print(f"language={transcript.language} duration={transcript.duration:.1f}s")
        print(f"segments={len(transcript.segments)}")
        print("--- prompt text (10 baris pertama) ---")
        print("\n".join(transcript.to_prompt_text().splitlines()[:10]))

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(_main())
