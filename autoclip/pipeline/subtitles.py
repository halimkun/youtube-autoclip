"""Generator file subtitle .ass dari word-level timestamps.

Mengubah word-level transcript (dari `transcribe.py`) menjadi file
`.ass` per klip: 2-4 kata per baris, posisi tengah-bawah, font besar
bold dengan outline (lihat BRIEF.md bagian 5.3 poin 3), untuk kemudian
di-burn oleh render.py via filter `ass=`. Timestamp di-rebase relatif
ke awal klip karena klip hasil cut punya timeline sendiri mulai dari 0.
"""

from __future__ import annotations

from pathlib import Path

from autoclip.config import get_settings
from autoclip.pipeline.transcribe import Transcript, TranscriptWord

_HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, \
BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, \
BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,90,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,-1,0,0,0,\
100,100,0,0,1,6,0,2,60,60,120,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def generate_ass(
    transcript: Transcript,
    clip_start_seconds: float,
    clip_end_seconds: float,
    words_per_line: int = 3,
) -> str:
    """Bangun isi file `.ass` untuk satu klip dari word-level transcript."""
    if not 2 <= words_per_line <= 4:
        raise ValueError("words_per_line harus 2-4 sesuai BRIEF.md bagian 5.3")

    clip_words = _words_in_range(transcript, clip_start_seconds, clip_end_seconds)
    dialogue_lines = "".join(
        _dialogue_line(group, clip_start_seconds, clip_end_seconds)
        for group in _chunked(clip_words, words_per_line)
    )
    return _HEADER + dialogue_lines


def write_ass_file(
    job_id: str,
    clip_index: int,
    transcript: Transcript,
    clip_start_seconds: float,
    clip_end_seconds: float,
    words_per_line: int = 3,
) -> Path:
    """Generate lalu simpan file `.ass` untuk satu klip ke `data/media/<job_id>/clips/`."""
    settings = get_settings()
    settings.ensure_job_dir(job_id)
    output_path = settings.clips_dir(job_id) / f"clip-{clip_index:02d}.ass"
    content = generate_ass(transcript, clip_start_seconds, clip_end_seconds, words_per_line)
    output_path.write_text(content, encoding="utf-8")
    return output_path


def _words_in_range(transcript: Transcript, start: float, end: float) -> list[TranscriptWord]:
    return [
        word
        for segment in transcript.segments
        for word in segment.words
        if start <= word.start < end
    ]


def _chunked(words: list[TranscriptWord], size: int) -> list[list[TranscriptWord]]:
    return [words[i : i + size] for i in range(0, len(words), size)]


def _dialogue_line(words: list[TranscriptWord], clip_start: float, clip_end: float) -> str:
    clip_duration = clip_end - clip_start
    rel_start = max(0.0, words[0].start - clip_start)
    rel_end = min(clip_duration, words[-1].end - clip_start)
    text = _escape_ass_text(" ".join(word.word.strip() for word in words))
    return (
        f"Dialogue: 0,{_format_ass_time(rel_start)},{_format_ass_time(rel_end)},"
        f"Default,,0,0,0,,{text}\n"
    )


def _escape_ass_text(text: str) -> str:
    return text.replace("{", "(").replace("}", ")")


def _format_ass_time(seconds: float) -> str:
    total_centiseconds = round(seconds * 100)
    hours, remainder = divmod(total_centiseconds, 360000)
    minutes, remainder = divmod(remainder, 6000)
    secs, centis = divmod(remainder, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{centis:02d}"
