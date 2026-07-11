"""Test untuk pipeline/subtitles.py: generator .ass dari word-level timestamps."""

from __future__ import annotations

import pytest

from autoclip.config import get_settings
from autoclip.pipeline.subtitles import _format_ass_time, generate_ass, write_ass_file
from autoclip.pipeline.transcribe import Transcript, TranscriptSegment, TranscriptWord


def _word(text: str, start: float, end: float) -> TranscriptWord:
    return TranscriptWord(word=text, start=start, end=end)


@pytest.fixture
def transcript() -> Transcript:
    # 7 kata melintasi batas klip [100, 106); kata di luar rentang harus terbuang.
    words = [
        _word("sebelum", 95.0, 96.0),  # sebelum clip, harus dibuang
        _word("satu", 100.0, 100.5),
        _word("dua", 100.5, 101.0),
        _word("tiga", 101.0, 101.5),
        _word("empat", 101.5, 102.0),
        _word("lima", 102.0, 102.5),
        _word("enam", 102.5, 103.0),
        _word("tujuh", 103.0, 103.5),
        _word("sesudah", 110.0, 111.0),  # setelah clip, harus dibuang
    ]
    return Transcript(
        language="id",
        duration=200.0,
        segments=[TranscriptSegment(start=95.0, end=111.0, text="dummy", words=words)],
    )


def test_generate_ass_header_has_bottom_center_bold_style(transcript: Transcript) -> None:
    content = generate_ass(transcript, clip_start_seconds=100.0, clip_end_seconds=106.0)

    assert "PlayResX: 1080" in content
    assert "PlayResY: 1920" in content
    style_line = next(line for line in content.splitlines() if line.startswith("Style: Default"))
    fields = style_line.split(",")
    assert fields[7] == "-1"  # Bold
    assert fields[18] == "2"  # Alignment: bottom-center


def test_generate_ass_uses_custom_play_res(transcript: Transcript) -> None:
    # Format ORIGINAL (render.py) memberi resolusi asli video sumber, bukan
    # 1080x1920 default -- supaya subtitle tetap proporsional di video
    # yang rasio aspeknya beda dari 9:16.
    content = generate_ass(
        transcript, clip_start_seconds=100.0, clip_end_seconds=106.0, play_res=(1920, 1080)
    )

    assert "PlayResX: 1920" in content
    assert "PlayResY: 1080" in content
    assert "PlayResX: 1080" not in content


def test_generate_ass_excludes_words_outside_clip_range(transcript: Transcript) -> None:
    content = generate_ass(transcript, clip_start_seconds=100.0, clip_end_seconds=106.0)

    assert "sebelum" not in content
    assert "sesudah" not in content
    assert "satu" in content


def test_generate_ass_groups_words_by_words_per_line(transcript: Transcript) -> None:
    content = generate_ass(
        transcript, clip_start_seconds=100.0, clip_end_seconds=106.0, words_per_line=3
    )

    dialogue_lines = [line for line in content.splitlines() if line.startswith("Dialogue:")]
    assert len(dialogue_lines) == 3  # 7 kata dalam rentang -> grup 3,3,1
    assert dialogue_lines[0].endswith("satu dua tiga")
    assert dialogue_lines[1].endswith("empat lima enam")
    assert dialogue_lines[2].endswith("tujuh")


def test_generate_ass_rebases_timestamps_relative_to_clip_start(transcript: Transcript) -> None:
    content = generate_ass(
        transcript, clip_start_seconds=100.0, clip_end_seconds=106.0, words_per_line=3
    )

    first_dialogue = next(line for line in content.splitlines() if line.startswith("Dialogue:"))
    # kata pertama mulai di detik 100 absolut -> 0 detik relatif terhadap klip
    assert first_dialogue.split(",")[1] == "0:00:00.00"


def test_generate_ass_rejects_invalid_words_per_line(transcript: Transcript) -> None:
    with pytest.raises(ValueError, match="2-4"):
        generate_ass(transcript, clip_start_seconds=100.0, clip_end_seconds=106.0, words_per_line=1)
    with pytest.raises(ValueError, match="2-4"):
        generate_ass(transcript, clip_start_seconds=100.0, clip_end_seconds=106.0, words_per_line=5)


def test_generate_ass_escapes_curly_braces() -> None:
    transcript = Transcript(
        language="id",
        duration=10.0,
        segments=[
            TranscriptSegment(
                start=0.0,
                end=2.0,
                text="dummy",
                words=[_word("{halo}", 0.0, 1.0), _word("dunia", 1.0, 2.0)],
            )
        ],
    )

    content = generate_ass(transcript, clip_start_seconds=0.0, clip_end_seconds=2.0)

    assert "{halo}" not in content
    assert "(halo)" in content


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0, "0:00:00.00"),
        (75.34, "0:01:15.34"),
        (3661.5, "1:01:01.50"),
    ],
)
def test_format_ass_time(seconds: float, expected: str) -> None:
    assert _format_ass_time(seconds) == expected


def test_write_ass_file_persists_to_clips_dir(transcript: Transcript) -> None:
    output_path = write_ass_file(
        "job-1", 0, transcript, clip_start_seconds=100.0, clip_end_seconds=106.0
    )

    settings = get_settings()
    assert output_path == settings.clips_dir("job-1") / "clip-00.ass"
    assert output_path.exists()
    assert output_path.read_text(encoding="utf-8") == generate_ass(transcript, 100.0, 106.0)
