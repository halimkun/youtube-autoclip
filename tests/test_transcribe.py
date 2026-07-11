"""Test untuk pipeline/transcribe.py: faster-whisper di-mock + format prompt LLM."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from autoclip.config import get_settings
from autoclip.pipeline.transcribe import (
    TranscribeError,
    Transcript,
    TranscriptSegment,
    TranscriptWord,
    _format_mmss,
    parse_vtt_captions,
    transcribe_audio,
)

_FIXTURES_DIR = Path(__file__).parent / "fixtures"


class _FakeWord:
    def __init__(self, word: str, start: float, end: float) -> None:
        self.word = word
        self.start = start
        self.end = end


class _FakeSegment:
    def __init__(self, start: float, end: float, text: str, words: list[_FakeWord]) -> None:
        self.start = start
        self.end = end
        self.text = text
        self.words = words


class _FakeWhisperModel:
    def __init__(self, model_name: str, **kwargs: object) -> None:
        self.model_name = model_name
        self.kwargs = kwargs

    def transcribe(self, audio_path: str, word_timestamps: bool = True):
        segments = [
            _FakeSegment(
                start=0.0,
                end=2.5,
                text=" Halo semua",
                words=[_FakeWord(" Halo", 0.0, 1.0), _FakeWord(" semua", 1.0, 2.5)],
            ),
            _FakeSegment(
                start=65.2,
                end=68.0,
                text=" ini menit kedua",
                words=[
                    _FakeWord(" ini", 65.2, 66.0),
                    _FakeWord(" menit", 66.0, 67.0),
                    _FakeWord(" kedua", 67.0, 68.0),
                ],
            ),
        ]
        info = SimpleNamespace(language="id", duration=70.0)
        return iter(segments), info


class _FailingWhisperModel:
    def __init__(self, model_name: str, **kwargs: object) -> None:
        pass

    def transcribe(self, audio_path: str, word_timestamps: bool = True):
        raise RuntimeError("model gagal load")


@pytest.fixture
def audio_file(tmp_path: Path) -> Path:
    path = tmp_path / "audio.m4a"
    path.write_bytes(b"fake audio bytes")
    return path


async def test_transcribe_audio_success(monkeypatch: pytest.MonkeyPatch, audio_file: Path) -> None:
    monkeypatch.setattr("autoclip.pipeline.transcribe.WhisperModel", _FakeWhisperModel)

    transcript = await transcribe_audio("job-1", audio_path=audio_file)

    assert transcript.language == "id"
    assert transcript.duration == 70.0
    assert len(transcript.segments) == 2
    assert transcript.segments[0].words[0].word == "Halo"

    output_path = get_settings().transcript_path("job-1")
    assert output_path.exists()
    persisted = Transcript.model_validate_json(output_path.read_text())
    assert persisted == transcript


async def test_transcribe_audio_skips_when_already_exists(
    monkeypatch: pytest.MonkeyPatch, audio_file: Path
) -> None:
    settings = get_settings()
    settings.ensure_job_dir("job-1")
    existing = Transcript(
        language="id",
        duration=5.0,
        segments=[
            TranscriptSegment(
                start=0.0,
                end=1.0,
                text="sudah ada",
                words=[TranscriptWord(word="x", start=0, end=1)],
            )
        ],
    )
    settings.transcript_path("job-1").write_text(existing.model_dump_json())

    def _boom(model_name: str) -> None:
        raise AssertionError("WhisperModel should not be called when transcript.json exists")

    monkeypatch.setattr("autoclip.pipeline.transcribe.WhisperModel", _boom)

    transcript = await transcribe_audio("job-1", audio_path=audio_file)

    assert transcript == existing


async def test_transcribe_audio_raises_when_source_missing() -> None:
    with pytest.raises(TranscribeError, match="tidak ditemukan"):
        await transcribe_audio("job-1", audio_path=Path("/does/not/exist.m4a"))


async def test_transcribe_audio_wraps_whisper_exception(
    monkeypatch: pytest.MonkeyPatch, audio_file: Path
) -> None:
    monkeypatch.setattr("autoclip.pipeline.transcribe.WhisperModel", _FailingWhisperModel)

    with pytest.raises(TranscribeError, match="faster-whisper gagal"):
        await transcribe_audio("job-1", audio_path=audio_file)


def test_format_mmss() -> None:
    assert _format_mmss(0) == "00:00"
    assert _format_mmss(65) == "01:05"
    assert _format_mmss(3661) == "61:01"


def test_to_prompt_text_uses_ringkas_format_without_word_level() -> None:
    transcript = Transcript(
        language="id",
        duration=70.0,
        segments=[
            TranscriptSegment(
                start=0.0,
                end=2.5,
                text=" Halo semua ",
                words=[TranscriptWord(word="Halo", start=0.0, end=1.0)],
            ),
            TranscriptSegment(
                start=65.2,
                end=68.0,
                text=" ini menit kedua",
                words=[TranscriptWord(word="ini", start=65.2, end=66.0)],
            ),
        ],
    )

    prompt_text = transcript.to_prompt_text()

    assert prompt_text == "[00:00] Halo semua\n[01:05] ini menit kedua"
    assert "0.0" not in prompt_text  # tidak ada timestamp per-kata


def test_parse_vtt_captions_extracts_words_without_duplication() -> None:
    transcript = parse_vtt_captions(_FIXTURES_DIR / "sample_auto_caption.vtt")

    assert transcript.language == "id"
    assert len(transcript.segments) == 2

    first, second = transcript.segments
    assert first.text == "Di video sila kali ini kita akan"
    assert second.text == "ngomongin kelemahan terbesar Presiden"

    all_words = [word.word for segment in transcript.segments for word in segment.words]
    assert all_words == [
        "Di",
        "video",
        "sila",
        "kali",
        "ini",
        "kita",
        "akan",
        "ngomongin",
        "kelemahan",
        "terbesar",
        "Presiden",
    ]

    # timing per-kata harus urut & tidak overlap
    first_word = first.words[0]
    assert first_word.word == "Di"
    assert first_word.start == pytest.approx(0.840)
    assert first_word.end == pytest.approx(1.120)

    last_word = second.words[-1]
    assert last_word.word == "Presiden"
    assert last_word.end == pytest.approx(6.269)
    assert transcript.duration == pytest.approx(6.269)


def test_parse_vtt_captions_raises_when_no_valid_cues(tmp_path: Path) -> None:
    bogus = tmp_path / "empty.vtt"
    bogus.write_text("WEBVTT\nKind: captions\nLanguage: id\n")

    with pytest.raises(ValueError, match="Tidak ada cue"):
        parse_vtt_captions(bogus)


async def test_transcribe_audio_uses_youtube_caption_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(model_name: str, **kwargs: object) -> None:
        raise AssertionError("WhisperModel should not be called when caption tersedia")

    monkeypatch.setattr("autoclip.pipeline.transcribe.WhisperModel", _boom)

    settings = get_settings()
    settings.ensure_job_dir("job-1")
    settings.subtitle_path("job-1").write_text(
        (_FIXTURES_DIR / "sample_auto_caption.vtt").read_text()
    )

    transcript = await transcribe_audio("job-1")

    assert transcript.language == "id"
    assert len(transcript.segments) == 2
    assert transcript.segments[0].text == "Di video sila kali ini kita akan"

    output_path = settings.transcript_path("job-1")
    assert output_path.exists()
    assert Transcript.model_validate_json(output_path.read_text()) == transcript


async def test_transcribe_audio_raises_immediately_when_no_caption_and_fallback_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(model_name: str, **kwargs: object) -> None:
        raise AssertionError("WhisperModel should not be called when fallback dimatikan")

    monkeypatch.setattr("autoclip.pipeline.transcribe.WhisperModel", _boom)
    monkeypatch.setenv("WHISPER_FALLBACK_ENABLED", "false")
    get_settings.cache_clear()

    try:
        with pytest.raises(TranscribeError, match="WHISPER_FALLBACK_ENABLED"):
            await transcribe_audio("job-1")
    finally:
        get_settings.cache_clear()


async def test_transcribe_audio_falls_back_to_whisper_when_caption_malformed(
    monkeypatch: pytest.MonkeyPatch, audio_file: Path
) -> None:
    settings = get_settings()
    settings.ensure_job_dir("job-1")
    settings.subtitle_path("job-1").write_text("bukan isi vtt yang valid sama sekali")

    monkeypatch.setattr("autoclip.pipeline.transcribe.WhisperModel", _FakeWhisperModel)

    transcript = await transcribe_audio("job-1", audio_path=audio_file)

    assert transcript.language == "id"
    assert transcript.duration == 70.0  # dari _FakeWhisperModel, bukan caption
