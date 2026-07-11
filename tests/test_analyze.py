"""Test untuk pipeline/analyze.py: LLM (OpenAI-compatible client) di-mock + retry."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from autoclip.config import get_settings
from autoclip.models import ClipPlan
from autoclip.pipeline.analyze import AnalyzeError, analyze_transcript
from autoclip.pipeline.transcribe import Transcript, TranscriptSegment, TranscriptWord


def _valid_plan_json(**overrides: Any) -> str:
    data: dict[str, Any] = {
        "video_summary": "Podcast tentang produktivitas.",
        "clips": [
            {
                "start": "00:04:12",
                "end": "00:05:28",
                "hook_reason": "momen paling lucu di video",
                "virality_score": 8,
                "title": "Judul hook",
                "description": "Deskripsi singkat.",
                "tags": ["tag1", "tag2"],
                "hashtags": ["#shorts"],
            }
        ],
    }
    data.update(overrides)
    return json.dumps(data)


def _make_fake_openai(responses: list[str | Exception]) -> tuple[type, list[dict]]:
    """Bikin kelas pengganti AsyncOpenAI yang membalas `responses` berurutan."""
    calls: list[dict] = []

    class _FakeCompletions:
        def __init__(self) -> None:
            self._responses = iter(responses)

        async def create(self, **kwargs: Any) -> SimpleNamespace:
            calls.append(kwargs)
            content = next(self._responses)
            if isinstance(content, Exception):
                raise content
            message = SimpleNamespace(content=content)
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    class _FakeChat:
        def __init__(self) -> None:
            self.completions = _FakeCompletions()

    class _FakeAsyncOpenAI:
        def __init__(self, base_url: str | None = None, api_key: str | None = None) -> None:
            self.base_url = base_url
            self.api_key = api_key
            self.chat = _FakeChat()

    return _FakeAsyncOpenAI, calls


@pytest.fixture
def transcript() -> Transcript:
    return Transcript(
        language="id",
        duration=600.0,
        segments=[
            TranscriptSegment(
                start=252.0,
                end=328.0,
                text="ini bagian lucu",
                words=[TranscriptWord(word="ini", start=252.0, end=252.5)],
            )
        ],
    )


async def test_analyze_transcript_success_first_try(
    monkeypatch: pytest.MonkeyPatch, transcript: Transcript
) -> None:
    fake_cls, calls = _make_fake_openai([_valid_plan_json()])
    monkeypatch.setattr("autoclip.pipeline.analyze.AsyncOpenAI", fake_cls)

    plan = await analyze_transcript("job-1", transcript)

    assert isinstance(plan, ClipPlan)
    assert plan.clips[0].title == "Judul hook"
    assert len(calls) == 1
    assert calls[0]["response_format"] == {"type": "json_object"}
    # max_tokens harus di-set eksplisit — beberapa model (mis. Gemini 2.5
    # Flash) memakai reasoning tokens tersembunyi yang bisa memotong output
    # JSON sebelum sempat ditulis kalau limitnya kecil/default provider.
    assert calls[0]["max_tokens"] >= 4000

    output_path = get_settings().plan_path("job-1")
    assert output_path.exists()
    assert ClipPlan.model_validate_json(output_path.read_text()) == plan


async def test_analyze_transcript_skips_when_plan_already_exists(
    monkeypatch: pytest.MonkeyPatch, transcript: Transcript
) -> None:
    settings = get_settings()
    settings.ensure_job_dir("job-1")
    existing = ClipPlan.model_validate_json(_valid_plan_json())
    settings.plan_path("job-1").write_text(existing.model_dump_json())

    def _boom(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("AsyncOpenAI should not be constructed when plan.json exists")

    monkeypatch.setattr("autoclip.pipeline.analyze.AsyncOpenAI", _boom)

    plan = await analyze_transcript("job-1", transcript)

    assert plan == existing


async def test_analyze_transcript_retries_on_invalid_json(
    monkeypatch: pytest.MonkeyPatch, transcript: Transcript
) -> None:
    fake_cls, calls = _make_fake_openai(["bukan json valid", _valid_plan_json()])
    monkeypatch.setattr("autoclip.pipeline.analyze.AsyncOpenAI", fake_cls)

    plan = await analyze_transcript("job-1", transcript)

    assert plan.clips[0].title == "Judul hook"
    assert len(calls) == 2
    # Percobaan kedua harus menyertakan pesan error dari percobaan pertama.
    assert "Percobaan sebelumnya gagal" in calls[1]["messages"][1]["content"]


async def test_analyze_transcript_retries_when_clip_exceeds_video_duration(
    monkeypatch: pytest.MonkeyPatch, transcript: Transcript
) -> None:
    too_long_clip = _valid_plan_json(
        clips=[
            {
                "start": "00:09:50",
                "end": "00:10:30",  # melebihi durasi video 600s (10 menit)
                "hook_reason": "x",
                "virality_score": 5,
                "title": "t",
                "description": "d",
                "tags": ["a"],
                "hashtags": ["#a"],
            }
        ]
    )
    fake_cls, calls = _make_fake_openai([too_long_clip, _valid_plan_json()])
    monkeypatch.setattr("autoclip.pipeline.analyze.AsyncOpenAI", fake_cls)

    plan = await analyze_transcript("job-1", transcript, video_duration_seconds=600.0)

    assert plan.clips[0].title == "Judul hook"
    assert len(calls) == 2


async def test_analyze_transcript_retries_on_api_error(
    monkeypatch: pytest.MonkeyPatch, transcript: Transcript
) -> None:
    fake_cls, calls = _make_fake_openai([RuntimeError("rate limited"), _valid_plan_json()])
    monkeypatch.setattr("autoclip.pipeline.analyze.AsyncOpenAI", fake_cls)

    plan = await analyze_transcript("job-1", transcript)

    assert plan.clips[0].title == "Judul hook"
    assert len(calls) == 2


async def test_analyze_transcript_raises_after_max_attempts(
    monkeypatch: pytest.MonkeyPatch, transcript: Transcript
) -> None:
    fake_cls, calls = _make_fake_openai(["invalid"] * 3)
    monkeypatch.setattr("autoclip.pipeline.analyze.AsyncOpenAI", fake_cls)

    with pytest.raises(AnalyzeError, match="3 percobaan"):
        await analyze_transcript("job-1", transcript)

    assert len(calls) == 3


async def test_analyze_transcript_tells_llm_the_actual_duration_and_count_limits(
    monkeypatch: pytest.MonkeyPatch, transcript: Transcript
) -> None:
    # Bug nyata: LLM tidak pernah diberi tahu batas durasi/jumlah klip secara
    # eksplisit, cuma menebak "pendek" tanpa angka -- menghasilkan klip di
    # luar batas berulang kali pada video panjang. Prompt sistem harus
    # menyebutkan angka [CLIP_MIN_SECONDS, CLIP_MAX_SECONDS, MAX_CLIPS_PER_VIDEO]
    # yang benar-benar dikonfigurasi lewat env, bukan hardcode.
    fake_cls, calls = _make_fake_openai([_valid_plan_json()])
    monkeypatch.setattr("autoclip.pipeline.analyze.AsyncOpenAI", fake_cls)

    await analyze_transcript("job-1", transcript)

    settings = get_settings()
    system_prompt = calls[0]["messages"][0]["content"]
    assert str(settings.clip_min_seconds) in system_prompt
    assert str(settings.clip_max_seconds) in system_prompt
    assert str(settings.max_clips_per_video) in system_prompt


async def test_analyze_transcript_saves_failed_attempts_for_debugging(
    monkeypatch: pytest.MonkeyPatch, transcript: Transcript
) -> None:
    # Kasus nyata: job gagal 3x lalu FAILED tanpa jejak apa pun soal ISI
    # respons LLM yang gagal divalidasi -- cuma pesan error Pydantic yang
    # tersimpan di log/db, raw JSON-nya sendiri hilang begitu saja. Setiap
    # percobaan yang gagal validasi harus ditulis ke disk untuk debugging.
    invalid_but_parseable = _valid_plan_json(clips=[])  # clips kosong -> gagal validasi
    fake_cls, calls = _make_fake_openai(
        [invalid_but_parseable, "bukan json valid sama sekali", invalid_but_parseable]
    )
    monkeypatch.setattr("autoclip.pipeline.analyze.AsyncOpenAI", fake_cls)

    with pytest.raises(AnalyzeError):
        await analyze_transcript("job-1", transcript)

    assert len(calls) == 3
    settings = get_settings()
    assert settings.llm_attempt_path("job-1", 1).read_text() == invalid_but_parseable
    assert settings.llm_attempt_path("job-1", 3).read_text() == invalid_but_parseable
    # Percobaan 2: JSON-nya sendiri tidak valid sama sekali (bukan cuma gagal
    # validasi skema) -- tetap harus tersimpan apa adanya untuk diagnosis.
    assert settings.llm_attempt_path("job-1", 2).read_text() == "bukan json valid sama sekali"


async def test_analyze_transcript_does_not_save_attempt_when_api_call_itself_fails(
    monkeypatch: pytest.MonkeyPatch, transcript: Transcript
) -> None:
    # Kalau _call_llm sendiri yang raise (mis. rate limit, konten kosong),
    # tidak ada raw_json sama sekali untuk disimpan -- pastikan tidak ada
    # file kosong/salah yang tertulis untuk percobaan itu.
    fake_cls, calls = _make_fake_openai([RuntimeError("rate limited"), _valid_plan_json()])
    monkeypatch.setattr("autoclip.pipeline.analyze.AsyncOpenAI", fake_cls)

    await analyze_transcript("job-1", transcript)

    assert len(calls) == 2
    assert not get_settings().llm_attempt_path("job-1", 1).exists()


async def test_analyze_transcript_uses_settings_for_client(
    monkeypatch: pytest.MonkeyPatch, transcript: Transcript
) -> None:
    fake_cls, _calls = _make_fake_openai([_valid_plan_json()])
    monkeypatch.setattr("autoclip.pipeline.analyze.AsyncOpenAI", fake_cls)

    instances: list[Any] = []
    original_init = fake_cls.__init__

    def _tracking_init(self: Any, **kwargs: Any) -> None:
        original_init(self, **kwargs)
        instances.append(self)

    monkeypatch.setattr(fake_cls, "__init__", _tracking_init)

    await analyze_transcript("job-1", transcript)

    settings = get_settings()
    assert instances[0].base_url == settings.llm_base_url
    assert instances[0].api_key == settings.llm_api_key
