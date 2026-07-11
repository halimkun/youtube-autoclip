# CLAUDE.md — AutoClip Pipeline

Panduan untuk Claude Code saat mengerjakan repo ini. Baca `BRIEF.md` untuk spesifikasi lengkap sebelum menulis kode apa pun.

## Apa proyek ini

Pipeline Python yang menerima link YouTube via bot Telegram, memotong video panjang menjadi klip pendek vertikal 9:16 (dengan bantuan LLM via OpenRouter untuk memilih momen viral + metadata), lalu mengirim hasilnya kembali ke Telegram. Detail arsitektur, skema JSON, dan aturan validasi ada di `BRIEF.md` — jangan menyimpang dari kontrak JSON di sana.

## Stack (sudah diputuskan — jangan diganti)

- Python 3.11+, aiogram v3, yt-dlp, faster-whisper, FFmpeg (subprocess), OpenCV+YuNet, SQLite, Pydantic v2, pydantic-settings
- LLM: provider-agnostic via endpoint OpenAI-compatible — env `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL` (OpenRouter, HF Inference Providers, atau Ollama lokal). Jangan hardcode nama provider di kode
- JANGAN menambahkan framework web, Docker, Celery, Redis, atau Postgres di v1 kecuali diminta eksplisit

## Perintah

```bash
pip install -e ".[dev]"          # install
python -m autoclip.bot           # jalankan bot + worker
pytest                           # test
ruff check . && ruff format .    # lint + format
```

## Konvensi kode

- Type hints di semua fungsi publik; Pydantic untuk semua data lintas-tahap
- Async untuk bot & I/O jaringan; subprocess FFmpeg via `asyncio.create_subprocess_exec`
- FFmpeg dipanggil sebagai list argumen (bukan string shell) — hindari shell injection
- Semua path artefak melalui helper di `db.py`/`config.py`, format `data/media/<job_id>/...`
- Logging pakai modul `logging` dengan job_id di setiap baris log, bukan print
- Error tiap tahap pipeline dibungkus exception spesifik (`DownloadError`, `TranscribeError`, `AnalyzeError`, `RenderError`) agar worker bisa set status FAILED dengan pesan jelas
- Bahasa komentar/docstring: Indonesia atau Inggris, konsisten per file

## Aturan penting yang sering dilanggar — jangan

1. Jangan kirim transkrip penuh word-level ke LLM — kirim format ringkas `[MM:SS] kalimat`
2. Jangan pakai stream copy (`-c copy`) untuk cutting — harus re-encode agar frame-accurate
3. Jangan hapus artefak intermediate saat job FAILED — dibutuhkan untuk retry resumable
4. Jangan hardcode API key, model, chat ID, atau durasi klip — semua dari env
5. Jangan percaya output LLM mentah — selalu lewat validasi Pydantic + aturan di BRIEF.md bagian 4
6. Jangan proses job untuk user di luar whitelist `TELEGRAM_ALLOWED_USERS`

## Urutan pengerjaan yang disarankan (kalau mulai dari nol)

1. `config.py` + `.env.example` + `models.py` (kontrak data dulu)
2. `db.py` (tabel jobs + transisi status)
3. `pipeline/download.py` → `transcribe.py` (bisa dites terpisah via CLI kecil)
4. `pipeline/analyze.py` (mock response LLM di test)
5. `pipeline/subtitles.py` + `facecrop.py` + `render.py`
6. `worker.py` (orkestrasi + retry per tahap)
7. `bot.py` + `telegram_sender.py`
8. Integration test end-to-end dengan video pendek

## Testing

- Unit test untuk: validasi ClipPlan (kasus timestamp invalid, overlap, durasi di luar batas), parser timestamp, generator .ass
- Pipeline test pakai video sampel pendek (≤2 menit) di `tests/fixtures/` — jangan commit file video besar
- Mock OpenRouter dan Telegram API di test; jangan pernah memanggil API asli dari pytest

## Environment (.env)

```
TELEGRAM_BOT_TOKEN=
TELEGRAM_ALLOWED_USERS=123456789,987654321
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_API_KEY=
LLM_MODEL=
CLIP_MIN_SECONDS=30
CLIP_MAX_SECONDS=180
MAX_CLIPS_PER_VIDEO=8
WHISPER_MODEL=small
WHISPER_FALLBACK_ENABLED=true
DATA_DIR=./data
```
