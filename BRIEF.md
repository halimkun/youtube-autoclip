# Project Brief — AutoClip Pipeline (Telegram-Integrated)

## 1. Ringkasan

Sistem otomatis yang menerima **link YouTube via bot Telegram**, memproses video panjang (podcast/talking head) menjadi **beberapa klip pendek vertikal 9:16** siap upload ke YouTube Shorts/TikTok/Reels, lalu **mengirimkan hasil klip + metadata kembali ke Telegram**.

LLM (via OpenRouter) hanya berperan sebagai "otak": membaca transkrip dan menentukan rentang waktu yang berpotensi viral beserta judul, deskripsi, dan tags. Seluruh kerja berat (download, transkripsi, cutting, cropping, subtitle) dilakukan lokal oleh Python + FFmpeg. Strategi ini menghemat token secara drastis karena LLM tidak pernah menyentuh file video.

## 2. Alur Sistem (End-to-End)

```
User kirim link YouTube ke bot Telegram
        │
        ▼
[1] Bot validasi link → buat job di database (status: QUEUED)
        │
        ▼
[2] Worker ambil job → download video+audio (yt-dlp)          status: DOWNLOADING
        │
        ▼
[3] Transkripsi audio (faster-whisper, word_timestamps=True)  status: TRANSCRIBING
        │
        ▼
[4] Kirim transkrip ke LLM via OpenRouter → JSON klip         status: ANALYZING
        │
        ▼
[5] Validasi JSON (Pydantic) → parsing timestamp              status: ANALYZING
        │
        ▼
[6] Per klip: cut → face-crop 9:16 → burn subtitle (FFmpeg)   status: RENDERING
        │
        ▼
[7] Kirim tiap klip ke Telegram + caption berisi metadata     status: DONE
```

Setiap tahap update status job. Jika gagal, status = FAILED dengan pesan error, dan artefak tahap sebelumnya (video mentah, transkrip, JSON LLM) TIDAK dihapus agar bisa retry dari tahap yang gagal tanpa mengulang dari awal.

## 3. Stack Teknologi

| Komponen | Pilihan | Alasan |
|---|---|---|
| Bahasa | Python 3.11+ | Ekosistem media processing terlengkap |
| Bot Telegram | aiogram v3 | Async-native, cocok dengan long-running job |
| Downloader | yt-dlp | Standar de facto, aktif di-maintain |
| Transkripsi | faster-whisper | Word-level timestamps, jalan di CPU/GPU lokal |
| LLM | Provider apa pun dengan endpoint OpenAI-compatible | OpenRouter, HF Inference Providers, atau Ollama lokal — pindah cukup ganti env |
| Video processing | FFmpeg (via subprocess) | Kontrol penuh, cepat |
| Face detection | OpenCV + YuNet (atau MediaPipe) | Ringan, gratis, cukup untuk talking head |
| Database/queue | SQLite + tabel jobs | Sederhana, cukup untuk semi-produksi single-machine |
| Config | .env + pydantic-settings | Rapi dan type-safe |
| Validasi | Pydantic v2 | Validasi output LLM & config |

Catatan: JANGAN pakai NestJS/Node untuk pipeline ini. Semua tooling inti native Python.

## 4. Skema JSON Output LLM (KONTRAK PENTING)

LLM WAJIB mengembalikan JSON persis dengan struktur ini (tanpa markdown fence, tanpa teks tambahan):

```json
{
  "video_summary": "Ringkasan 1-2 kalimat isi video",
  "clips": [
    {
      "start": "00:04:12",
      "end": "00:05:28",
      "hook_reason": "Alasan singkat kenapa momen ini berpotensi viral",
      "virality_score": 8,
      "title": "Judul hook maksimal 100 karakter",
      "description": "Deskripsi 2-4 kalimat, boleh mengandung CTA",
      "tags": ["tag1", "tag2", "tag3"],
      "hashtags": ["#shorts", "#podcast"]
    }
  ]
}
```

Aturan validasi (implementasikan dengan Pydantic):
- `start`/`end` format `HH:MM:SS`, `end > start`
- Durasi klip: min 30 detik, max 180 detik (configurable via env)
- `end` tidak boleh melebihi durasi video sumber
- `virality_score` integer 1-10
- Maksimal 8 klip per video (configurable)
- Klip tidak boleh saling overlap lebih dari 10 detik
- Jika JSON tidak valid → retry request LLM maksimal 2x dengan pesan error disertakan, lalu FAILED

## 5. Detail Tahap Kritis

### 5.1 Transkripsi
- Ekstrak audio saja untuk whisper (`yt-dlp -x` atau ffmpeg dari file video)
- `faster-whisper` model `small`/`medium`, `word_timestamps=True`
- Simpan dua format: (a) JSON word-level untuk subtitle, (b) teks ringkas `[MM:SS] kalimat` untuk prompt LLM

### 5.2 Prompt LLM
- System prompt: peran "viral clip analyst" untuk konten podcast Indonesia
- Input: transkrip ringkas + durasi total video
- Minta `hook_reason` + `virality_score` SEBELUM field lain (memaksa reasoning)
- Gunakan `response_format: json_object` jika model mendukung; jika tidak, andalkan validator + retry
- Provider LLM diatur via env generik: `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`. Contoh nilai:
  - OpenRouter: `https://openrouter.ai/api/v1`
  - HF Inference Providers: `https://router.huggingface.co/v1`
  - Ollama lokal: `http://localhost:11434/v1` (api key boleh dummy)
- Kode `analyze.py` TIDAK boleh mengandung nama provider secara hardcoded

### 5.3 Rendering (FFmpeg)
Per klip, urutan operasi:
1. **Cut**: `-ss <start> -to <end>` dengan re-encode (bukan stream copy, agar frame-accurate)
2. **Crop 9:16**: deteksi wajah di beberapa frame sampel dari rentang klip → rata-rata posisi X wajah → crop window statis `ih*9/16 : ih` berpusat di wajah → scale ke 1080x1920
3. **Subtitle**: generate file `.ass` dari word-level timestamps (2-4 kata per baris, posisi tengah-bawah, font besar bold dengan outline) → burn via `-vf ass=`
4. Output: H.264, CRF 21, preset veryfast, AAC 128k, `<slug-judul>-<n>.mp4`

### 5.4 Integrasi Telegram
- Perintah bot: kirim link YouTube langsung (tanpa command), `/status <job_id>`, `/cancel <job_id>`
- Saat job diterima: reply konfirmasi + job_id
- Update progress per perubahan status (edit message, bukan spam message baru)
- Hasil akhir: kirim tiap klip sebagai video + caption berisi judul, deskripsi, hashtags (siap copy-paste)
- Batas ukuran upload Bot API: 50 MB per video — jika lebih, kompres ulang atau kirim sebagai dokumen; klip <3 menit 1080x1920 CRF 21 umumnya aman
- Whitelist user ID (env `TELEGRAM_ALLOWED_USERS`) — bot ini bukan untuk publik

## 6. Struktur Proyek

```
autoclip/
├── CLAUDE.md
├── BRIEF.md
├── .env.example
├── pyproject.toml
├── autoclip/
│   ├── __init__.py
│   ├── config.py          # pydantic-settings
│   ├── bot.py             # aiogram handlers
│   ├── models.py          # Pydantic: ClipPlan, Clip, Job
│   ├── db.py              # SQLite job queue
│   ├── worker.py          # loop pengambil job, orkestrasi pipeline
│   ├── pipeline/
│   │   ├── download.py    # yt-dlp
│   │   ├── transcribe.py  # faster-whisper
│   │   ├── analyze.py     # OpenRouter call + retry + validasi
│   │   ├── subtitles.py   # word timestamps → .ass
│   │   ├── facecrop.py    # deteksi wajah → parameter crop
│   │   └── render.py      # ffmpeg cut/crop/burn
│   └── telegram_sender.py # kirim hasil ke chat
├── data/
│   ├── jobs.db
│   └── media/<job_id>/    # raw.mp4, audio.m4a, transcript.json, plan.json, clips/
└── tests/
```

## 7. Prinsip Desain

1. **LLM tidak pernah menyentuh video** — hanya teks masuk, JSON keluar
2. **Setiap tahap idempotent & resumable** — artefak disimpan per job, retry tidak mengulang tahap yang sudah sukses
3. **Fail loudly** — error dikirim ke Telegram dengan konteks, bukan silent fail
4. **Config lewat env** — tidak ada nilai hardcoded (model, durasi klip, jumlah klip, path)
5. **Satu worker dulu** — proses job serial; concurrency bisa ditambah nanti kalau perlu

## 8. Di Luar Scope (v1)

- Auto-upload ke YouTube/TikTok (v2)
- Split layout 2 pembicara (v2)
- Web dashboard (v2)
- Multi-worker / distributed queue (v2)

## 9. Catatan Legal

Sistem hanya untuk memproses video milik sendiri atau yang berizin. Memotong dan re-upload konten pihak ketiga tanpa lisensi berisiko copyright strike.
