# AutoClip

Pipeline Python yang menerima link YouTube lewat bot Telegram, memotong video panjang (podcast/talking head) menjadi beberapa klip pendek siap upload ke YouTube Shorts/TikTok/Reels, lalu mengirim hasilnya kembali ke Telegram lengkap dengan judul, deskripsi, tags, dan hashtag.

LLM (via endpoint OpenAI-compatible apa pun — OpenRouter, HF Inference Providers, Ollama lokal, dll.) cuma berperan sebagai "otak": membaca transkrip ringkas dan menentukan rentang waktu mana yang berpotensi viral, beserta metadata-nya. Seluruh kerja berat (download, transkripsi, cutting, crop, subtitle) dilakukan lokal oleh Python + FFmpeg, jadi LLM tidak pernah menyentuh file video sama sekali — hemat token secara drastis.

## Fitur

- Terima link YouTube langsung di chat Telegram (tanpa command), whitelist user via `TELEGRAM_ALLOWED_USERS`
- Transkripsi otomatis (`faster-whisper`, word-level timestamps)
- Seleksi momen viral + metadata (judul, deskripsi, tags, hashtag) via LLM, provider-agnostic
- Pilihan format output per job lewat tombol Telegram: **Vertical (9:16)** dengan crop wajah otomatis, atau **Original** (rasio aspek video sumber dipertahankan apa adanya)
- Crop wajah otomatis (OpenCV + YuNet) yang mengelompokkan wajah terdeteksi dan mengunci ke pembicara paling dominan — termasuk penanganan video split-screen 2 pembicara (lihat [Kekurangan](#kekurangan-known-limitations))
- Subtitle burn-in otomatis dari transkrip word-level
- Progress job real-time di Telegram (pesan diedit per perpindahan tahap, bukan spam pesan baru)
- Pipeline resumable: gagal di satu tahap bisa retry tanpa mengulang dari awal (artefak intermediate tidak dihapus)

## Alur Sistem (End-to-End)

```
User kirim link YouTube ke bot Telegram
        │
        ▼
[1] Bot validasi whitelist + link → balas 2 tombol format (Vertical / Original)
        │           (job BELUM dibuat di database sampai titik ini)
        ▼
[2] User pilih format → job dibuat (status: QUEUED)
        │
        ▼
[3] Worker ambil job → download video+audio (yt-dlp)          status: DOWNLOADING
        │
        ▼
[4] Transkripsi audio (faster-whisper, word_timestamps=True)  status: TRANSCRIBING
        │
        ▼
[5] Kirim transkrip ringkas ke LLM → JSON rencana klip         status: ANALYZING
        │           (validasi Pydantic; retry maks 2x kalau invalid)
        ▼
[6] Per klip: cut → crop (9:16 atau original) → burn subtitle  status: RENDERING
        │           (FFmpeg, re-encode frame-accurate)
        ▼
[7] Kirim tiap klip ke Telegram + caption siap copy-paste      status: DONE
```

Setiap tahap meng-update status job di SQLite. Kalau gagal di satu tahap, status jadi `FAILED` dengan pesan error, dan artefak tahap-tahap sebelumnya (video mentah, transkrip, JSON LLM) **tidak dihapus** — supaya bisa retry dari tahap yang gagal saja, tidak perlu download/transkripsi ulang.

## Stack

Python 3.11+, aiogram v3 (bot Telegram), yt-dlp, faster-whisper, FFmpeg (subprocess), OpenCV + YuNet (deteksi wajah), SQLite, Pydantic v2 + pydantic-settings.

## Menjalankan

```bash
pip install -e ".[dev]"          # install
cp .env.example .env             # isi TELEGRAM_BOT_TOKEN, LLM_*, dst.
python -m autoclip.bot           # jalankan bot + worker (satu proses)
```

```bash
pytest                           # test
ruff check . && ruff format .    # lint + format
```

Konfigurasi lengkap (token bot, whitelist user, endpoint/model LLM, batas durasi & jumlah klip, model whisper) diatur lewat `.env` — lihat `.env.example` dan `CLAUDE.md`.

## Kekurangan (Known Limitations)

- **Split-screen belum tertangani sepenuhnya.** Untuk video dengan 2 pembicara bersebelahan yang sama-sama dominan sepanjang klip, pipeline mendeteksi kondisi ini dan merender layout bertumpuk (masing-masing speaker jadi strip terpisah) alih-alih memotong keduanya jadi setengah-setengah. Tapi ini masih pendekatan heuristik dengan keterbatasan nyata:
  - Cuma menangani **2 klaster wajah**; split-screen 3 orang ke atas (mis. panel diskusi) atau layout non-horizontal (picture-in-picture, grid, overlay facecam+screen-share) tidak terdeteksi dengan benar.
  - Keputusan "split atau tidak" diambil dari sampel beberapa frame saja per klip (bukan per-frame), jadi kalau layout berubah di tengah klip (mis. potongan kamera, speaker keluar-masuk frame) hasil crop-nya tetap statis mengikuti sampel awal.
  - Akurasinya bergantung pada deteksi wajah YuNet — pencahayaan buruk, wajah menyamping, atau resolusi rendah bisa membuat klaster salah terhitung.
- **Crop wajah statis per klip, bukan dinamis per-frame.** Kalau pembicara aktif berpindah tengah-klip (bukan split-screen, tapi mis. kamera pan atau ganti pembicara di single-frame), window crop tidak ikut bergerak.
- **LLM bisa berhalusinasi timestamp pada transkrip sangat panjang.** Untuk video >1 jam, model kadang menghasilkan timestamp yang melebihi durasi video sungguhan — divalidasi & di-retry, tapi tidak dijamin selalu berhasil dalam 3 percobaan.
- **Satu worker, serial.** Job diproses satu-satu berurutan (sesuai desain v1), belum ada paralelisme/antrian prioritas.
- **Tidak ada retry otomatis di background.** Job yang `FAILED` harus di-retry manual (re-submit link atau panggil ulang proses job); tidak ada scheduler yang otomatis mencoba lagi.
