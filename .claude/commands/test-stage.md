Jalankan dan verifikasi satu tahap pipeline secara terisolasi: $ARGUMENTS

Langkah:
1. Baca BRIEF.md bagian yang relevan dengan tahap tersebut, dan CLAUDE.md untuk konvensi.
2. Jika tahap tersebut belum punya CLI kecil untuk dites terpisah (mis. `python -m autoclip.pipeline.transcribe <audio>`), buatkan entry point `if __name__ == "__main__"` sederhana.
3. Jalankan tahap tersebut memakai fixture di tests/fixtures/ (atau minta saya menyediakan file sampel jika belum ada).
4. Tampilkan hasilnya (path artefak, potongan isi transkrip/JSON/log ffmpeg) dan verifikasi terhadap aturan validasi di BRIEF.md bagian 4.
5. Jika ada kegagalan, diagnosis dan perbaiki, lalu ulangi dari langkah 3.
6. Terakhir jalankan `ruff check .` dan pytest untuk modul terkait.
