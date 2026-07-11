"""Test untuk pipeline/facecrop.py: sampling frame & perhitungan window crop 9:16."""

from __future__ import annotations

from pathlib import Path

import pytest

from autoclip.pipeline.facecrop import (
    CropWindow,
    FaceBox,
    SplitCropWindow,
    YuNetFaceDetector,
    _build_half_crop,
    _dominant_face_center_x,
    _pick_split_clusters,
    _sample_timestamps,
    compute_crop_window,
    probe_frame_size,
)


class _FixedFaceDetector:
    """Detektor tiruan: selalu mengembalikan satu wajah di posisi X tetap."""

    def __init__(self, center_x: float, box_width: float = 100.0, box_height: float = 100.0):
        self._center_x = center_x
        self._box_width = box_width
        self._box_height = box_height

    def detect(self, frame) -> list[FaceBox]:
        return [
            FaceBox(
                x=self._center_x - self._box_width / 2,
                y=0,
                width=self._box_width,
                height=self._box_height,
            )
        ]


class _NoFaceDetector:
    def detect(self, frame) -> list[FaceBox]:
        return []


class _FixedTwoFaceDetector:
    """Detektor tiruan: dua wajah tetap di posisi X berbeda tiap frame
    (simulasi video split-screen dua pembicara bersebelahan)."""

    def __init__(self, left_x: float, right_x: float, box_size: float = 100.0):
        self._left_x = left_x
        self._right_x = right_x
        self._box_size = box_size

    def detect(self, frame) -> list[FaceBox]:
        return [
            FaceBox(
                x=self._left_x - self._box_size / 2,
                y=0,
                width=self._box_size,
                height=self._box_size,
            ),
            FaceBox(
                x=self._right_x - self._box_size / 2,
                y=0,
                width=self._box_size,
                height=self._box_size,
            ),
        ]


class _SplitScreenDetector:
    """Detektor tiruan split-screen: wajah kiri terdeteksi di SETIAP sampel,
    wajah kanan cuma sebagian (mis. sesekali menoleh) -- untuk menguji klaster
    mana yang dianggap dominan."""

    def __init__(self, dominant_x: float, secondary_x: float, secondary_every: int = 2):
        self._dominant_x = dominant_x
        self._secondary_x = secondary_x
        self._secondary_every = secondary_every
        self._call_count = 0

    def detect(self, frame) -> list[FaceBox]:
        self._call_count += 1
        faces = [FaceBox(x=self._dominant_x - 50.0, y=0, width=100.0, height=100.0)]
        if self._call_count % self._secondary_every == 0:
            faces.append(FaceBox(x=self._secondary_x - 50.0, y=0, width=100.0, height=100.0))
        return faces


def test_sample_timestamps_returns_evenly_spaced_points_inside_range() -> None:
    timestamps = _sample_timestamps(start=10.0, end=20.0, count=3)

    assert len(timestamps) == 3
    assert all(10.0 < t < 20.0 for t in timestamps)
    assert timestamps == sorted(timestamps)


def test_sample_timestamps_single_sample_is_midpoint() -> None:
    assert _sample_timestamps(start=10.0, end=20.0, count=1) == [15.0]


def test_compute_crop_window_centers_on_detected_face(sample_video: Path) -> None:
    # Video 640x360 -> crop width = round(360 * 9/16) = 202
    detector = _FixedFaceDetector(center_x=500.0)

    window = compute_crop_window(sample_video, 0.5, 2.5, detector=detector)

    assert isinstance(window, CropWindow)
    assert window.height == 360
    assert window.width == 202
    # window berpusat di x=500 -> x = 500 - 202/2 = 399
    assert window.x == 399
    assert window.y == 0


def test_compute_crop_window_falls_back_to_center_when_no_face(sample_video: Path) -> None:
    window = compute_crop_window(sample_video, 0.5, 2.5, detector=_NoFaceDetector())

    # Tanpa wajah terdeteksi, crop di tengah frame: (640 - 202) / 2 = 219
    assert window.x == 219


def test_compute_crop_window_clamps_when_face_near_left_edge(sample_video: Path) -> None:
    window = compute_crop_window(sample_video, 0.5, 2.5, detector=_FixedFaceDetector(center_x=0.0))

    assert window.x == 0


def test_compute_crop_window_clamps_when_face_near_right_edge(sample_video: Path) -> None:
    window = compute_crop_window(
        sample_video, 0.5, 2.5, detector=_FixedFaceDetector(center_x=640.0)
    )

    assert window.x == 640 - window.width


def test_probe_frame_size_returns_actual_dimensions(sample_video: Path) -> None:
    # sample_video di conftest digenerate 640x360 -- lihat fixture.
    assert probe_frame_size(sample_video) == (640, 360)


def test_probe_frame_size_raises_when_video_cannot_be_opened() -> None:
    with pytest.raises(ValueError, match="Tidak bisa membuka video"):
        probe_frame_size(Path("/does/not/exist.mp4"))


def test_compute_crop_window_raises_when_video_cannot_be_opened() -> None:
    with pytest.raises(ValueError, match="Tidak bisa membuka video"):
        compute_crop_window(Path("/does/not/exist.mp4"), 0.0, 1.0, detector=_NoFaceDetector())


def test_yunet_face_detector_raises_when_model_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Model YuNet tidak ditemukan"):
        YuNetFaceDetector(model_path=tmp_path / "missing.onnx")


def test_face_box_center_x() -> None:
    box = FaceBox(x=100.0, y=0.0, width=50.0, height=80.0)
    assert box.center_x == 125.0


def test_dominant_face_center_x_returns_frame_center_when_no_faces() -> None:
    assert _dominant_face_center_x([], crop_width=200, frame_width=640) == 320


def test_dominant_face_center_x_averages_single_cluster() -> None:
    # Satu wajah yang posisinya bergeser sedikit antar sampel -- tetap satu
    # klaster (celahnya jauh di bawah crop_width), jadi dirata-ratakan seperti biasa.
    assert _dominant_face_center_x([100.0, 120.0, 140.0], crop_width=200, frame_width=640) == 120.0


def test_dominant_face_center_x_picks_larger_cluster_in_split_screen() -> None:
    # Celah antara klaster kiri (~100) dan kanan (500) = 360, lebih lebar dari
    # crop_width 200 -> tidak mungkin muat sekaligus, harus dianggap 2 klaster
    # terpisah. Klaster kiri terdeteksi 3x, kanan cuma 1x -> pakai klaster kiri.
    center_xs = [95.0, 100.0, 105.0, 500.0]
    assert _dominant_face_center_x(center_xs, crop_width=200, frame_width=640) == 100.0


def test_dominant_face_center_x_breaks_tie_by_leftmost_cluster() -> None:
    center_xs = [100.0, 500.0]
    assert _dominant_face_center_x(center_xs, crop_width=200, frame_width=640) == 100.0


def test_compute_crop_window_picks_dominant_cluster_in_split_screen(sample_video: Path) -> None:
    # Kasus nyata yang dilaporkan: video split-screen 2 pembicara. Merata-
    # ratakan center_x kedua wajah akan jatuh di garis pemisah (~300) dan
    # memotong kedua wajah jadi setengah-setengah. Wajah kiri (x=100) terdeteksi
    # di SETIAP sampel, wajah kanan (x=500) cuma separuhnya -- crop harus
    # mengunci ke wajah kiri yang dominan, bukan ke titik tengah keduanya.
    detector = _SplitScreenDetector(dominant_x=100.0, secondary_x=500.0, secondary_every=2)

    window = compute_crop_window(sample_video, 0.5, 2.5, detector=detector, sample_count=5)

    assert window.x == max(0, round(100.0 - window.width / 2))


def test_compute_crop_window_splits_when_both_faces_tied_every_frame(
    sample_video: Path,
) -> None:
    # Kedua wajah terdeteksi SAMA BANYAK di SETIAP frame -- ini split-screen
    # genuine (dua speaker sama-sama tampil terus), bukan cuma "kebetulan
    # seri" di sampel. Tidak ada speaker dominan untuk dipilih, jadi hasilnya
    # harus SplitCropWindow (dua strip ditumpuk), bukan crop tunggal yang
    # memaksa pilih salah satu dan menghilangkan yang lain.
    detector = _FixedTwoFaceDetector(left_x=100.0, right_x=500.0)

    window = compute_crop_window(sample_video, 0.5, 2.5, detector=detector)

    assert isinstance(window, SplitCropWindow)
    # Wajah kiri (x=100) jadi window atas, wajah kanan (x=500) jadi bawah --
    # urutan kiri-ke-kanan yang stabil, bukan urutan deteksi.
    assert window.top.x < window.bottom.x


def _face(center_x: float, center_y: float = 50.0, size: float = 100.0) -> FaceBox:
    return FaceBox(x=center_x - size / 2, y=center_y - size / 2, width=size, height=size)


def test_pick_split_clusters_returns_none_for_single_cluster() -> None:
    clusters = [[_face(100.0), _face(105.0)]]
    assert _pick_split_clusters(clusters) is None


def test_pick_split_clusters_returns_none_when_one_side_dominant() -> None:
    # Selisih deteksi 3 (5 vs 2) -- ada pemenang yang jelas, bukan seri.
    clusters = [[_face(100.0)] * 5, [_face(500.0)] * 2]
    assert _pick_split_clusters(clusters) is None


def test_pick_split_clusters_returns_none_when_minor_cluster_is_single_detection() -> None:
    # Klaster kedua cuma 1 deteksi -- terlalu rawan noise untuk dianggap speaker asli.
    clusters = [[_face(100.0)] * 3, [_face(500.0)]]
    assert _pick_split_clusters(clusters) is None


def test_pick_split_clusters_detects_genuine_tie() -> None:
    clusters = [[_face(500.0)] * 4, [_face(100.0)] * 3]

    result = _pick_split_clusters(clusters)

    assert result is not None
    left, right = result
    assert all(f.center_x == 100.0 for f in left)
    assert all(f.center_x == 500.0 for f in right)


def test_build_half_crop_centers_on_cluster_average(sample_video: Path) -> None:
    cluster = [_face(100.0, center_y=80.0), _face(120.0, center_y=100.0)]

    window = _build_half_crop(cluster, frame_width=640, frame_height=360)

    assert window.width == round(window.height * 9 / 8)
    expected_x = max(0, round(110.0 - window.width / 2))
    expected_y = max(0, round(90.0 - window.height / 2))
    assert window.x == expected_x
    assert window.y == expected_y
