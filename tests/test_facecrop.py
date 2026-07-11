"""Test untuk pipeline/facecrop.py: sampling frame & perhitungan window crop 9:16."""

from __future__ import annotations

from pathlib import Path

import pytest

from autoclip.pipeline.facecrop import (
    CropWindow,
    FaceBox,
    YuNetFaceDetector,
    _sample_timestamps,
    compute_crop_window,
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


def test_compute_crop_window_raises_when_video_cannot_be_opened() -> None:
    with pytest.raises(ValueError, match="Tidak bisa membuka video"):
        compute_crop_window(Path("/does/not/exist.mp4"), 0.0, 1.0, detector=_NoFaceDetector())


def test_yunet_face_detector_raises_when_model_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Model YuNet tidak ditemukan"):
        YuNetFaceDetector(model_path=tmp_path / "missing.onnx")


def test_face_box_center_x() -> None:
    box = FaceBox(x=100.0, y=0.0, width=50.0, height=80.0)
    assert box.center_x == 125.0
