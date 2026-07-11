"""Deteksi wajah (OpenCV + YuNet) untuk parameter crop 9:16.

Mendeteksi wajah pada beberapa frame sampel dari rentang klip,
merata-ratakan posisi X wajah, lalu menghitung window crop statis
`ih*9/16 : ih` yang berpusat pada wajah (lihat BRIEF.md bagian 5.3
poin 2) untuk dipakai render.py sebelum di-scale ke 1080x1920. Kalau
tidak ada wajah terdeteksi sama sekali, crop jatuh ke tengah frame
secara horizontal.

`YuNetFaceDetector` butuh file model `face_detection_yunet_2023mar.onnx`
(unduh dari OpenCV Zoo) ditaruh di `autoclip/assets/`. Untuk unit test,
suntikkan implementasi `FaceDetector` tiruan lewat parameter `detector`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

_ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
_DEFAULT_MODEL_PATH = _ASSETS_DIR / "face_detection_yunet_2023mar.onnx"
_SAMPLE_COUNT = 5


@dataclass(frozen=True)
class FaceBox:
    x: float
    y: float
    width: float
    height: float

    @property
    def center_x(self) -> float:
        return self.x + self.width / 2


@dataclass(frozen=True)
class CropWindow:
    """Parameter crop 9:16 statis (piksel), siap dipakai ffmpeg crop filter."""

    x: int
    y: int
    width: int
    height: int


class FaceDetector(Protocol):
    """Antarmuka minimal detektor wajah: satu frame BGR -> daftar bounding box."""

    def detect(self, frame: np.ndarray) -> list[FaceBox]: ...


class YuNetFaceDetector:
    """Wrapper OpenCV YuNet. Butuh file model .onnx, lihat `_DEFAULT_MODEL_PATH`."""

    def __init__(
        self,
        model_path: Path = _DEFAULT_MODEL_PATH,
        input_size: tuple[int, int] = (320, 320),
    ) -> None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model YuNet tidak ditemukan di {model_path}. Unduh "
                "face_detection_yunet_2023mar.onnx dari OpenCV Zoo dan taruh di path tsb."
            )
        self._detector = cv2.FaceDetectorYN.create(str(model_path), "", input_size)

    def detect(self, frame: np.ndarray) -> list[FaceBox]:
        height, width = frame.shape[:2]
        self._detector.setInputSize((width, height))
        _, faces = self._detector.detect(frame)
        if faces is None:
            return []
        return [FaceBox(x=row[0], y=row[1], width=row[2], height=row[3]) for row in faces]


def compute_crop_window(
    video_path: Path,
    clip_start_seconds: float,
    clip_end_seconds: float,
    detector: FaceDetector | None = None,
    sample_count: int = _SAMPLE_COUNT,
) -> CropWindow:
    """Sampling beberapa frame dari rentang klip, rata-ratakan posisi X wajah,
    lalu hitung window crop 9:16 statis berpusat pada wajah.
    """
    face_detector = detector if detector is not None else YuNetFaceDetector()

    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Tidak bisa membuka video: {video_path}")

        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        center_xs: list[float] = []
        for timestamp in _sample_timestamps(clip_start_seconds, clip_end_seconds, sample_count):
            frame = _read_frame_at(capture, timestamp)
            if frame is None:
                continue
            center_xs.extend(face.center_x for face in face_detector.detect(frame))
    finally:
        capture.release()

    crop_width = min(round(frame_height * 9 / 16), frame_width)
    face_center_x = _dominant_face_center_x(center_xs, crop_width, frame_width)

    x = round(face_center_x - crop_width / 2)
    x = max(0, min(x, frame_width - crop_width))

    return CropWindow(x=x, y=0, width=crop_width, height=frame_height)


def _dominant_face_center_x(center_xs: list[float], crop_width: int, frame_width: int) -> float:
    """Rata-ratakan posisi X wajah dari klaster TERBANYAK saja, bukan semua wajah
    tercampur jadi satu rata-rata.

    Video split-screen (2 pembicara bersebelahan) mendeteksi dua wajah yang
    letaknya jauh terpisah di tiap frame. Merata-ratakan semuanya menghasilkan
    titik tengah PERSIS DI GARIS PEMISAH split screen, bukan di salah satu
    wajah -- crop 9:16 di titik itu malah memotong kedua wajah jadi
    setengah-setengah (bug nyata yang pernah terjadi). Kelompokkan dulu posisi
    wajah berdasarkan celah X: dianggap wajah berbeda kalau jaraknya lebih
    lebar dari crop_width, karena keduanya memang tidak mungkin muat dalam
    satu crop window sekaligus. Lalu pakai klaster dengan jumlah deteksi
    terbanyak (pembicara paling sering terdeteksi di sampel frame).
    """
    if not center_xs:
        return frame_width / 2

    sorted_xs = sorted(center_xs)
    clusters: list[list[float]] = [[sorted_xs[0]]]
    for value in sorted_xs[1:]:
        if value - clusters[-1][-1] > crop_width:
            clusters.append([value])
        else:
            clusters[-1].append(value)

    dominant = max(clusters, key=len)
    return sum(dominant) / len(dominant)


def _sample_timestamps(start: float, end: float, count: int) -> list[float]:
    if count <= 1:
        return [(start + end) / 2]
    step = (end - start) / (count + 1)
    return [start + step * i for i in range(1, count + 1)]


def _read_frame_at(capture: cv2.VideoCapture, timestamp_seconds: float) -> np.ndarray | None:
    capture.set(cv2.CAP_PROP_POS_MSEC, timestamp_seconds * 1000)
    success, frame = capture.read()
    return frame if success else None
