"""Deteksi wajah (OpenCV + YuNet) untuk parameter crop 9:16.

Mendeteksi wajah pada beberapa frame sampel dari rentang klip, kelompokkan
berdasarkan posisi X, lalu menghitung window crop statis `ih*9/16 : ih`
berpusat pada klaster wajah terbanyak (lihat BRIEF.md bagian 5.3 poin 2)
untuk dipakai render.py sebelum di-scale ke 1080x1920. Kalau tidak ada
wajah terdeteksi sama sekali, crop jatuh ke tengah frame secara horizontal.
Kalau dua klaster sama-sama dominan (split-screen genuine, lihat
`_pick_split_clusters`), dikembalikan `SplitCropWindow` (dua strip 9:8
ditumpuk) alih-alih memaksa pilih satu speaker.

`YuNetFaceDetector` butuh file model `face_detection_yunet_2023mar.onnx`
(unduh dari OpenCV Zoo) ditaruh di `autoclip/assets/`. Untuk unit test,
suntikkan implementasi `FaceDetector` tiruan lewat parameter `detector`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypeVar

import cv2
import numpy as np

_T = TypeVar("_T")

_ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
_DEFAULT_MODEL_PATH = _ASSETS_DIR / "face_detection_yunet_2023mar.onnx"
_SAMPLE_COUNT = 9
# Default OpenCV (0.9) sering false-negative untuk wajah redup/menyamping --
# video split-screen dengan pencahayaan buruk butuh threshold lebih longgar
# supaya tidak jatuh ke fallback tengah-frame gara-gara nol wajah terdeteksi.
_SCORE_THRESHOLD = 0.6
_NMS_THRESHOLD = 0.3
_TOP_K = 5000

# Dua klaster dianggap "seri" (split-screen genuine, dua pembicara sama-sama
# dominan) kalau selisih jumlah deteksinya di bawah ini. Nilai kecil sengaja
# dipilih supaya kasus "satu speaker jelas lebih sering" (rasio besar) tetap
# masuk jalur crop tunggal seperti biasa.
_SPLIT_TIE_MARGIN = 1
# Klaster minor butuh minimal ini biar bukan cuma satu deteksi nyasar/noise.
_MIN_SPLIT_DETECTIONS = 2


@dataclass(frozen=True)
class FaceBox:
    x: float
    y: float
    width: float
    height: float

    @property
    def center_x(self) -> float:
        return self.x + self.width / 2

    @property
    def center_y(self) -> float:
        return self.y + self.height / 2


@dataclass(frozen=True)
class CropWindow:
    """Parameter crop 9:16 statis (piksel), siap dipakai ffmpeg crop filter."""

    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class SplitCropWindow:
    """Dua window crop untuk split-screen genuine (dua pembicara terdeteksi
    sama-sama dominan sepanjang klip -- lihat `_pick_split_clusters`).

    Satu window crop tunggal terpaksa memilih salah satu speaker dan
    menghilangkan yang lain sepenuhnya; di sini keduanya di-crop sebagai
    strip 9:8 masing-masing lalu ditumpuk atas-bawah oleh render.py supaya
    keduanya tetap terlihat di output 9:16.
    """

    top: CropWindow
    bottom: CropWindow


class FaceDetector(Protocol):
    """Antarmuka minimal detektor wajah: satu frame BGR -> daftar bounding box."""

    def detect(self, frame: np.ndarray) -> list[FaceBox]: ...


class YuNetFaceDetector:
    """Wrapper OpenCV YuNet. Butuh file model .onnx, lihat `_DEFAULT_MODEL_PATH`."""

    def __init__(
        self,
        model_path: Path = _DEFAULT_MODEL_PATH,
        input_size: tuple[int, int] = (320, 320),
        score_threshold: float = _SCORE_THRESHOLD,
    ) -> None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model YuNet tidak ditemukan di {model_path}. Unduh "
                "face_detection_yunet_2023mar.onnx dari OpenCV Zoo dan taruh di path tsb."
            )
        self._detector = cv2.FaceDetectorYN.create(
            str(model_path), "", input_size, score_threshold, _NMS_THRESHOLD, _TOP_K
        )

    def detect(self, frame: np.ndarray) -> list[FaceBox]:
        height, width = frame.shape[:2]
        self._detector.setInputSize((width, height))
        _, faces = self._detector.detect(frame)
        if faces is None:
            return []
        return [FaceBox(x=row[0], y=row[1], width=row[2], height=row[3]) for row in faces]


def probe_frame_size(video_path: Path) -> tuple[int, int]:
    """Buka video sebentar cuma untuk baca (width, height) asli -- TANPA
    sampling/deteksi wajah sama sekali. Dipakai render.py untuk format
    ORIGINAL (tidak butuh crop wajah), cuma perlu tahu resolusi asli untuk
    `PlayRes` subtitle di subtitles.py."""
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Tidak bisa membuka video: {video_path}")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    return width, height


def compute_crop_window(
    video_path: Path,
    clip_start_seconds: float,
    clip_end_seconds: float,
    detector: FaceDetector | None = None,
    sample_count: int = _SAMPLE_COUNT,
) -> CropWindow | SplitCropWindow:
    """Sampling beberapa frame dari rentang klip, kelompokkan wajah yang
    terdeteksi berdasarkan posisi X, lalu hitung window crop.

    Kasus normal (satu speaker dominan atau tidak ada wajah sama sekali):
    satu `CropWindow` 9:16 statis berpusat pada klaster wajah terbanyak.
    Kasus split-screen genuine (dua klaster sama-sama dominan sepanjang
    klip -- lihat `_pick_split_clusters`): `SplitCropWindow` dua strip 9:8
    yang ditumpuk oleh render.py, supaya kedua speaker tetap terlihat
    alih-alih salah satunya "menang" dan yang lain hilang total.
    """
    face_detector = detector if detector is not None else YuNetFaceDetector()

    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Tidak bisa membuka video: {video_path}")

        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        faces: list[FaceBox] = []
        for timestamp in _sample_timestamps(clip_start_seconds, clip_end_seconds, sample_count):
            frame = _read_frame_at(capture, timestamp)
            if frame is None:
                continue
            faces.extend(face_detector.detect(frame))
    finally:
        capture.release()

    crop_width = min(round(frame_height * 9 / 16), frame_width)
    clusters = _cluster_by_key(faces, crop_width, key=lambda face: face.center_x)

    split = _pick_split_clusters(clusters)
    if split is not None:
        left, right = split
        top = _build_half_crop(left, frame_width, frame_height)
        bottom = _build_half_crop(right, frame_width, frame_height)
        return SplitCropWindow(top=top, bottom=bottom)

    center_xs = [face.center_x for face in faces]
    face_center_x = _dominant_face_center_x(center_xs, crop_width, frame_width)

    x = round(face_center_x - crop_width / 2)
    x = max(0, min(x, frame_width - crop_width))

    return CropWindow(x=x, y=0, width=crop_width, height=frame_height)


def _cluster_by_key(items: list[_T], gap: int, key: Callable[[_T], float]) -> list[list[_T]]:
    """Kelompokkan `items` berdasarkan celah `key(item)`: dianggap klaster
    berbeda kalau jaraknya lebih lebar dari `gap`, karena keduanya memang
    tidak mungkin muat dalam satu crop window sekaligus."""
    if not items:
        return []
    sorted_items = sorted(items, key=key)
    clusters: list[list[_T]] = [[sorted_items[0]]]
    for item in sorted_items[1:]:
        if key(item) - key(clusters[-1][-1]) > gap:
            clusters.append([item])
        else:
            clusters[-1].append(item)
    return clusters


def _dominant_face_center_x(center_xs: list[float], crop_width: int, frame_width: int) -> float:
    """Rata-ratakan posisi X wajah dari klaster TERBANYAK saja, bukan semua wajah
    tercampur jadi satu rata-rata.

    Video split-screen (2 pembicara bersebelahan) mendeteksi dua wajah yang
    letaknya jauh terpisah di tiap frame. Merata-ratakan semuanya menghasilkan
    titik tengah PERSIS DI GARIS PEMISAH split screen, bukan di salah satu
    wajah -- crop 9:16 di titik itu malah memotong kedua wajah jadi
    setengah-setengah (bug nyata yang pernah terjadi). Kelompokkan dulu posisi
    wajah berdasarkan celah X, lalu pakai klaster dengan jumlah deteksi
    terbanyak (pembicara paling sering terdeteksi di sampel frame).
    """
    if not center_xs:
        return frame_width / 2

    clusters = _cluster_by_key(center_xs, crop_width, key=lambda x: x)
    dominant = max(clusters, key=len)
    return sum(dominant) / len(dominant)


def _pick_split_clusters(
    clusters: list[list[FaceBox]],
) -> tuple[list[FaceBox], list[FaceBox]] | None:
    """Deteksi split-screen genuine: dua klaster wajah yang JUMLAH DETEKSINYA
    HAMPIR SAMA (selisih <= `_SPLIT_TIE_MARGIN`) sepanjang klip -- artinya
    tidak ada speaker yang benar-benar dominan, keduanya sama-sama tampil
    terus. Beda dengan kasus "dua host tapi satu lebih sering ngomong/kelihatan"
    (rasio deteksi timpang), yang tetap ditangani `_dominant_face_center_x`
    seperti biasa karena ada pemenang yang jelas.

    Return None kalau bukan kasus seri (termasuk kalau salah satu klaster
    cuma 1 deteksi -- terlalu rawan cuma noise, bukan speaker sungguhan).
    """
    if len(clusters) < 2:
        return None

    ranked = sorted(clusters, key=len, reverse=True)
    a, b = ranked[0], ranked[1]
    if len(a) < _MIN_SPLIT_DETECTIONS or len(b) < _MIN_SPLIT_DETECTIONS:
        return None
    if len(a) - len(b) > _SPLIT_TIE_MARGIN:
        return None

    def _cluster_center_x(cluster: list[FaceBox]) -> float:
        return sum(face.center_x for face in cluster) / len(cluster)

    left, right = sorted((a, b), key=_cluster_center_x)
    return left, right


def _build_half_crop(cluster: list[FaceBox], frame_width: int, frame_height: int) -> CropWindow:
    """Crop satu strip 9:8 berpusat pada satu klaster wajah -- dipakai
    berpasangan (atas+bawah) untuk output split-screen 9:16 di render.py."""
    half_width = min(round(frame_height * 9 / 8), frame_width // 2)
    half_height = min(round(half_width * 8 / 9), frame_height)

    center_x = sum(face.center_x for face in cluster) / len(cluster)
    center_y = sum(face.center_y for face in cluster) / len(cluster)

    x = round(center_x - half_width / 2)
    x = max(0, min(x, frame_width - half_width))
    y = round(center_y - half_height / 2)
    y = max(0, min(y, frame_height - half_height))

    return CropWindow(x=x, y=y, width=half_width, height=half_height)


def _sample_timestamps(start: float, end: float, count: int) -> list[float]:
    if count <= 1:
        return [(start + end) / 2]
    step = (end - start) / (count + 1)
    return [start + step * i for i in range(1, count + 1)]


def _read_frame_at(capture: cv2.VideoCapture, timestamp_seconds: float) -> np.ndarray | None:
    capture.set(cv2.CAP_PROP_POS_MSEC, timestamp_seconds * 1000)
    success, frame = capture.read()
    return frame if success else None
