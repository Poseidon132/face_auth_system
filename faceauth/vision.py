"""Image ingestion, face quality, OpenCV DNN embedding extraction.

This is a local biometric demo. Quality checks are NOT liveness/anti-spoofing.
"""
from __future__ import annotations

import hashlib
import io
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .config import (
    DEFAULT_THRESHOLD,
    DETECTOR_FILE,
    DETECTOR_SHA256,
    MAX_IMAGE_BYTES,
    MAX_IMAGE_SIDE,
    MODEL_DIR,
    RECOGNIZER_FILE,
    RECOGNIZER_SHA256,
)


class FaceInputError(ValueError):
    """Invalid or unsuitable input photograph."""


def read_photo(contents: bytes) -> np.ndarray:
    """Decode an untrusted upload with EXIF orientation and size guards.

    Returns BGR uint8 image. Never writes raw photos to the database.
    """
    if not contents:
        raise FaceInputError("图片为空，请重新上传。")
    if len(contents) > MAX_IMAGE_BYTES:
        raise FaceInputError("图片超过 12 MB，请压缩后重试。")
    try:
        with Image.open(io.BytesIO(contents)) as img:
            if (img.format or "").upper() not in {"JPEG", "PNG", "WEBP"}:
                raise FaceInputError("仅支持 JPG、PNG 或 WebP 图片。")
            if img.width < 64 or img.height < 64:
                raise FaceInputError("图片太小，建议至少 256×256 像素。")
            if max(img.size) > MAX_IMAGE_SIDE:
                raise FaceInputError("图片边长不得超过 4096 像素。")
            rgb = ImageOps.exif_transpose(img).convert("RGB")
            if max(rgb.size) > 1600:
                rgb.thumbnail((1600, 1600), resample=Image.Resampling.LANCZOS)
            pixels = np.array(rgb, dtype=np.uint8, copy=True)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, ValueError) as exc:
        if isinstance(exc, FaceInputError):
            raise
        raise FaceInputError("图片无法解析，请上传 JPG、PNG 或 WebP 文件。") from exc
    return cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)


def read_photo_file(path: str | Path) -> np.ndarray:
    path = Path(path)
    if not path.is_file():
        raise FaceInputError(f"图片不存在：{path}")
    if path.stat().st_size > MAX_IMAGE_BYTES:
        raise FaceInputError("图片超过大小限制。")
    return read_photo(path.read_bytes())


def encode_jpeg_preview(image: np.ndarray, *, max_width: int = 660) -> bytes:
    h, w = image.shape[:2]
    view = image
    if w > max_width:
        view = cv2.resize(image, (max_width, round(h * max_width / w)))
    ok, data = cv2.imencode(".jpg", view, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise FaceInputError("无法生成预览图片")
    return data.tobytes()


def normalize_feature(feature: np.ndarray) -> np.ndarray:
    emb = np.asarray(feature, dtype=np.float32).reshape(-1)
    if not 16 <= len(emb) <= 4096 or not np.isfinite(emb).all():
        raise ValueError("模型输出无效：维度或数值不正确。")
    norm = float(np.linalg.norm(emb))
    if norm < 1e-10:
        raise ValueError("模型输出为零向量。")
    return np.ascontiguousarray(emb / norm, dtype=np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = normalize_feature(a)
    b = normalize_feature(b)
    if a.shape != b.shape:
        raise ValueError("特征维度不一致，可能使用了不同的识别模型。")
    return float(np.clip(np.dot(a, b), -1.0, 1.0))


@dataclass(frozen=True)
class FaceObservation:
    embedding: np.ndarray
    detection_score: float
    box_xywh: tuple[int, int, int, int]
    face_width: int
    face_height: int
    blur_score: float
    detect_ms: float
    encode_ms: float
    preview_jpeg: bytes | None = None
    detected_face_count: int = 1
    selection_policy: str = "strict"
    detection_input_size: tuple[int, int] | None = None


def choose_lfw_primary_face(faces: np.ndarray, image_width: int, image_height: int) -> tuple[np.ndarray, int]:
    """Select the largest YuNet box in an already centered LFW crop.

    This benchmark-only policy is not appropriate for live identity verification:
    multiple real people in an authentication image must remain a rejection.
    """
    rows = np.asarray(faces, dtype=np.float32)
    if rows.ndim != 2 or rows.shape[1] < 15 or rows.shape[0] == 0:
        raise ValueError("YuNet detector returned invalid face rows")
    def sort_key(index: int):
        row = rows[index]
        area = max(0.0, float(row[2])) * max(0.0, float(row[3]))
        center_x = float(row[0] + row[2]/2)
        center_y = float(row[1] + row[3]/2)
        center_dist = ((center_x - image_width/2) / max(1,image_width))**2 + ((center_y - image_height/2) / max(1,image_height))**2
        return (area, -center_dist, float(row[14]))
    idx = max(range(len(rows)), key=sort_key)
    return rows[idx], idx


class FaceEngine:
    """YuNet face detection + 5-landmark SFace alignment/embedding.

    OpenCV prebuilt wheels use CPU inference by default, even on CUDA-capable PCs.
    A lock protects changing the detector input size between requests.
    """

    def __init__(
        self,
        model_dir: str | Path = MODEL_DIR,
        score_threshold: float = 0.8,
        min_face_size: int = 55,
        min_blur_score: float = 0.0,
    ) -> None:
        model_dir = Path(model_dir)
        detector_path = model_dir / DETECTOR_FILE
        recognizer_path = model_dir / RECOGNIZER_FILE
        if not detector_path.exists() or not recognizer_path.exists():
            raise FileNotFoundError(
                "缺少预训练 ONNX 文件。请运行 python scripts/download_models.py。"
            )
        for path, expected_hash in (
            (detector_path, DETECTOR_SHA256),
            (recognizer_path, RECOGNIZER_SHA256),
        ):
            if path.stat().st_size < 100000:
                raise RuntimeError(f"模型文件疑似 Git LFS 指针或未下载完整：{path}")
            hasher = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 20), b""):
                    hasher.update(chunk)
            if hasher.hexdigest() != expected_hash:
                raise RuntimeError(f"模型 SHA-256 校验失败，请重新下载：{path.name}")
        if not hasattr(cv2, "FaceDetectorYN") or not hasattr(cv2, "FaceRecognizerSF"):
            raise RuntimeError("当前 opencv-python 不支持 YuNet/SFace，请升级至 4.8 以上版本。")
        self._lock = threading.RLock()
        self.detector = cv2.FaceDetectorYN.create(
            str(detector_path), "", (320, 320), float(score_threshold), 0.3, 5000
        )
        self.recognizer = cv2.FaceRecognizerSF.create(str(recognizer_path), "")
        self.min_face_size = int(min_face_size)
        self.min_blur_score = float(min_blur_score)
        # Store a stable model ID; registrations created by another model won't be silently reused.
        self.model_id = RECOGNIZER_SHA256

    def _detect_on_suitable_size(
        self, image: np.ndarray, *, selection_policy: str
    ) -> tuple[np.ndarray | None, tuple[int, int]]:
        """Detect on bounded-size frames and map faces/landmarks to source pixels.

        A tall phone photo can contain a 600+ pixel face, outside the scale at
        which YuNet is usually reliable. Run detection at a practical scale,
        but ALWAYS align SFace against the original BGR image.

        The LFW benchmark keeps its existing single-pass behavior so historical
        test metrics remain comparable. The live strict policy still rejects
        any detection pass that finds more than one face; it NEVER silently
        selects a face from a multi-face photo.

        This helper is called only while ``self._lock`` is held.
        """
        h, w = image.shape[:2]
        longest = max(h, w)
        if selection_policy == "lfw-largest":
            sides = [longest]
        elif longest > 960:
            sides = [960, 640, 480]
        else:
            sides = [longest]
            sides += [cap for cap in (640, 480) if cap < longest]

        tested = set()
        last_input = (w, h)
        for side in sides:
            scale = min(1.0, float(side) / float(longest))
            target_w = max(1, round(w * scale))
            target_h = max(1, round(h * scale))
            if (target_w, target_h) in tested:
                continue
            tested.add((target_w, target_h))
            last_input = (target_w, target_h)
            if (target_w, target_h) == (w, h):
                detector_image = image
            else:
                detector_image = cv2.resize(
                    image, (target_w, target_h), interpolation=cv2.INTER_AREA
                )
            self.detector.setInputSize((target_w, target_h))
            _, faces = self.detector.detect(detector_image)
            if faces is None or len(faces) == 0:
                continue
            # Never hide a detected second person in a real authentication flow.
            if len(faces) > 1 and selection_policy == "strict":
                return np.asarray(faces, dtype=np.float32), last_input
            # YuNet returns [x,y,w,h, 5*(landmark_x,landmark_y), confidence].
            # Map all the coordinates back; confidence must remain unchanged.
            mapped = np.asarray(faces, dtype=np.float32).copy()
            mapped[:, 0:14:2] *= float(w) / float(target_w)
            mapped[:, 1:14:2] *= float(h) / float(target_h)
            return mapped, last_input
        return None, last_input

    def extract(self, image: np.ndarray, *, annotate: bool = True,
                selection_policy: str = "strict") -> FaceObservation:
        if selection_policy not in {"strict", "lfw-largest"}:
            raise ValueError("Unknown face-selection policy")
        if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise FaceInputError("图片应为 OpenCV BGR uint8 三通道格式。")
        h, w = image.shape[:2]
        if h < 64 or w < 64:
            raise FaceInputError("图片尺寸太小。")
        t0 = time.perf_counter()
        with self._lock:
            faces, detection_input_size = self._detect_on_suitable_size(
                image, selection_policy=selection_policy
            )
            detect_ms = (time.perf_counter() - t0) * 1000
            if faces is None or len(faces) == 0:
                raise FaceInputError("未检测到人脸。请正对镜头并保证光线充足。")
            if len(faces) != 1 and selection_policy == "strict":
                raise FaceInputError(f"检测到 {len(faces)} 张人脸。为避免错误认证，请只保留一张人脸。")
            if selection_policy == "lfw-largest":
                face, _ = choose_lfw_primary_face(faces, w, h)
            else:
                face = np.asarray(faces[0], dtype=np.float32)
            x, y, bw, bh = [int(round(float(v))) for v in face[:4]]
            if bw < self.min_face_size or bh < self.min_face_size:
                raise FaceInputError(f"人脸过小（{bw}×{bh}），请靠近镜头。")
            cx, cy = max(0, x), max(0, y)
            crop = image[cy:min(h, y+bh), cx:min(w, x+bw)]
            if crop.size == 0:
                raise FaceInputError("人脸边界超出图片范围。")
            blur_score = float(cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
            if blur_score < self.min_blur_score:
                raise FaceInputError(f"人脸模糊度过高（清晰度指标 {blur_score:.1f}），请重新拍摄。")
            t1 = time.perf_counter()
            aligned = self.recognizer.alignCrop(image, face)
            raw_embedding = self.recognizer.feature(aligned)
            embedding = normalize_feature(raw_embedding)
            encode_ms = (time.perf_counter() - t1) * 1000
        preview = None
        if annotate:
            view = image.copy()
            cv2.rectangle(view, (max(0,x),max(0,y)), (min(w-1,x+bw),min(h-1,y+bh)), (13,173,105), 2)
            preview = encode_jpeg_preview(view)
        return FaceObservation(
            embedding=embedding,
            detection_score=float(face[14]),
            box_xywh=(x, y, bw, bh),
            face_width=bw,
            face_height=bh,
            blur_score=blur_score,
            detect_ms=detect_ms,
            encode_ms=encode_ms,
            preview_jpeg=preview,
            detected_face_count=int(len(faces)),
            selection_policy=selection_policy,
            detection_input_size=detection_input_size,
        )
