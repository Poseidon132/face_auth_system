"""Paths and model identities. No runtime network requests happen inside the UI."""
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
DETECTOR_FILE = "face_detection_yunet_2023mar.onnx"
RECOGNIZER_FILE = "face_recognition_sface_2021dec.onnx"
DETECTOR_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
RECOGNIZER_SHA256 = "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"
DEFAULT_DB = DATA_DIR / "faceauth.sqlite3"
# A demonstrative, intentionally conservative starting point, not a measured FAR guarantee.
DEFAULT_THRESHOLD = 0.45
MAX_IMAGE_BYTES = 12 * 1024 * 1024
MAX_IMAGE_SIDE = 4096
MAX_ENROLLMENTS_PER_PERSON = 10


@dataclass(frozen=True)
class ModelInfo:
    filename: str
    sha256: str
    sources: tuple[str, ...]


MODELS = (
    ModelInfo(DETECTOR_FILE, DETECTOR_SHA256, (
        "https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx",
        "https://huggingface.co/opencv/opencv_zoo/resolve/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "https://github.com/opencv/opencv_zoo/raw/refs/heads/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    )),
    ModelInfo(RECOGNIZER_FILE, RECOGNIZER_SHA256, (
        "https://huggingface.co/opencv/face_recognition_sface/resolve/main/face_recognition_sface_2021dec.onnx",
        "https://huggingface.co/opencv/opencv_zoo/resolve/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        "https://github.com/opencv/opencv_zoo/raw/refs/heads/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
    )),
)
