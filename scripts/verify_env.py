"""Check local dependencies and actual model inference if supplied with a consented photo."""
import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, default=None, help="Single-face photo consented for local diagnostic")
    args = parser.parse_args()
    print(f"Python: {sys.version.split()[0]}")
    missing = []
    for name in ("cv2", "numpy", "PIL", "streamlit", "pandas"):
        available = importlib.util.find_spec(name) is not None
        print(f"Dependency {name}: {'OK' if available else 'MISSING'}")
        if not available:
            missing.append(name)
    if missing:
        print("Install missing dependencies: python -m pip install -r requirements.txt")
        return 1
    import cv2
    from faceauth.config import MODEL_DIR
    from scripts.download_models import validated
    from faceauth.config import MODELS
    from faceauth.vision import FaceEngine, read_photo_file
    print("OpenCV:", cv2.__version__)
    if not hasattr(cv2, "FaceDetectorYN") or not hasattr(cv2, "FaceRecognizerSF"):
        print("FAIL: current OpenCV lacks YuNet / SFace support")
        return 2
    all_ok = True
    for info in MODELS:
        ok = validated(MODEL_DIR / info.filename, info.sha256)
        print(f"ONNX: {info.filename}: {'OK' if ok else 'MISSING OR BAD SHA-256'}")
        all_ok &= ok
    if not all_ok:
        print("Run: python scripts/download_models.py")
        return 3
    engine = FaceEngine()
    print("YuNet and SFace loaded into OpenCV DNN: OK")
    if args.image:
        observation = engine.extract(read_photo_file(args.image))
        print(f"REAL INFERENCE OK: face={observation.box_xywh}, embedding_dim={len(observation.embedding)}, "
              f"detection={observation.detect_ms:.2f} ms, feature={observation.encode_ms:.2f} ms")
    else:
        print("Real single-face inference not run: supply --image path/to/consented.jpg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
