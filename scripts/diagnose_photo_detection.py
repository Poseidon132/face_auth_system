"""Local-only YuNet diagnostics for a user-supplied (consented) photo.

Prints detection counts at several resolutions. Does not upload or save the photo.
Usage: python scripts/diagnose_photo_detection.py "D:\\Edge\\my_photo.jpg"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from faceauth.vision import FaceEngine, FaceInputError, read_photo_file


def analyze(image_path: Path) -> None:
    image = read_photo_file(image_path)
    h, w = image.shape[:2]
    engine = FaceEngine()
    print(f"Decoded image: {w} x {h} (after EXIF/thumbnail handling)")
    print("YuNet detection counts at different scales (threshold 0.80):")
    seen = set()
    for cap in (max(w, h), 960, 640, 480, 320):
        scale = min(1.0, cap / max(w, h))
        tw, th = max(1, round(w * scale)), max(1, round(h * scale))
        if (tw, th) in seen:
            continue
        seen.add((tw, th))
        frame = image if (tw, th) == (w, h) else cv2.resize(image, (tw, th), interpolation=cv2.INTER_AREA)
        with engine._lock:
            engine.detector.setInputSize((tw, th))
            _, faces = engine.detector.detect(frame)
        count = 0 if faces is None else len(faces)
        conf = [] if count == 0 else [round(float(row[14]), 3) for row in faces]
        print(f"  {tw} x {th}: faces={count}, scores={conf}")
    try:
        face = engine.extract(image, annotate=False)
        print(f"PASS: detected 1 face | bbox={face.box_xywh} | "
              f"detector_frame={face.detection_input_size} | "
              f"detector={face.detect_ms:.1f}ms | SFace={face.encode_ms:.1f}ms")
    except FaceInputError as exc:
        print(f"FAIL: {exc}")


def main() -> None:
    p = argparse.ArgumentParser(description="Diagnose YuNet scaling on a local face photo")
    p.add_argument("photo", type=Path, help="local JPG/PNG/WebP photo (no upload)")
    args = p.parse_args()
    analyze(args.photo)


if __name__ == "__main__":
    main()
