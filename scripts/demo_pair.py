"""Minimal real SFace model sanity check on two consented local face photographs."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from faceauth.config import DEFAULT_THRESHOLD
from faceauth.vision import FaceEngine, cosine_similarity, read_photo_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_a", type=Path, help="Consented local image (single face)")
    parser.add_argument("image_b", type=Path, help="Consented local image (single face)")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = parser.parse_args()
    engine = FaceEngine()
    a = engine.extract(read_photo_file(args.image_a))
    b = engine.extract(read_photo_file(args.image_b))
    score = cosine_similarity(a.embedding,b.embedding)
    print(f"Real SFace embedding dimension: {len(a.embedding)}")
    print(f"A: detect={a.detect_ms:.1f}ms, embed={a.encode_ms:.1f}ms, face={a.box_xywh}")
    print(f"B: detect={b.detect_ms:.1f}ms, embed={b.encode_ms:.1f}ms, face={b.box_xywh}")
    print(f"Cosine similarity: {score:.6f} | threshold: {args.threshold:.2f}")
    print("MATCH" if score >= args.threshold else "NO MATCH")
    print("This score is not a calibrated identity probability. It does not detect spoofing.")


if __name__ == "__main__":
    main()
