"""Inspect YuNet multi-box detections on LFW face crops, without doing identification.

Annotated face photos are saved to local data/ (gitignored). Do not publish them.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from faceauth.vision import FaceEngine, choose_lfw_primary_face, read_photo_file
from scripts.eval_pairs import get_pairs, safe_image_path


def main():
    p = argparse.ArgumentParser(description="Visualize YuNet multiple-box detections in LFW crops")
    p.add_argument("--csv", type=Path, default=Path("data/lfw_pairs/val_pairs.csv"))
    p.add_argument("--images", type=Path, default=Path("data/lfw_home/lfw_funneled"))
    p.add_argument("--output-dir", type=Path, default=Path("data/lfw_detection_audit"))
    p.add_argument("--limit", type=int, default=12)
    p.add_argument("--yunet-score", type=float, default=0.8)
    p.add_argument("--min-face-size", type=int, default=55)
    args = p.parse_args()
    if not 1 <= args.limit <= 100:
        p.error("--limit must be between 1 and 100")
    if not 0 < args.yunet_score <= 1:
        p.error("--yunet-score must be in (0,1]")
    if not 1 <= args.min_face_size <= 1000:
        p.error("--min-face-size must be in [1,1000]")
    if "lfw_funneled" not in str(args.images):
        p.error("Only use the local LFW funneled crop dataset")

    engine = FaceEngine(score_threshold=args.yunet_score, min_face_size=args.min_face_size)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    names = []
    seen = set()
    for a,b,_ in get_pairs(args.csv):
        for name in (a,b):
            if name not in seen:
                seen.add(name)
                names.append(name)

    multi_count = 0
    inspected = 0
    records = []
    for name in names:
        image = read_photo_file(safe_image_path(args.images, name))
        h,w = image.shape[:2]
        with engine._lock:
            engine.detector.setInputSize((w,h))
            _, faces = engine.detector.detect(image)
        inspected += 1
        if faces is None or len(faces) <= 1:
            continue
        multi_count += 1
        arr = np.asarray(faces, dtype=np.float32)
        _, chosen_idx = choose_lfw_primary_face(arr,w,h)
        canvas=image.copy()
        face_info=[]
        for i,face in enumerate(arr):
            x,y,bw,bh=tuple(int(round(float(v))) for v in face[:4])
            score=float(face[14])
            chosen=i==chosen_idx
            color=(42,180,30) if chosen else (25,130,250)
            cv2.rectangle(canvas,(max(0,x),max(0,y)),(min(w-1,x+bw),min(h-1,y+bh)),color,2)
            cv2.putText(canvas, f"{i}: {score:.2f}", (max(0,x), max(14,y-6)),
                        cv2.FONT_HERSHEY_SIMPLEX,0.44,color,1,cv2.LINE_AA)
            face_info.append({"index":i,"x":x,"y":y,"width":bw,"height":bh,
                              "score":round(score,5),"largest":chosen,
                              "min_size_passed":bw >= args.min_face_size and bh >= args.min_face_size})
        dst=args.output_dir / f"face_{multi_count:02d}.jpg"
        if not cv2.imwrite(str(dst),canvas):
            raise OSError(f"Could not save {dst}")
        records.append({"local_image":name,"output_image":dst.name,
                        "candidate_count":len(arr),"selected_index":chosen_idx,
                        "faces":face_info})
        print(f"[{multi_count:02d}] {name}: {len(arr)} boxes; largest={chosen_idx}; saved={dst}")
        if multi_count >= args.limit:
            break
    result={"dataset":"LFW funneled crop", "inspected_unique_images": inspected,
            "saved_multibox_images":multi_count,"yunet_score":args.yunet_score,
            "min_face_size":args.min_face_size,
            "notice":"Local visual diagnostics only. Contains real face images; do not upload to GitHub.",
            "detections":records}
    (args.output_dir/"audit.json").write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(f"DONE: {multi_count} annotated images saved to {args.output_dir}")


if __name__ == "__main__":
    main()
