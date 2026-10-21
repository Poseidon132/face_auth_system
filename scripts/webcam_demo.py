"""Optional local webcam preview. ESC/Q closes the camera window.

No liveness detection and no access-control integration is provided.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2

from faceauth.config import DEFAULT_DB, DEFAULT_THRESHOLD
from faceauth.service import FaceAuthService
from faceauth.store import FaceStore
from faceauth.vision import FaceEngine, FaceInputError


def main():
    parser = argparse.ArgumentParser(description="Local SFace webcam recognition demo")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--mode", choices=["identify", "verify"], default="identify")
    parser.add_argument("--person", type=str, help="Person name for verify mode")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = parser.parse_args()
    if args.mode == "verify" and not args.person:
        parser.error("--mode verify requires --person")
    engine = FaceEngine()
    store = FaceStore(DEFAULT_DB, model_id=engine.model_id)
    if not store.persons():
        parser.error("No enrolled identities. Register a consenting person in the Streamlit app.")
    service = FaceAuthService(engine, store)
    # CAP_DSHOW only exists on Windows and may not support every camera.
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW) if sys.platform == "win32" else cv2.VideoCapture(args.camera)
    if not cap.isOpened() and sys.platform == "win32":
        cap.release()
        cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("无法打开摄像头。检查系统隐私权限、摄像头编号和其它正在占用摄像头的程序。")
        return 1
    last_infer = -100.0
    display = "Waiting for capture..."
    color = (220, 220, 220)
    try:
        print("Webcam running. Press Q or ESC to quit. Demo only; photos can spoof verification.")
        while True:
            ok, frame = cap.read()
            if not ok:
                print("摄像头画面读取失败")
                break
            if time.monotonic() - last_infer >= 1.1:
                last_infer = time.monotonic()
                try:
                    if args.mode == "identify":
                        result = service.identify(frame, threshold=args.threshold, top_k=1)
                    else:
                        result = service.verify(args.person, frame, threshold=args.threshold)
                    label = result.name if result.accepted else "UNKNOWN"
                    # cv2.putText only supports ASCII well. Non-ASCII names are represented generically.
                    display = f"MATCH: {label}" if label.isascii() and result.accepted else ("MATCH" if result.accepted else "UNKNOWN")
                    if result.similarity is not None:
                        display += f"  sim={result.similarity:.3f}"
                    color = (49, 191, 94) if result.accepted else (65, 65, 222)
                    x,y,w,h = result.observation.box_xywh
                    last_box = (x,y,w,h)
                except FaceInputError as exc:
                    display = "NO SINGLE FACE"
                    color = (65, 65, 222)
                    last_box = None
                except Exception as exc:
                    display = "ERROR; see terminal"
                    print(type(exc).__name__, exc)
                    break
            if 'last_box' in locals() and last_box is not None:
                x,y,w,h = last_box
                cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 44), (25, 39, 54), -1)
            cv2.putText(frame, display, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (240,245,245), 2)
            cv2.imshow("FaceAuth local webcam (Q/ESC: exit)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
