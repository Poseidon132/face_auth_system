"""Download official pretrained YuNet/SFace ONNX weights, verify pinned SHA-256."""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from faceauth.config import MODEL_DIR, MODELS


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validated(path: Path, expected: str) -> bool:
    return path.is_file() and path.stat().st_size > 100_000 and sha256(path) == expected


def fetch(url: str, target: Path, expected: str, max_size: int = 60 * 1024 * 1024) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "face-auth-educational-demo/1.0"})
    temp = target.with_name(target.name + ".partial")
    try:
        digest = hashlib.sha256()
        total = 0
        with urllib.request.urlopen(req, timeout=45) as response, temp.open("wb") as outfile:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_size:
                    raise ValueError("下载文件异常，已超过允许大小")
                digest.update(chunk)
                outfile.write(chunk)
        if total < 100_000 or digest.hexdigest() != expected:
            raise ValueError(f"模型文件校验失败（{total} bytes）。可能下载到了 HTML 或 LFS 指针。")
        temp.replace(target)
    finally:
        temp.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Official OpenCV Zoo YuNet/SFace model downloader")
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--from-local", type=Path, help="Use previously downloaded model files from a folder")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    args.model_dir.mkdir(parents=True, exist_ok=True)
    for model in MODELS:
        target = args.model_dir / model.filename
        if validated(target, model.sha256):
            print(f"OK (cached): {model.filename}")
            continue
        if args.verify_only:
            print(f"MISSING OR INVALID: {model.filename}")
            return 1
        if target.exists():
            print(f"Invalid existing file (not trusted): {target}")
            target.unlink()
        if args.from_local:
            src = args.from_local / model.filename
            if not validated(src, model.sha256):
                print(f"Local model missing or checksum mismatch: {src}")
                return 1
            shutil.copy2(src, target)
            print(f"OK (copied): {model.filename}")
            continue
        errors = []
        for index, url in enumerate(model.sources, start=1):
            try:
                print(f"Downloading {model.filename} from official source {index}/{len(model.sources)} ...", flush=True)
                fetch(url, target, model.sha256)
                print(f"OK: {model.filename} ({target.stat().st_size / 1024**2:.2f} MiB)")
                break
            except Exception as exc:
                errors.append(f"  {url}: {exc}")
                print(f"Source {index} failed: {type(exc).__name__}: {exc}")
        else:
            print("所有下载源均失败。请手动从 README 提供的官方地址下载两个 ONNX 到 models/，然后运行 --verify-only。")
            for line in errors:
                print(line)
            return 2
    print("SUCCESS: model files are available and SHA-256 verified. Run python scripts/verify_env.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
