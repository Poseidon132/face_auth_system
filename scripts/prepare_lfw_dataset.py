"""Download LFW funneled JPEGs and produce pair CSVs compatible with eval_pairs.py.

Source: Labeled Faces in the Wild (UMass), mirrored by Figshare.
Checksums and URLs cross-checked against scikit-learn's _lfw.py dataset loader.

No face photos or biometric features are included in this source distribution.
The source images retain their original public-dataset usage restrictions.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
import tarfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
DATA_HOME = ROOT / "data" / "lfw_home"
PAIR_HOME = ROOT / "data" / "lfw_pairs"


@dataclass(frozen=True)
class Source:
    name: str
    url: str
    sha256: str
    size_limit: int = 350 * 1024 * 1024


# Authoritative URLs and SHA-256 values from scikit-learn's LFW loader.
SOURCES = (
    Source("lfw-funneled.tgz", "https://ndownloader.figshare.com/files/5976015", "b47c8422c8cded889dc5a13418c4bc2abbda121092b3533a83306f90d900100a"),
    Source("pairsDevTrain.txt", "https://ndownloader.figshare.com/files/5976012", "1d454dada7dfeca0e7eab6f65dc4e97a6312d44cf142207be28d688be92aabfa", size_limit=1024*1024),
    Source("pairsDevTest.txt", "https://ndownloader.figshare.com/files/5976009", "7cb06600ea8b2814ac26e946201cdb304296262aad67d046a16a7ec85d0ff87c", size_limit=1024*1024),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as src:
        for chunk in iter(lambda: src.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_hash(path: Path, info: Source) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"缺少文件: {path.name}")
    got = sha256(path)
    if got != info.sha256:
        raise ValueError(f"SHA-256 错误: {path.name}; actual={got}; expected={info.sha256}. 请删除损坏文件并重新下载。")


def download_file(info: Source, dest: Path, *, no_download: bool) -> None:
    if dest.is_file():
        require_hash(dest, info)
        print(f"Verified existing file: {dest.name}", flush=True)
        return
    if no_download:
        raise FileNotFoundError(
            f"缺少 {dest}. 请手动从 {info.url} 下载并保存成 {info.name}，或重新运行时去掉 --no-download。"
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    if part.exists():
        part.unlink()
    errors = []
    for attempt in range(1, 4):
        try:
            print(f"Downloading {info.name} (attempt {attempt}/3): {info.url}", flush=True)
            request = urllib.request.Request(info.url, headers={"User-Agent": "FaceAuth-LFW-Prep/1.0"})
            written = 0
            with urllib.request.urlopen(request, timeout=60) as response, part.open("wb") as output:
                declared = response.headers.get("Content-Length")
                if declared and int(declared) > info.size_limit:
                    raise ValueError(f"下载文件超过预期大小: {info.name}")
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > info.size_limit:
                        raise ValueError(f"下载超过允许大小: {info.name}")
                    output.write(chunk)
                    if written // (20 * 1024 * 1024) != (written - len(chunk)) // (20 * 1024 * 1024):
                        print(f"  downloaded {written / 1024 / 1024:.0f} MiB", flush=True)
            require_hash(part, info)
            os.replace(part, dest)
            print(f"Downloaded and verified: {dest.name} ({written:,} bytes)", flush=True)
            return
        except (OSError, urllib.error.URLError, ValueError) as exc:
            errors.append(str(exc))
            if part.exists():
                part.unlink()
            if attempt < 3:
                time.sleep(attempt)
    raise RuntimeError(f"无法下载 {info.name}: {errors[-1]}\n请使用浏览器打开 {info.url}，将文件存入 {dest.parent}，然后执行 --no-download。")


def _safe_member_destination(destination: Path, member_name: str) -> Path:
    # LFW archive should contain only the lfw_funneled/ tree. Never follow links.
    if "\\" in member_name or "\x00" in member_name:
        raise ValueError(f"不安全的 tar 成员名称: {member_name!r}")
    parts = PurePosixPath(member_name).parts
    if not parts or parts[0] != "lfw_funneled" or ".." in parts or PurePosixPath(member_name).is_absolute():
        raise ValueError(f"不安全的 tar 成员路径: {member_name!r}")
    result = destination.joinpath(*parts)
    if not result.resolve().is_relative_to(destination.resolve()):
        raise ValueError(f"tar 成员越界: {member_name!r}")
    return result


def safe_unpack_lfw(archive_path: Path, home: Path) -> None:
    """Extract only regular members under lfw_funneled/, rejecting links/traversal."""
    images = home / "lfw_funneled"
    if images.is_dir() and any(images.iterdir()):
        print(f"Using existing extracted images: {images}", flush=True)
        return
    home.mkdir(parents=True, exist_ok=True)
    count, total = 0, 0
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            if not (member.isfile() or member.isdir()):
                raise ValueError(f"归档文件包含不允许的链接或特殊文件: {member.name!r}")
            target = _safe_member_destination(home, member.name)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if member.size < 0 or member.size > 15 * 1024 * 1024:
                raise ValueError(f"图片大小异常: {member.name!r}")
            total += member.size
            if total > 900 * 1024 * 1024:
                raise ValueError("归档展开大小超出限制")
            target.parent.mkdir(parents=True, exist_ok=True)
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"tar 内容缺失: {member.name!r}")
            with stream, target.open("wb") as outfile:
                shutil.copyfileobj(stream, outfile, length=1024*1024)
            count += 1
    if count < 10000:
        raise ValueError(f"解压图片数量不足: {count}")
    print(f"Extracted {count} files to {images}", flush=True)


PERSON_NAME = re.compile(r"^[A-Za-z0-9_'-]+$")


def lfw_image_rel(person: str, number: str) -> str:
    if not PERSON_NAME.fullmatch(person):
        raise ValueError(f"身份名称不安全: {person!r}")
    index = int(number)
    if not (1 <= index <= 9999):
        raise ValueError(f"照片序号越界: {number!r}")
    return f"{person}/{person}_{index:04d}.jpg"


def parse_official_pairs(text: str) -> list[tuple[str, str, int]]:
    """LFW format: 'person i j' or 'person_a i person_b j', 1-based indices."""
    records = []
    for line_num, raw in enumerate(text.splitlines(), 1):
        cols = raw.strip().split()
        if not cols:
            continue
        if len(cols) == 1 and cols[0].isdigit():
            continue  # LFW pair-count header
        if len(cols) == 3:
            person, a, b = cols
            records.append((lfw_image_rel(person, a), lfw_image_rel(person, b), 1))
        elif len(cols) == 4:
            person_a, a, person_b, b = cols
            records.append((lfw_image_rel(person_a, a), lfw_image_rel(person_b, b), 0))
        else:
            raise ValueError(f"pairs 文件第 {line_num} 行格式错误: {raw[:120]}")
    if not records or not {v[2] for v in records} == {0, 1}:
        raise ValueError("正负图文对必须均非空")
    return records


def check_images_exist(records: list[tuple[str, str, int]], images_root: Path) -> None:
    missing = []
    for a, b, _ in records:
        for rel in (a, b):
            if not (images_root / rel).is_file():
                missing.append(rel)
                if len(missing) >= 10:
                    break
        if len(missing) >= 10:
            break
    if missing:
        raise FileNotFoundError(f"数据集图片缺失，示例: {missing}")


def save_csv(path: Path, records: list[tuple[str, str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_a", "image_b", "same"])
        writer.writerows(records)


def balanced_preview(records: list[tuple[str, str, int]], per_label: int = 15) -> list[tuple[str, str, int]]:
    positives = [rec for rec in records if rec[2] == 1][:per_label]
    negatives = [rec for rec in records if rec[2] == 0][:per_label]
    if len(positives) < per_label or len(negatives) < per_label:
        raise ValueError("无法构建平衡的调试小样本")
    result = []
    for pos, neg in zip(positives, negatives):
        result.extend([pos, neg])
    return result


def prepare(data_home: Path = DATA_HOME, pairs_home: Path = PAIR_HOME, *, no_download: bool = False) -> dict:
    data_home = data_home.resolve()
    pairs_home = pairs_home.resolve()
    images_root = data_home / "lfw_funneled"
    data_home.mkdir(parents=True, exist_ok=True)
    for source in SOURCES:
        # If images were previously extracted, avoid a second 200MB download.
        if source.name == "lfw-funneled.tgz" and images_root.is_dir() and any(images_root.iterdir()):
            print(f"Images already extracted: {images_root}", flush=True)
            continue
        download_file(source, data_home / source.name, no_download=no_download)
    if not images_root.is_dir() or not any(images_root.iterdir()):
        safe_unpack_lfw(data_home / SOURCES[0].name, data_home)
    summary = {}
    for split, filename in (("val", "pairsDevTrain.txt"), ("test", "pairsDevTest.txt")):
        official = (data_home / filename).read_text(encoding="utf-8")
        records = parse_official_pairs(official)
        check_images_exist(records, images_root)
        save_csv(pairs_home / f"{split}_pairs.csv", records)
        positives = sum(r[2] for r in records)
        summary[split] = {"pairs": len(records), "positive": positives, "negative": len(records) - positives}
        print(f"{split}: {len(records)} pairs ({positives} positive / {len(records)-positives} negative)", flush=True)
        if split == "val":
            save_csv(pairs_home / "debug_val_30_pairs.csv", balanced_preview(records))
    summary.update({
        "image_directory": str(images_root),
        "pair_directory": str(pairs_home),
        "dataset": "LFW funneled images and official development pair lists",
        "source_urls": {f.name: f.url for f in SOURCES},
        "note": "Development pair test is not the standard LFW 10-fold benchmark. Pretrained SFace is used without training.",
    })
    pairs_home.mkdir(parents=True, exist_ok=True)
    (pairs_home / "manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print("DONE: dataset prepared; no model was trained or evaluated.", flush=True)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare LFW full-resolution JPEGs and val/test pair CSVs for FaceAuth")
    parser.add_argument("--no-download", action="store_true", help="Only use downloaded files already present in data/lfw_home")
    parser.add_argument("--data-home", type=Path, default=DATA_HOME)
    parser.add_argument("--pairs-home", type=Path, default=PAIR_HOME)
    args = parser.parse_args()
    try:
        prepare(args.data_home, args.pairs_home, no_download=args.no_download)
    except (OSError, ValueError, RuntimeError, tarfile.TarError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
