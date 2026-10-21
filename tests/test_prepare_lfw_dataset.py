"""Offline fixtures only; test does not contain or download human face photos."""
import csv
import io
import tarfile
from pathlib import Path

import pytest
from scripts.prepare_lfw_dataset import (
    _safe_member_destination, balanced_preview, check_images_exist,
    lfw_image_rel, parse_official_pairs, safe_unpack_lfw, save_csv,
)


def test_pair_conversion():
    records = parse_official_pairs("2\nFirst_Person\t1\t2\nFirst_Person\t1\tSecond_Person\t1\n")
    assert records == [
        ("First_Person/First_Person_0001.jpg", "First_Person/First_Person_0002.jpg", 1),
        ("First_Person/First_Person_0001.jpg", "Second_Person/Second_Person_0001.jpg", 0),
    ]


def test_path_names_restricted():
    with pytest.raises(ValueError):
        lfw_image_rel("../untrusted", "1")
    with pytest.raises(ValueError):
        lfw_image_rel("Person", "10000")
    with pytest.raises(ValueError):
        _safe_member_destination(Path("/tmp"), "../../etc/passwd")
    with pytest.raises(ValueError):
        _safe_member_destination(Path("/tmp"), "outside.txt")


def test_missing_pair_images(tmp_path):
    records = parse_official_pairs("1\nA\t1\t2\nA\t1\tB\t1")
    with pytest.raises(FileNotFoundError):
        check_images_exist(records, tmp_path)
    for name in ("A/A_0001.jpg", "A/A_0002.jpg", "B/B_0001.jpg"):
        path = tmp_path/name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"synthetic")
    check_images_exist(records, tmp_path)


def test_save_csv(tmp_path):
    records = parse_official_pairs("1\nA\t1\t2\nA\t1\tB\t1")
    output = tmp_path/"pairs.csv"
    save_csv(output, records)
    with output.open(encoding="utf-8", newline="") as handle:
        lines = list(csv.DictReader(handle))
    assert len(lines) == 2 and lines[1]["same"] == "0"


def test_preview_balancing():
    items = [("a", "b", 1)]*3 + [("a", "c", 0)]*3
    rows = balanced_preview(items, per_label=2)
    assert [r[2] for r in rows] == [1, 0, 1, 0]


def test_tar_traversal_rejection(tmp_path):
    archive = tmp_path/"fake.tgz"
    with tarfile.open(archive, "w:gz") as tar:
        data = b"malicious"
        info = tarfile.TarInfo("../../bad.txt")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    with pytest.raises(ValueError):
        safe_unpack_lfw(archive, tmp_path/"extract")
    assert not (tmp_path/"bad.txt").exists()


def test_tar_symlink_rejection(tmp_path):
    archive = tmp_path/"fake.tgz"
    with tarfile.open(archive, "w:gz") as tar:
        info = tarfile.TarInfo("lfw_funneled/evil")
        info.type = tarfile.SYMTYPE
        info.linkname = "/tmp/evil"
        tar.addfile(info)
    with pytest.raises(ValueError):
        safe_unpack_lfw(archive, tmp_path/"extract")
