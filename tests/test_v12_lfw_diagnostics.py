"""Regression coverage for v1.2 reporting without real face images."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from faceauth.vision import FaceInputError
from scripts import eval_pairs


def test_failure_reason_categories():
    assert eval_pairs.classify_failure("\u672a\u68c0\u6d4b\u5230\u4eba\u8138") == "no_face_detected"
    assert eval_pairs.classify_failure("\u68c0\u6d4b\u5230 2 \u5f20\u4eba\u8138") == "multiple_faces_detected"
    assert eval_pairs.classify_failure("\u4eba\u8138\u8fc7\u5c0f (40x40)") == "face_too_small"
    assert eval_pairs.classify_failure("\u56fe\u7247\u4e0d\u5b58\u5728") == "missing_image"


def test_failure_coverage_and_reject_policy():
    failures = [
        {"row": 2, "label": 1, "reason_code": "no_face_detected"},
        {"row": 4, "label": 0, "reason_code": "no_face_detected"},
    ]
    diagnostic = eval_pairs.summarize_failures(failures, pair_total=4)
    assert diagnostic["valid_pairs"] == 2
    assert diagnostic["valid_pair_rate"] == .5
    assert diagnostic["failure_by_label"] == {"positive": 1, "negative": 1}
    metrics = {"tp": 1, "tn": 1, "fp": 0, "fn": 0}
    all_pairs = eval_pairs.overall_reject_on_failure(metrics, failures)
    assert all_pairs["accuracy"] == .75
    assert all_pairs["tar"] == .5
    assert all_pairs["far"] == 0
    assert all_pairs["frr"] == .5
    assert (all_pairs["tp"], all_pairs["tn"], all_pairs["fp"], all_pairs["fn"]) == (1, 2, 0, 1)


def test_evaluation_returns_failure_label_and_reason(monkeypatch, tmp_path):
    monkeypatch.setattr(eval_pairs, "read_photo_file", lambda filename: Path(filename).stem)

    class FakeEngine:
        def extract(self, photo, annotate=False):
            if photo == "bad":
                raise FaceInputError("\u672a\u68c0\u6d4b\u5230\u4eba\u8138")
            embedding = np.array([1., 0.] + [0.] * 14 if photo == "a" else [0., 1.] + [0.] * 14, dtype=np.float32)
            return SimpleNamespace(embedding=embedding)

    pairs = [("a.jpg", "a.jpg", 1), ("a.jpg", "bad.jpg", 1),
             ("a.jpg", "b.jpg", 0), ("b.jpg", "bad.jpg", 0)]
    scores, labels, failures = eval_pairs.evaluate_images(pairs, tmp_path, FakeEngine())
    assert scores == [1., 0.]
    assert labels == [1, 0]
    assert len(failures) == 2
    assert failures[0]["label"] == 1 and failures[1]["label"] == 0
    assert failures[0]["reason_code"] == "no_face_detected"
