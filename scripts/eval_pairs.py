"""Evaluate consensual image pairs. Calibrate ONLY on val, never on test.

CSV: image_a,image_b,same
same: 1 for identical person, 0 for different persons.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from faceauth.metrics import auc_rank, approximate_eer, choose_threshold_on_validation, rates_at
from faceauth.vision import FaceEngine, FaceInputError, cosine_similarity, read_photo_file


def safe_image_path(root: Path, name: str) -> Path:
    candidate = (root / name).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"图片路径不得超出图片根目录：{name}") from exc
    return candidate


def get_pairs(csv_path: Path) -> list[tuple[str, str, int]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not {"image_a", "image_b", "same"}.issubset(reader.fieldnames):
            raise ValueError("CSV 必须包含 image_a,image_b,same 三列")
        pairs = []
        for row in reader:
            try:
                label = int(row["same"])
                if label not in (0, 1):
                    raise ValueError
            except (ValueError, TypeError) as exc:
                raise ValueError(f"same 必须为 0 或 1：{row}") from exc
            if not row["image_a"] or not row["image_b"]:
                raise ValueError("不能有空图片路径")
            pairs.append((row["image_a"], row["image_b"], label))
    if not pairs:
        raise ValueError("CSV 不包含评估配对")
    return pairs



def classify_failure(message: str) -> str:
    """Assign stable reason categories to evaluation failures."""
    if "\u672a\u68c0\u6d4b\u5230\u4eba\u8138" in message:
        return "no_face_detected"
    if "\u68c0\u6d4b\u5230" in message and "\u5f20\u4eba\u8138" in message:
        return "multiple_faces_detected"
    if "\u4eba\u8138\u8fc7\u5c0f" in message:
        return "face_too_small"
    if "\u6a21\u7cca" in message:
        return "face_blurry"
    if "\u56fe\u7247\u4e0d\u5b58\u5728" in message or "No such file" in message:
        return "missing_image"
    if "\u56fe\u7247\u65e0\u6cd5\u89e3\u6790" in message:
        return "bad_image_decode"
    return "other"


def summarize_failures(failures: list[dict], pair_total: int) -> dict:
    return {
        "requested_pairs": pair_total,
        "skipped_pairs": len(failures),
        "valid_pairs": pair_total - len(failures),
        "valid_pair_rate": (pair_total - len(failures)) / pair_total if pair_total else None,
        "failure_by_reason": dict(sorted(Counter(f["reason_code"] for f in failures).items())),
        "failure_by_label": {
            "positive": sum(int(f["label"] == 1) for f in failures),
            "negative": sum(int(f["label"] == 0) for f in failures),
        },
    }


def overall_reject_on_failure(valid_metrics: dict, failures: list[dict]) -> dict:
    """Simulate operational verification when feature extraction failure means reject.

    This is a 1:1 pair-level measurement, NOT open-set 1:N identification.
    """
    skipped_pos = sum(1 for f in failures if f["label"] == 1)
    skipped_neg = sum(1 for f in failures if f["label"] == 0)
    tp = int(valid_metrics["tp"])
    tn = int(valid_metrics["tn"]) + skipped_neg
    fp = int(valid_metrics["fp"])
    fn = int(valid_metrics["fn"]) + skipped_pos
    total = tp + tn + fp + fn
    n_pos = tp + fn
    n_neg = tn + fp
    return {
        "policy": "detection_or_embedding_failure_is_authentication_rejection",
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "total_pairs": total,
        "accuracy": (tp + tn) / total if total else None,
        "tar": tp / n_pos if n_pos else None,
        "far": fp / n_neg if n_neg else None,
        "frr": fn / n_pos if n_pos else None,
        "positives": n_pos, "negatives": n_neg,
    }


def evaluate_images(pairs, root: Path, engine: FaceEngine, *,
                    face_selection: str = "strict", image_audit: dict | None = None):
    cache = {}
    failed_cache = {}
    multi_resolved = set()
    score_list = []
    label_list = []
    failures = []
    for idx, (name_a, name_b, label) in enumerate(pairs, start=1):
        try:
            for name in (name_a, name_b):
                if name in failed_cache:
                    raise FaceInputError(failed_cache[name])
                if name not in cache:
                    try:
                        photo = read_photo_file(safe_image_path(root, name))
                        if face_selection == "strict":
                            observation = engine.extract(photo, annotate=False)
                        else:
                            observation = engine.extract(photo, annotate=False,
                                                         selection_policy=face_selection)
                        cache[name] = observation.embedding
                        if getattr(observation, "detected_face_count", 1) > 1:
                            multi_resolved.add(name)
                    except (FaceInputError, OSError, RuntimeError, ValueError) as exc:
                        failed_cache[name] = str(exc)
                        raise
            sim = cosine_similarity(cache[name_a], cache[name_b])
            score_list.append(sim)
            label_list.append(label)
        except (FaceInputError, OSError, RuntimeError, ValueError) as exc:
            failures.append({
                "row": idx,
                "label": int(label),
                "reason_code": classify_failure(str(exc)),
                "reason": str(exc)[:180],
            })
        if idx % 50 == 0 or idx == len(pairs):
            print(f"Pairs processed: {idx}/{len(pairs)} | valid={len(score_list)} | skipped={len(failures)}")
    if image_audit is not None:
        image_audit.update({
            "unique_images_with_embeddings": len(cache),
            "unique_images_failed": len(failed_cache),
            "unique_images_with_multiple_yunet_boxes_selected": len(multi_resolved),
        })
    return score_list, label_list, failures


def main():
    parser = argparse.ArgumentParser(description="Face pair ROC/FAR/FRR evaluation (local, consented images)")
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--images", type=Path, default=None, help="Base directory for image paths; default CSV parent")
    parser.add_argument("--split", choices=["val", "test"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--calibration-out", type=Path, help="Validation ONLY: write chosen FAR-constrained threshold")
    parser.add_argument("--threshold-file", type=Path, help="Test: fixed threshold JSON from validation")
    parser.add_argument("--target-far", type=float, default=0.01)
    parser.add_argument("--eval-yunet-score", type=float, default=0.8,
                        help="YuNet detector score threshold for offline evaluation only; default 0.8")
    parser.add_argument("--eval-min-face-size", type=int, default=55,
                        help="Min detected face box in pixels for offline evaluation; default 55")
    parser.add_argument("--face-selection", choices=["strict", "lfw-largest"], default="strict",
                        help="Offline LFW only: strict rejects multiple boxes; lfw-largest selects largest detected box")
    args = parser.parse_args()
    if args.split == "test" and (args.calibration_out or not args.threshold_file):
        parser.error("测试集必须提供 --threshold-file，且不允许 --calibration-out")
    if args.split == "val" and args.threshold_file:
        parser.error("验证集不可使用 --threshold-file，请明确进行校准")
    if args.split == "val" and not args.calibration_out:
        parser.error("验证集请提供 --calibration-out 以保存所选阈值")
    if not 0.0 < args.eval_yunet_score <= 1.0:
        parser.error("--eval-yunet-score must be in (0, 1]")
    if not 1 <= args.eval_min_face_size <= 1000:
        parser.error("--eval-min-face-size must be in [1, 1000]")
    evaluation_detector = {
        "yunet_score_threshold": float(args.eval_yunet_score),
        "min_face_size_px": int(args.eval_min_face_size),
    }
    if args.face_selection != "strict":
        # Add only for non-default policy to remain compatible with v1.2 strict calibrations.
        evaluation_detector["face_selection_policy"] = args.face_selection
        if "lfw_funneled" not in str(args.images or args.csv.parent):
            parser.error("lfw-largest is reserved for evaluation on the LFW funneled crop dataset")
    pair_list = get_pairs(args.csv)
    engine = FaceEngine(score_threshold=args.eval_yunet_score, min_face_size=args.eval_min_face_size)
    base_dir = args.images if args.images else args.csv.parent
    start = time.perf_counter()
    audit = {}
    scores, labels, skipped = evaluate_images(pair_list, base_dir, engine,
                                               face_selection=args.face_selection, image_audit=audit)
    if not scores:
        raise RuntimeError("没有成功处理的配对，无法计算指标")
    if args.split == "val":
        calibration = choose_threshold_on_validation(scores, labels, target_far=args.target_far)
        threshold = float(calibration["threshold"])
        args.calibration_out.parent.mkdir(parents=True, exist_ok=True)
        args.calibration_out.write_text(json.dumps({
            "threshold": threshold, "target_far": args.target_far,
            "trained_or_calibrated_on": "val", "model_id": engine.model_id,
            "positive_pairs": calibration["positive_pairs"],
            "negative_pairs": calibration["negative_pairs"],
            "evaluation_detector": evaluation_detector,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    else:
        calibration = json.loads(args.threshold_file.read_text(encoding="utf-8"))
        if calibration.get("trained_or_calibrated_on") != "val" or calibration.get("model_id") != engine.model_id:
            raise RuntimeError("必须使用此 SFace 模型在 val 划分上选择的阈值")
        expected_detector = calibration.get("evaluation_detector")
        if expected_detector is None:
            if evaluation_detector != {"yunet_score_threshold": 0.8, "min_face_size_px": 55}:
                raise RuntimeError("Legacy threshold JSON assumes YuNet=0.8 / min_face=55; recalibrate on val")
        elif expected_detector != evaluation_detector:
            raise RuntimeError("Test detector settings must match the val calibration JSON")
        threshold = float(calibration["threshold"])
    metrics = rates_at(scores, labels, threshold)
    diagnostic = summarize_failures(skipped, len(pair_list))
    end_to_end = overall_reject_on_failure(metrics, skipped)
    metrics.update({
        "split": args.split,
        "roc_auc": auc_rank(scores, labels),
        "eer_info_descriptive_only": approximate_eer(scores, labels),
        "pairs_requested": len(pair_list), "pairs_skipped": len(skipped),
        "skipped_examples": skipped[:20],
        "coverage_diagnostics": diagnostic,
        "end_to_end_reject_on_failure": end_to_end,
        "evaluation_detector": evaluation_detector,
        "image_selection_audit": audit,
        "elapsed_s": round(time.perf_counter()-start, 3),
        "calibration_source": "val",
        "warning": "Local biometric demo only. Calibrate on representative independent data; do not use test to tune threshold.",
    })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for name in ("accuracy", "tar", "far", "frr", "roc_auc"):
        value = metrics[name]
        print(f"{name:>12}: {value:.4f}" if value is not None else f"{name:>12}: N/A")
    print(f"Threshold: {threshold:.4f}; valid pairs={len(scores)}; skipped={len(skipped)}")
    print(f"Coverage: {diagnostic['valid_pair_rate']:.2%} | failure reasons: {diagnostic['failure_by_reason']}")
    print(f"Failed positives/negatives: {diagnostic['failure_by_label']}")
    print(f"End-to-end (reject on detection failure): Accuracy={end_to_end['accuracy']:.4f}, "
          f"TAR={end_to_end['tar']:.4f}, FAR={end_to_end['far']:.4f}, FRR={end_to_end['frr']:.4f}")
    print(f"Face selection: {args.face_selection} | image audit: {audit}")
    if args.face_selection != "strict":
        print("WARNING: lfw-largest is for LFW face-crop benchmarking only, NOT live authentication.")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
