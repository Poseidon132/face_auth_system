"""Offline LFW 1:N identification evaluation, no SQLite or Streamlit changes.

Run validation to calibrate gallery-size-specific thresholds, then run a
completely identity-disjoint test split with those saved thresholds.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from faceauth.ident_eval import (
    FeatureCache, PROTOCOL_VERSION, build_protocol, calibrate_unknown_threshold,
    report_for_stage, scan_lfw_images, score_one_gallery, summarize_gallery,
)
from faceauth.vision import FaceEngine


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, report: dict) -> None:
    rows = report["gallery_results"]
    fields = [
        "gallery_people", "threshold", "known_query_count", "unknown_query_count",
        "known_coverage", "unknown_coverage", "closed_set_rank1_end_to_end",
        "closed_set_rank1_conditional", "dir_at_threshold", "known_wrong_id_rate",
        "known_rejection_rate", "fpir", "unknown_correct_reject_rate",
        "gallery_templates_success", "gallery_templates_failed",
        "matching_mean_ms", "matching_p95_ms", "model_plus_match_estimate_mean_ms",
        "model_plus_match_estimate_p95_ms",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for size in sorted(rows, key=int):
            result = rows[size]
            times = result["timing_ms"]
            flat = {key: result.get(key) for key in fields}
            flat.update({
                "matching_mean_ms": times["vector_match_only"]["mean_ms"],
                "matching_p95_ms": times["vector_match_only"]["p95_ms"],
                "model_plus_match_estimate_mean_ms": times["estimated_query_model_plus_match"]["mean_ms"],
                "model_plus_match_estimate_p95_ms": times["estimated_query_model_plus_match"]["p95_ms"],
            })
            writer.writerow(flat)


def write_chart_if_matplotlib(path: Path, report: dict) -> bool:
    """Optional local chart; charts only show real run metrics, never samples."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except ImportError:
        return False
    sizes = sorted(int(k) for k in report["gallery_results"])
    results = [report["gallery_results"][str(n)] for n in sizes]
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.plot(sizes, [r["closed_set_rank1_end_to_end"] * 100 for r in results], "o-", label="Closed-set Rank-1")
    ax.plot(sizes, [r["dir_at_threshold"] * 100 for r in results], "s-", label="DIR @ calibrated threshold")
    ax.plot(sizes, [r["fpir"] * 100 for r in results], "^-", label="Unknown FPIR")
    ax.set(xlabel="Registered identities", ylabel="Rate (%)",
           title=f"LFW custom 1:N {report['stage']} / identity-disjoint split")
    ax.set_xticks(sizes)
    ax.set_ylim(bottom=0, top=105)
    ax.grid(True, alpha=.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    fig2, ax2 = plt.subplots(figsize=(7.4, 4.2))
    ax2.plot(sizes, [r["timing_ms"]["vector_match_only"]["mean_ms"] for r in results],
             "o-", label="Exact cosine search only")
    ax2.plot(sizes, [r["timing_ms"]["estimated_query_model_plus_match"]["mean_ms"]
                     for r in results], "s-", label="Model + search (estimate)")
    ax2.set(xlabel="Registered identities", ylabel="Mean time (ms)",
            title=f"LFW custom 1:N {report['stage']} / offline per-query timing")
    ax2.set_xticks(sizes)
    ax2.set_ylim(bottom=0)
    ax2.grid(True, alpha=.25)
    ax2.legend(loc="best", fontsize=8)
    fig2.tight_layout()
    fig2.savefig(path.with_name(path.stem + "_timing.png"), dpi=160)
    plt.close(fig2)
    return True


def validate_fixed_file(cal: dict, protocol, *, model_id: str, policy: str,
                        detector: dict) -> None:
    expected = {
        "protocol": PROTOCOL_VERSION,
        "stage": "val",
        "protocol_fingerprint": protocol.protocol_fingerprint,
        "model_id": model_id,
        "face_selection": policy,
        "detector": detector,
        "gallery_sizes": list(protocol.gallery_sizes),
    }
    for k, v in expected.items():
        if cal.get(k) != v:
            raise ValueError(f"校准文件与当前测试配置不匹配：{k}。请使用相同参数重新运行验证集。")
    if not isinstance(cal.get("target_fpir"), (int, float)) or not 0 <= cal["target_fpir"] < 1:
        raise ValueError("校准文件缺少有效 target_fpir")
    expected_keys = {str(n) for n in protocol.gallery_sizes}
    if set(cal.get("per_gallery", {})) != expected_keys:
        raise ValueError("校准文件缺少与库规模对应的阈值")
    for v in cal["per_gallery"].values():
        t = v.get("threshold")
        if not isinstance(t, (int, float)) or not math.isfinite(t) or not (0 <= t <= 1):
            raise ValueError("校准文件阈值无效")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LFW custom identity-disjoint 1:N open-set evaluation")
    parser.add_argument("--stage", choices=("val", "test"), required=True)
    parser.add_argument("--images", type=Path, default=ROOT / "data/lfw_home/lfw_funneled")
    parser.add_argument("--sizes", nargs="+", type=int, default=[10, 50, 100])
    parser.add_argument("--enroll-per-person", type=int, default=2)
    parser.add_argument("--queries-per-person", type=int, default=2)
    parser.add_argument("--unknown-queries", type=int, default=300,
                        help="Number of distinct unknown identities/queries PER split, reused for 10/50/100")
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--target-fpir", type=float, default=.01)
    parser.add_argument("--face-selection", choices=("lfw-largest", "strict"), default="lfw-largest")
    parser.add_argument("--yunet-score", type=float, default=.8)
    parser.add_argument("--min-face-size", type=int, default=55)
    parser.add_argument("--calibration-out", type=Path)
    parser.add_argument("--threshold-file", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--no-chart", action="store_true", help="Do not attempt optional matplotlib PNG")
    return parser


def run(args: argparse.Namespace) -> dict:
    parser = make_parser()
    if args.stage == "val" and (not args.calibration_out or args.threshold_file):
        parser.error("验证集必须指定 --calibration-out，不能指定 --threshold-file")
    if args.stage == "test" and (not args.threshold_file or args.calibration_out):
        parser.error("测试集必须指定 --threshold-file，不能指定 --calibration-out")
    if not 0 <= args.target_fpir < 1 or not .0 < args.yunet_score <= 1 or not 1 <= args.min_face_size <= 1000:
        parser.error("--target-fpir 必须在 [0,1) 内，YuNet score 在 (0,1]，人脸尺寸在 [1,1000]")
    if args.face_selection == "lfw-largest" and args.images.resolve().name != "lfw_funneled":
        parser.error("lfw-largest 只允许用于 LFW 已裁剪的人脸图片目录 lfw_funneled")
    inventory = scan_lfw_images(args.images)
    protocol = build_protocol(
        inventory, gallery_sizes=args.sizes, enroll_per_identity=args.enroll_per_person,
        queries_per_identity=args.queries_per_person,
        unknown_queries=args.unknown_queries, seed=args.seed,
    )
    detector = {"yunet_score": args.yunet_score, "min_face_size": args.min_face_size}
    engine = FaceEngine(score_threshold=args.yunet_score, min_face_size=args.min_face_size)
    cal = None
    if args.stage == "test":
        cal = json.loads(args.threshold_file.read_text(encoding="utf-8"))
        validate_fixed_file(cal, protocol, model_id=engine.model_id, policy=args.face_selection, detector=detector)
        if abs(float(cal["target_fpir"]) - args.target_fpir) > 1e-12:
            raise ValueError("--target-fpir 与验证集校准文件不同。请使用验证集原参数。")
    chosen_known = protocol.known_val if args.stage == "val" else protocol.known_test
    chosen_unknown = protocol.unknown_val if args.stage == "val" else protocol.unknown_test
    print(f"LFW 1:N [{args.stage}] identity-disjoint protocol, seed={protocol.seed}; gallery sizes={protocol.gallery_sizes}")
    print(f"Known gallery max={len(chosen_known)}, unknown queries={len(chosen_unknown)}; face selection={args.face_selection}")
    features = FeatureCache(engine, args.images, policy=args.face_selection)
    outcomes: dict[str, dict] = {}
    calibration = {}
    for n in protocol.gallery_sizes:
        raw = score_one_gallery(chosen_known, chosen_unknown, n, features)
        if args.stage == "val":
            unknown_scores = [q["score"] for q in raw["unknown_queries"] if q["valid"]]
            choice = calibrate_unknown_threshold(unknown_scores, args.target_fpir)
            calibration[str(n)] = choice
            threshold = choice["threshold"]
        else:
            threshold = float(cal["per_gallery"][str(n)]["threshold"])
        row = summarize_gallery(raw, threshold)
        outcomes[str(n)] = row
        print(
            f"Gallery {n:3d}: known={row['known_query_count']} (coverage={row['known_coverage']:.1%}) "
            f"unknown={row['unknown_query_count']} (coverage={row['unknown_coverage']:.1%}) "
            f"Rank1={row['closed_set_rank1_end_to_end']:.2%}, DIR={row['dir_at_threshold']:.2%}, "
            f"FPIR={row['fpir']:.2%}, threshold={threshold:.6f}, "
            f"match mean={row['timing_ms']['vector_match_only']['mean_ms']:.3f} ms",
            flush=True,
        )
    if args.stage == "val":
        write_json(args.calibration_out, {
            "protocol": PROTOCOL_VERSION, "stage": "val",
            "protocol_fingerprint": protocol.protocol_fingerprint,
            "model_id": engine.model_id, "face_selection": args.face_selection,
            "detector": detector, "gallery_sizes": list(protocol.gallery_sizes),
            "target_fpir": args.target_fpir, "per_gallery": calibration,
        })
        print(f"Validation thresholds saved: {args.calibration_out}")
    report = report_for_stage(protocol, args.stage, outcomes, model_id=engine.model_id,
                              face_selection=args.face_selection, target_fpir=args.target_fpir)
    report["detector"] = detector
    report["unique_images_processed"] = len(features.items)
    report["unique_images_failed"] = sum(v["feature"] is None for v in features.items.values())
    write_json(args.out, report)
    write_csv(args.out.with_suffix(".csv"), report)
    if not args.no_chart:
        if write_chart_if_matplotlib(args.out.with_suffix(".png"), report):
            print(f"Charts saved: {args.out.with_suffix('.png')} and timing PNG")
        else:
            print("Chart skipped: matplotlib not installed. Optional: python -m pip install matplotlib")
    print(f"Report saved: {args.out} + {args.out.with_suffix('.csv')}")
    print("NOT official LFW 10-fold and NOT the Web strict multi-face verification policy.")
    return report


def main() -> None:
    run(make_parser().parse_args())


if __name__ == "__main__":
    main()
