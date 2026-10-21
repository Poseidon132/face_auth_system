"""Identity-disjoint, offline LFW 1:N closed/open-set evaluation helpers.

No enrollment store is touched. No photos or biometric vectors are written.
This experiment is NOT liveness testing or production authentication.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


PROTOCOL_VERSION = "lfw-identity-disjoint-1n-v1"


@dataclass(frozen=True)
class IdentitySample:
    identity: str
    enrollment: tuple[str, ...]
    queries: tuple[str, ...]


@dataclass(frozen=True)
class Protocol:
    known_val: tuple[IdentitySample, ...]
    known_test: tuple[IdentitySample, ...]
    unknown_val: tuple[str, ...]
    unknown_test: tuple[str, ...]
    gallery_sizes: tuple[int, ...]
    seed: int
    enroll_per_identity: int
    queries_per_identity: int
    unknown_queries: int
    inventory_fingerprint: str
    protocol_fingerprint: str
    all_identity_count: int
    eligible_known_count: int


def quantile_ms(numbers: list[float] | tuple[float, ...], fraction: float = .95) -> float | None:
    if not numbers:
        return None
    ordered = sorted(float(x) for x in numbers)
    idx = (len(ordered) - 1) * fraction
    floor = int(math.floor(idx))
    ceil = int(math.ceil(idx))
    return ordered[floor] + (ordered[ceil] - ordered[floor]) * (idx - floor)


def latency_summary(numbers: list[float]) -> dict:
    return {
        "samples": len(numbers),
        "mean_ms": round(statistics.fmean(numbers), 4) if numbers else None,
        "median_ms": round(float(statistics.median(numbers)), 4) if numbers else None,
        "p95_ms": round(float(quantile_ms(numbers)), 4) if numbers else None,
    }


def scan_lfw_images(root: Path) -> dict[str, tuple[str, ...]]:
    """List image paths under one-level LFW person directories, no symlinks."""
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"找不到 LFW 图片目录：{root}")
    result: dict[str, tuple[str, ...]] = {}
    for folder in sorted(root.iterdir()):
        if not folder.is_dir() or folder.is_symlink():
            continue
        names = []
        for path in sorted(folder.glob("*.jpg")):
            if path.is_symlink() or not path.is_file():
                continue
            if not path.name.startswith(folder.name + "_"):
                continue
            names.append(f"{folder.name}/{path.name}")
        if names:
            result[folder.name] = tuple(names)
    if not result:
        raise ValueError(f"LFW 目录未找到 Person/Person_0001.jpg 格式的图片：{root}")
    return result


def _fingerprint(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def _shuffle_person_photos(identity: str, paths: tuple[str, ...], seed: int) -> tuple[str, ...]:
    stable_seed = int(hashlib.sha256(f"{seed}|{identity}".encode()).hexdigest()[:16], 16)
    shuffled = list(paths)
    random.Random(stable_seed).shuffle(shuffled)
    return tuple(shuffled)


def build_protocol(inventory: Mapping[str, tuple[str, ...]], *, gallery_sizes=(10, 50, 100),
                   enroll_per_identity=2, queries_per_identity=2, unknown_queries=300,
                   seed=20261021) -> Protocol:
    """Create non-overlapping val/test galleries and disjoint unknown identities.

    Each known identity contributes distinct enrollment/query images. The first
    N known identities define an N-person gallery; gallery sizes are nested
    *within* each split. Val and test use disjoint identities, not just photos.
    """
    sizes = tuple(int(n) for n in gallery_sizes)
    if not sizes or sizes != tuple(sorted(set(sizes))) or any(n < 2 for n in sizes):
        raise ValueError("--sizes 必须是递增、无重复且至少 2 人的整数列表")
    if enroll_per_identity < 1 or queries_per_identity < 1 or unknown_queries < 2:
        raise ValueError("模板数、已知查询数须 >=1；未知查询数须 >=2")
    normalized = {str(k): tuple(sorted(set(v))) for k, v in inventory.items() if v}
    eligible = sorted(k for k, v in normalized.items() if len(v) >= enroll_per_identity + 1)
    max_n = sizes[-1]
    if len(eligible) < 2 * max_n:
        raise ValueError(f"需要至少 {2*max_n} 个身份（每人至少 {enroll_per_identity+1} 张图片），当前只有 {len(eligible)}")
    master = random.Random(seed)
    master.shuffle(eligible)
    val_ids = eligible[:max_n]
    test_ids = eligible[max_n:2*max_n]

    def make_known(ids: list[str]) -> tuple[IdentitySample, ...]:
        samples = []
        for ident in ids:
            images = _shuffle_person_photos(ident, normalized[ident], seed)
            samples.append(IdentitySample(ident, images[:enroll_per_identity],
                              images[enroll_per_identity:enroll_per_identity + queries_per_identity]))
        return tuple(samples)

    reserved = set(val_ids) | set(test_ids)
    unknown_ids = sorted(set(normalized) - reserved)
    master.shuffle(unknown_ids)
    required = 2 * unknown_queries
    if len(unknown_ids) < required:
        raise ValueError(f"需要 >= {required} 个不与已注册身份重叠的未知身份，当前只有 {len(unknown_ids)}")

    def make_unknown(ids: list[str]) -> tuple[str, ...]:
        return tuple(_shuffle_person_photos(k, normalized[k], seed)[0] for k in ids)

    inventory_hash = _fingerprint({k: normalized[k] for k in sorted(normalized)})
    protocol_inputs = {
        "version": PROTOCOL_VERSION, "inventory": inventory_hash, "sizes": sizes,
        "seed": seed, "enroll": enroll_per_identity, "queries": queries_per_identity,
        "unknown_queries": unknown_queries,
    }
    return Protocol(
        known_val=make_known(val_ids), known_test=make_known(test_ids),
        unknown_val=make_unknown(unknown_ids[:unknown_queries]),
        unknown_test=make_unknown(unknown_ids[unknown_queries:required]),
        gallery_sizes=sizes, seed=seed, enroll_per_identity=enroll_per_identity,
        queries_per_identity=queries_per_identity, unknown_queries=unknown_queries,
        inventory_fingerprint=inventory_hash,
        protocol_fingerprint=_fingerprint(protocol_inputs),
        all_identity_count=len(normalized), eligible_known_count=len(eligible),
    )


def max_template_scores(template_matrix: np.ndarray, template_identity_idx: np.ndarray,
                        n_identities: int, query: np.ndarray) -> tuple[int, float, np.ndarray]:
    """Vectorized same-person maximum over all L2-normalized SFace templates."""
    matrix = np.asarray(template_matrix, dtype=np.float32)
    owners = np.asarray(template_identity_idx, dtype=np.intp)
    feat = np.asarray(query, dtype=np.float32).reshape(-1)
    if matrix.ndim != 2 or matrix.shape[0] != len(owners) or matrix.shape[1] != len(feat):
        raise ValueError("模板与查询向量维度不兼容")
    if n_identities < 1 or not len(owners) or np.any((owners < 0) | (owners >= n_identities)):
        raise ValueError("无效的注册身份索引")
    sims = np.clip(matrix @ feat, -1.0, 1.0)
    best_scores = np.full(n_identities, -np.inf, dtype=np.float32)
    np.maximum.at(best_scores, owners, sims)
    index = int(np.argmax(best_scores))
    return index, float(best_scores[index]), best_scores


def calibrate_unknown_threshold(unknown_scores: list[float], target_fp_rate: float) -> dict:
    """Choose a strict threshold using validation UNKNOWN queries only.

    Decision: accept if similarity >= threshold. Maximum tolerated false
    accepts is floor(n * target); choosing just above sorted[k] ensures <=k.
    This controls measured val FPIR only, not the population/production rate.
    """
    if not 0 <= target_fp_rate < 1:
        raise ValueError("target-fpir 必须在 [0,1) 内")
    if len(unknown_scores) < 2 or not np.isfinite(unknown_scores).all():
        raise ValueError("阈值校准需要至少两个有效且有限的未知身份得分")
    ranked = sorted((float(v) for v in unknown_scores), reverse=True)
    allowed = math.floor(len(ranked) * target_fp_rate + 1e-10)
    threshold = max(0.0, float(np.nextafter(ranked[allowed], np.inf)))
    if threshold > 1.0:
        raise ValueError("未知身份最高分为 1.0，无法达到目标 FPIR")
    observed = sum(s >= threshold for s in ranked) / len(ranked)
    return {"threshold": threshold, "target_fp_rate": target_fp_rate,
            "val_unknown_count": len(ranked), "allowed_false_accepts": allowed,
            "observed_val_fpir": observed,
            "method": "empirical_val_unknown_max_score_strict_above_kth"}


class FeatureCache:
    """One runtime extraction per image; cache kept only in process memory."""
    def __init__(self, engine: Any, image_root: Path, policy="lfw-largest") -> None:
        if policy not in {"strict", "lfw-largest"}:
            raise ValueError("不支持的人脸选择策略")
        self.engine = engine
        self.root = Path(image_root).resolve()
        self.policy = policy
        self.items: dict[str, dict] = {}

    def get(self, relative_path: str) -> dict:
        if relative_path in self.items:
            return self.items[relative_path]
        from .vision import FaceInputError, normalize_feature, read_photo_file
        path = (self.root / relative_path).resolve()
        try:
            path.relative_to(self.root)
        except ValueError as exc:
            raise ValueError("无效 LFW 图片路径") from exc
        try:
            start = time.perf_counter()
            photo = read_photo_file(path)
            read_ms = (time.perf_counter() - start) * 1000
            start = time.perf_counter()
            obs = self.engine.extract(photo, annotate=False, selection_policy=self.policy)
            full_extract_ms = (time.perf_counter() - start) * 1000
            item = {"feature": normalize_feature(obs.embedding),
                    "detect_ms": float(obs.detect_ms), "encode_ms": float(obs.encode_ms),
                    "extract_wall_ms": full_extract_ms, "read_ms": read_ms, "error": None}
        except (FaceInputError, OSError, RuntimeError, ValueError) as exc:
            item = {"feature": None, "detect_ms": None, "encode_ms": None,
                    "extract_wall_ms": None, "read_ms": None, "error": type(exc).__name__}
        self.items[relative_path] = item
        return item


def score_one_gallery(known: tuple[IdentitySample, ...], unknown_queries: tuple[str, ...],
                      size: int, feature_cache: FeatureCache) -> dict:
    """Unthresholded evaluation; returns PRIVATE in-memory scores, not image paths."""
    selected = known[:size]
    if len(selected) != size:
        raise ValueError("缺少用于构建身份库的身份")
    template_vectors = []
    owners = []
    enrollment_failed = 0
    for idx, identity in enumerate(selected):
        success_count = 0
        for img in identity.enrollment:
            obj = feature_cache.get(img)
            if obj["feature"] is None:
                enrollment_failed += 1
                continue
            template_vectors.append(obj["feature"])
            owners.append(idx)
            success_count += 1
        if success_count == 0:
            raise RuntimeError(f"第 {idx+1} 个注册身份所有模板均提取失败。无法按 {size} 人规模评价，请检查图片或检测策略。")
    matrix = np.stack(template_vectors).astype(np.float32, copy=False)
    owner_ids = np.asarray(owners, dtype=np.intp)
    known_results = []
    unknown_results = []
    match_times = []
    feature_times = []
    detect_times = []
    encode_times = []
    read_times = []
    for is_known, work in (
        (True, ((idx, path) for idx, ident in enumerate(selected) for path in ident.queries)),
        (False, ((None, path) for path in unknown_queries)),
    ):
        for expected, path in work:
            obj = feature_cache.get(path)
            if obj["feature"] is None:
                result = {"valid": False, "score": None, "correct_rank1": False,
                          "matched_correct": False, "match_ms": None}
            else:
                start = time.perf_counter()
                best_idx, best_score, _ = max_template_scores(matrix, owner_ids, size, obj["feature"])
                match_ms = (time.perf_counter() - start) * 1000
                match_times.append(match_ms)
                feature_times.append(obj["extract_wall_ms"])
                detect_times.append(obj["detect_ms"])
                encode_times.append(obj["encode_ms"])
                read_times.append(obj["read_ms"])
                result = {"valid": True, "score": best_score,
                          "correct_rank1": bool(is_known and best_idx == expected),
                          "matched_correct": bool(is_known and best_idx == expected),
                          "match_ms": match_ms}
            (known_results if is_known else unknown_results).append(result)
    return {
        "gallery_people": size, "gallery_templates_success": len(owners),
        "gallery_templates_failed": enrollment_failed,
        "known_queries": known_results, "unknown_queries": unknown_results,
        "timing": {
            "detection": latency_summary(detect_times),
            "recognition": latency_summary(encode_times),
            "feature_total": latency_summary(feature_times),
            "image_decode": latency_summary(read_times),
            "vector_match_only": latency_summary(match_times),
            "estimated_query_model_plus_match": latency_summary([
                e + m for e, m in zip(feature_times, match_times)
            ]),
        },
    }


def summarize_gallery(raw: dict, threshold: float) -> dict:
    if not -1.0 <= threshold <= 1.0:
        raise ValueError("无效身份匹配阈值")
    known = raw["known_queries"]
    unknown = raw["unknown_queries"]
    if not known or not unknown:
        raise ValueError("已知和未知查询必须均非空")
    known_valid = sum(q["valid"] for q in known)
    unknown_valid = sum(q["valid"] for q in unknown)
    correct_rank1 = sum(q["correct_rank1"] for q in known)
    known_correct_accept = sum(q["valid"] and q["correct_rank1"] and q["score"] >= threshold for q in known)
    known_wrong_accept = sum(q["valid"] and not q["correct_rank1"] and q["score"] >= threshold for q in known)
    known_reject = len(known) - known_correct_accept - known_wrong_accept
    unknown_false_accept = sum(q["valid"] and q["score"] >= threshold for q in unknown)
    # Detection/embedding failure produces rejection in the simulated open-set flow.
    return {
        "gallery_people": raw["gallery_people"],
        "gallery_templates_success": raw["gallery_templates_success"],
        "gallery_templates_failed": raw["gallery_templates_failed"],
        "threshold": float(threshold),
        "known_query_count": len(known), "unknown_query_count": len(unknown),
        "known_feature_success": known_valid, "unknown_feature_success": unknown_valid,
        "known_coverage": known_valid / len(known), "unknown_coverage": unknown_valid / len(unknown),
        "closed_set_rank1_end_to_end": correct_rank1 / len(known),
        "closed_set_rank1_conditional": (correct_rank1 / known_valid) if known_valid else None,
        "known_correct_accept": known_correct_accept,
        "known_wrong_accept": known_wrong_accept,
        "known_reject": known_reject,
        "dir_at_threshold": known_correct_accept / len(known),
        "known_wrong_id_rate": known_wrong_accept / len(known),
        "known_rejection_rate": known_reject / len(known),
        "unknown_false_accept": unknown_false_accept,
        "unknown_correct_reject": len(unknown) - unknown_false_accept,
        "fpir": unknown_false_accept / len(unknown),
        "unknown_correct_reject_rate": 1 - unknown_false_accept / len(unknown),
        "fpir_conditional_on_extraction": (unknown_false_accept / unknown_valid) if unknown_valid else None,
        "timing_ms": raw["timing"],
    }


def report_for_stage(protocol: Protocol, stage: str, results: Mapping[str, dict], *,
                     model_id: str, face_selection: str, target_fpir: float) -> dict:
    return {
        "protocol": PROTOCOL_VERSION, "stage": stage, "dataset": "LFW-funneled custom 1:N identity-disjoint split (NOT official LFW benchmark)",
        "seed": protocol.seed, "protocol_fingerprint": protocol.protocol_fingerprint,
        "inventory_fingerprint": protocol.inventory_fingerprint,
        "image_person_count": protocol.all_identity_count,
        "eligible_known_identities": protocol.eligible_known_count,
        "model_id": model_id, "face_selection": face_selection,
        "enroll_per_identity": protocol.enroll_per_identity,
        "queries_per_identity_max": protocol.queries_per_identity,
        "unknown_query_count_per_gallery": protocol.unknown_queries,
        "target_validation_fpir": target_fpir,
        "gallery_results": dict(results),
        "interpretation": (
            "Experiment-specific 1:N matching of pre-trained embeddings. "
            "lfw-largest chooses a face in centered LFW crops; it is not the strict live Web rule. "
            "Detection failures count as known rejection or unknown rejection. "
            "FPIR calibration constrains observed validation error only, not deployed security. "
            "No anti-spoofing or subject-consent verification is implemented."
        ),
    }
