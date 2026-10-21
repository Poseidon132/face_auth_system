"""Verification metrics and validation-only threshold calibration."""
from __future__ import annotations

import numpy as np


def _arrays(scores, same) -> tuple[np.ndarray, np.ndarray]:
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    y = np.asarray(same).reshape(-1)
    if len(s) != len(y) or not len(s):
        raise ValueError("分数和标签数量应相等且非空")
    if not np.isfinite(s).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("分数需为有限值，标签必须是 0 或 1")
    return s, y.astype(np.bool_)


def auc_rank(scores, same) -> float | None:
    s, pos = _arrays(scores, same)
    np_, nn_ = int(pos.sum()), int((~pos).sum())
    if not np_ or not nn_:
        return None
    order = np.argsort(s, kind="stable")
    ordered = s[order]
    ranks = np.empty(len(s), dtype=np.float64)
    left = 0
    while left < len(s):
        right = left + 1
        while right < len(s) and ordered[right] == ordered[left]:
            right += 1
        avg_rank = ((left + 1) + right) / 2.0
        ranks[order[left:right]] = avg_rank
        left = right
    return float((ranks[pos].sum() - np_ * (np_ + 1) / 2.0) / (np_ * nn_))


def rates_at(scores, same, threshold: float) -> dict[str, float | int | None]:
    s, y = _arrays(scores, same)
    pred = s >= threshold
    tp = int(np.count_nonzero(pred & y))
    tn = int(np.count_nonzero(~pred & ~y))
    fp = int(np.count_nonzero(pred & ~y))
    fn = int(np.count_nonzero(~pred & y))
    n_pos, n_neg = tp + fn, tn + fp
    return {
        "threshold": float(threshold), "pairs": int(len(s)),
        "positive_pairs": n_pos, "negative_pairs": n_neg,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": (tp + tn) / len(s),
        "tar": tp / n_pos if n_pos else None,
        "far": fp / n_neg if n_neg else None,
        "frr": fn / n_pos if n_pos else None,
    }


def approximate_eer(scores, same) -> dict[str, float] | None:
    s, y = _arrays(scores, same)
    if not y.any() or y.all():
        return None
    candidates = np.r_[np.nextafter(s.max(), np.inf), np.unique(s)[::-1], np.nextafter(s.min(), -np.inf)]
    best = None
    for t in candidates:
        r = rates_at(s, y, float(t))
        gap = abs(r["far"] - r["frr"])
        if best is None or gap < best[0]:
            best = (gap, float(t), float((r["far"] + r["frr"]) / 2.0))
    return {"threshold": best[1], "eer_approx": best[2]}


def choose_threshold_on_validation(scores, same, *, target_far: float = 0.01) -> dict:
    s, y = _arrays(scores, same)
    if not y.any() or y.all():
        raise ValueError("阈值校准必须同时具有正例和负例")
    if not 0 <= target_far < 1:
        raise ValueError("target_far 应属于 [0,1)")
    # Maximize TAR subject to FAR <= target, tie-break by a stricter threshold.
    candidates = np.r_[1.0, np.unique(s)[::-1]]
    candidates = candidates[(candidates > 0.0) & (candidates <= 1.0)]
    eligible = []
    for t in candidates:
        r = rates_at(s, y, float(t))
        if r["far"] <= target_far + 1e-12:
            eligible.append((float(r["tar"]), float(t)))
    if not eligible:
        raise RuntimeError("无法选出满足 FAR 约束的阈值")
    # Higher TAR then higher threshold for tie.
    tar, threshold = max(eligible, key=lambda item: (item[0], item[1]))
    result = rates_at(s, y, threshold)
    result.update({"target_far": target_far, "calibration_split": "val", "notes": "Empirical validation FAR is not a production security guarantee."})
    return result
