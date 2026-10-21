"""Read aggregate LFW benchmark reports for the optional Streamlit dashboard.

This module reads only fixed aggregate JSON reports in reports/ and never opens
images, SQLite person databases, embeddings, or audit logs.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

MAX_REPORT_BYTES = 2_000_000


class ReportDataError(ValueError):
    """Report is invalid or not compatible with the expected evaluation schema."""


def _read_json(path: Path) -> dict[str, Any]:
    path = Path(path)
    if path.is_symlink():
        raise ReportDataError('Report may not be a symbolic link')
    if not path.is_file():
        raise FileNotFoundError(str(path))
    if path.stat().st_size > MAX_REPORT_BYTES:
        raise ReportDataError('Aggregate report is unexpectedly large')
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReportDataError('Could not read the aggregate JSON report') from exc
    if not isinstance(value, dict):
        raise ReportDataError('Expected a JSON object')
    return value


def _fraction(obj: dict, name: str) -> float:
    value = _finite(obj, name)
    if not 0.0 <= value <= 1.0:
        raise ReportDataError(f'{name}: expected fraction in [0,1]')
    return value


def _finite(obj: dict, name: str) -> float:
    value = obj.get(name)
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ReportDataError(f'{name}: required numeric field missing')
    result = float(value)
    if not math.isfinite(result):
        raise ReportDataError(f'{name}: expected finite number')
    return result


def _count(obj: dict, name: str) -> int:
    value = obj.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ReportDataError(f'{name}: expected non-negative integer')
    return value


def _timing(summary: dict, category: str, stat: str) -> float | None:
    data = summary.get('timing_ms')
    if not isinstance(data, dict):
        return None
    details = data.get(category)
    if not isinstance(details, dict):
        return None
    value = details.get(stat)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReportDataError(f'{category}.{stat}: expected numeric timing')
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ReportDataError(f'{category}.{stat}: invalid timing')
    return result


def load_pair_test(path: Path) -> dict:
    """Normalize 1:1 pair report; includes all failed detections in coverage."""
    obj = _read_json(path)
    if obj.get('split') != 'test':
        raise ReportDataError('Expected pair test report, not validation')
    total = _count(obj, 'pairs_requested')
    skipped = _count(obj, 'pairs_skipped')
    if total < 1 or skipped > total:
        raise ReportDataError('Invalid pair totals')
    audit = obj.get('evaluation_detector')
    if not isinstance(audit, dict):
        audit = {}
    # v1.3 reports set face_selection_policy only for non-strict evaluations.
    policy = audit.get('face_selection_policy', 'strict')
    if policy not in ('strict', 'lfw-largest'):
        raise ReportDataError('Unrecognized 1:1 face-selection policy')
    e2e = obj.get('end_to_end_reject_on_failure')
    if not isinstance(e2e, dict):
        raise ReportDataError('Missing pair-level end-to-end metrics')
    if _count(e2e, 'total_pairs') != total:
        raise ReportDataError('Inconsistent end-to-end pair count')
    result = {
        'source': str(path), 'total': total, 'processed': total - skipped,
        'skipped': skipped, 'coverage': (total - skipped) / total,
        'policy': policy, 'threshold': _finite(obj, 'threshold') if 'threshold' in obj else None,
        'accuracy_valid': _fraction(obj, 'accuracy'),
        'accuracy_e2e': _fraction(e2e, 'accuracy'),
        'tar_e2e': _fraction(e2e, 'tar'),
        'far_e2e': _fraction(e2e, 'far'),
        'frr_e2e': _fraction(e2e, 'frr'),
        'roc_auc_valid': _fraction(obj, 'roc_auc'),
        'tp': _count(e2e, 'tp'), 'tn': _count(e2e, 'tn'),
        'fp': _count(e2e, 'fp'), 'fn': _count(e2e, 'fn'),
    }
    if result['tp'] + result['tn'] + result['fp'] + result['fn'] != total:
        raise ReportDataError('Inconsistent confusion-matrix totals')
    return result


def load_ident_test(path: Path) -> dict:
    """Normalize v1.5 identity-disjoint test, without inferring missing values."""
    obj = _read_json(path)
    if obj.get('protocol') != 'lfw-identity-disjoint-1n-v1' or obj.get('stage') != 'test':
        raise ReportDataError('Expected v1.5 identity-disjoint test report')
    galleries = obj.get('gallery_results')
    if not isinstance(galleries, dict) or not galleries:
        raise ReportDataError('Missing gallery results')
    rows = []
    for key, entry in galleries.items():
        if not isinstance(entry, dict):
            raise ReportDataError('Invalid gallery metrics')
        size = _count(entry, 'gallery_people')
        if key != str(size) or size < 2:
            raise ReportDataError('Mismatched gallery size keys')
        known = _count(entry, 'known_query_count')
        unknown = _count(entry, 'unknown_query_count')
        if known == 0 or unknown == 0:
            raise ReportDataError('Empty known or unknown query set')
        rank1 = _fraction(entry, 'closed_set_rank1_end_to_end')
        dir_rate = _fraction(entry, 'dir_at_threshold')
        fpir = _fraction(entry, 'fpir')
        if dir_rate > rank1 + 1e-9:
            raise ReportDataError('DIR may not exceed Rank-1')
        correct = _count(entry, 'known_correct_accept')
        unknown_false = _count(entry, 'unknown_false_accept')
        if correct > known or unknown_false > unknown:
            raise ReportDataError('Invalid accepted query counts')
        if not math.isclose(correct / known, dir_rate, abs_tol=1e-8):
            raise ReportDataError('DIR and accepted query count mismatch')
        if not math.isclose(unknown_false / unknown, fpir, abs_tol=1e-8):
            raise ReportDataError('FPIR and false accept count mismatch')
        rows.append({
            'gallery_people': size, 'known_queries': known,
            'unknown_queries': unknown, 'known_correct_accept': correct,
            'unknown_false_accept': unknown_false, 'rank1': rank1,
            'dir': dir_rate, 'fpir': fpir,
            'known_coverage': _fraction(entry, 'known_coverage'),
            'unknown_coverage': _fraction(entry, 'unknown_coverage'),
            'threshold': _finite(entry, 'threshold'),
            'template_count': _count(entry, 'gallery_templates_success'),
            'match_mean_ms': _timing(entry, 'vector_match_only', 'mean_ms'),
            'match_p95_ms': _timing(entry, 'vector_match_only', 'p95_ms'),
            'model_search_est_mean_ms': _timing(entry, 'estimated_query_model_plus_match', 'mean_ms'),
            'model_search_est_p95_ms': _timing(entry, 'estimated_query_model_plus_match', 'p95_ms'),
            'detect_mean_ms': _timing(entry, 'detection', 'mean_ms'),
            'recognition_mean_ms': _timing(entry, 'recognition', 'mean_ms'),
        })
    rows.sort(key=lambda x: x['gallery_people'])
    return {
        'source': str(path), 'rows': rows,
        'face_selection': str(obj.get('face_selection', 'unknown')),
        'protocol_fingerprint': str(obj.get('protocol_fingerprint', '')),
        'seed': obj.get('seed'),
        'enroll_per_identity': obj.get('enroll_per_identity'),
        'target_validation_fpir': obj.get('target_validation_fpir'),
    }


def locate_reports(root: Path) -> tuple[Path | None, Path | None]:
    root = Path(root)
    pair_candidates = (
        root / 'lfw_metrics_test_lfwprimary.json',
        root / 'lfw_metrics_test.json',
    )
    pair = next((p for p in pair_candidates if p.is_file()), None)
    ident = root / 'lfw_1n_test.json'
    return pair, ident if ident.is_file() else None
