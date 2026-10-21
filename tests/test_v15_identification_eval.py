"""v1.5 reproducible 1:N gallery, open-set metrics and leakage defenses."""
import json
from pathlib import Path

import numpy as np
import pytest

from faceauth.ident_eval import (
    FeatureCache, PROTOCOL_VERSION, build_protocol, calibrate_unknown_threshold,
    max_template_scores, scan_lfw_images, score_one_gallery, summarize_gallery,
    report_for_stage,
)
from scripts.eval_identification import validate_fixed_file, write_csv


def sample_inventory(n_multi=250, n_single=800):
    return {
        **{f"person_{i:04d}": tuple(f"person_{i:04d}/person_{i:04d}_{j:04d}.jpg" for j in range(1, 6))
           for i in range(n_multi)},
        **{f"single_{i:04d}": (f"single_{i:04d}/single_{i:04d}_0001.jpg",)
           for i in range(n_single)}
    }


def test_protocol_identity_disjoint_val_test_and_unknown_and_repeatable():
    inventory = sample_inventory()
    p = build_protocol(inventory, seed=42, gallery_sizes=(10, 50, 100), unknown_queries=200)
    p2 = build_protocol(inventory, seed=42, gallery_sizes=(10, 50, 100), unknown_queries=200)
    assert p == p2
    assert len(p.known_val) == len(p.known_test) == 100
    known_val = {x.identity for x in p.known_val}
    known_test = {x.identity for x in p.known_test}
    uv = {x.split('/')[0] for x in p.unknown_val}
    ut = {x.split('/')[0] for x in p.unknown_test}
    assert all(len(s) for s in (known_val, known_test, uv, ut))
    assert len(uv) == len(ut) == 200
    assert not (known_val & known_test or known_val & uv or known_val & ut
                or known_test & uv or known_test & ut or uv & ut)
    for group in (p.known_val, p.known_test):
        for sample in group:
            assert len(sample.enrollment) == 2
            assert len(sample.queries) == 2
            assert set(sample.enrollment).isdisjoint(sample.queries)
    p_alt_seed = build_protocol(inventory, seed=43, unknown_queries=200)
    assert p.protocol_fingerprint != p_alt_seed.protocol_fingerprint


def test_insufficient_known_or_unknown_safely_refused():
    with pytest.raises(ValueError, match="至少 200"):
        build_protocol(sample_inventory(n_multi=199), unknown_queries=20)
    with pytest.raises(ValueError, match="未知身份"):
        build_protocol(sample_inventory(n_multi=201, n_single=3), unknown_queries=20)
    with pytest.raises(ValueError, match="递增"):
        build_protocol(sample_inventory(), gallery_sizes=(50, 10))


def test_scan_lfw_folder_no_symlinks(tmp_path):
    folder = tmp_path / 'abc'
    folder.mkdir()
    (folder / 'abc_0001.jpg').write_bytes(b'a')
    (folder / 'other.jpg').write_bytes(b'b')
    assert scan_lfw_images(tmp_path) == {'abc': ('abc/abc_0001.jpg',)}


def test_max_template_scores_person_max_correct():
    matrix = np.asarray([[1, 0, 0], [.8, .6, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    ids = np.asarray([0, 0, 1, 2])
    idx, score, per_id = max_template_scores(matrix, ids, 3,
                                              np.asarray([.6, .8, 0], dtype=np.float32))
    assert idx == 0
    assert score == pytest.approx(.96)
    assert per_id[1] == pytest.approx(.8)
    # Person 0's second template wins even though first template is weaker.


def test_most_similar_person_not_first_template():
    matrix = np.asarray([[.8, .6], [0, 1], [1, 0]], dtype=np.float32)
    idx, score, _ = max_template_scores(matrix, np.array([0, 0, 1]), 2,
                                         np.array([0, 1], np.float32))
    assert idx == 0  # person's second template is an exact match
    assert score == pytest.approx(1)


def test_validation_threshold_bounds_fpir_and_equal_score_tie():
    scores = [.95, .8, .8, .7, .5, .1, 0, -.1, -.2, -.3]
    cal = calibrate_unknown_threshold(scores, target_fp_rate=.1)
    assert cal['allowed_false_accepts'] == 1
    assert sum(s >= cal['threshold'] for s in scores) <= 1
    assert cal['observed_val_fpir'] <= .1
    with pytest.raises(ValueError, match='至少两个'):
        calibrate_unknown_threshold([], .01)
    with pytest.raises(ValueError, match='target-fpir'):
        calibrate_unknown_threshold([.2, .3], 1.0)


def _raw_scores():
    def q(valid, score=None, correct=False):
        return {'valid': valid, 'score': score, 'correct_rank1': correct,
                'matched_correct': correct, 'match_ms': .1 if valid else None}
    return {
        'gallery_people': 3, 'gallery_templates_success': 6, 'gallery_templates_failed': 0,
        'known_queries': [q(True,.9,True), q(True,.8,False), q(True,.1,True), q(False)],
        'unknown_queries': [q(True,.7), q(True,.2), q(False)],
        'timing': {'vector_match_only': {'mean_ms': .1, 'p95_ms': .12},
                   'estimated_query_model_plus_match': {'mean_ms': 4.1, 'p95_ms': 4.2}},
    }


def test_open_set_metrics_count_detection_failures_and_wrong_person():
    result = summarize_gallery(_raw_scores(), threshold=.5)
    assert result['known_query_count'] == 4
    assert result['known_coverage'] == .75
    assert result['closed_set_rank1_end_to_end'] == .5
    assert result['known_correct_accept'] == 1
    assert result['known_wrong_accept'] == 1
    assert result['known_reject'] == 2
    assert result['dir_at_threshold'] == .25
    assert result['known_wrong_id_rate'] == .25
    assert result['fpir'] == pytest.approx(1/3)
    assert result['unknown_coverage'] == pytest.approx(2/3)
    assert result['fpir_conditional_on_extraction'] == .5
    assert result['unknown_correct_reject'] == 2


class FakeFeatureCache:
    def __init__(self, mapping):
        self.mapping = mapping
    def get(self, path):
        if path in self.mapping:
            return self.mapping[path]
        return {'feature': None}


def _valid(feature):
    return {'feature': np.asarray(feature, dtype=np.float32), 'detect_ms':1,
            'encode_ms':2, 'extract_wall_ms':4, 'read_ms':1}


def test_real_gallery_matching_and_repeated_template_max():
    from faceauth.ident_eval import IdentitySample
    known = (
        IdentitySample('A', ('A1','A2'), ('AQ',)),
        IdentitySample('B', ('B1','B2'), ('BQ',)),
    )
    cache = FakeFeatureCache({
        'A1': _valid([1,0,0]), 'A2': _valid([.8,.6,0]), 'AQ': _valid([1,0,0]),
        'B1': _valid([0,1,0]), 'B2': _valid([0,.8,.6]), 'BQ': _valid([0,1,0]),
        'U': _valid([0,0,1]),
    })
    raw = score_one_gallery(known, ('U',), 2, cache)
    summary = summarize_gallery(raw, threshold=.5)
    assert summary['gallery_people'] == 2
    assert summary['known_query_count'] == 2
    assert summary['known_correct_accept'] == 2
    assert summary['closed_set_rank1_end_to_end'] == 1
    assert summary['unknown_false_accept'] == 1  # B2 has score .6


def test_missing_all_templates_for_identity_is_error():
    from faceauth.ident_eval import IdentitySample
    known = (IdentitySample('A', ('missing',), ('A1',)),)
    with pytest.raises(RuntimeError, match='所有模板均提取失败'):
        score_one_gallery(known, ('U',), 1, FakeFeatureCache({}))


def test_threshold_file_compatibility_and_csv(tmp_path):
    p = build_protocol(sample_inventory(), seed=17, unknown_queries=200)
    detector = {'yunet_score': .8, 'min_face_size': 55}
    cal = {'protocol': PROTOCOL_VERSION, 'stage': 'val',
           'protocol_fingerprint': p.protocol_fingerprint, 'model_id': 'mock',
           'face_selection': 'lfw-largest', 'detector': detector,
           'gallery_sizes': [10,50,100], 'target_fpir': .01,
           'per_gallery': {'10': {'threshold': .2}, '50': {'threshold': .3}, '100': {'threshold': .4}}}
    validate_fixed_file(cal, p, model_id='mock', policy='lfw-largest', detector=detector)
    bad = {**cal, 'protocol_fingerprint': 'broken'}
    with pytest.raises(ValueError, match='protocol_fingerprint'):
        validate_fixed_file(bad, p, model_id='mock', policy='lfw-largest', detector=detector)
    bad = {**cal, 'face_selection': 'strict'}
    with pytest.raises(ValueError, match='face_selection'):
        validate_fixed_file(bad, p, model_id='mock', policy='lfw-largest', detector=detector)
    report = report_for_stage(p, 'test', {'10': summarize_gallery(_raw_scores(), .5)},
                              model_id='mock', face_selection='lfw-largest', target_fpir=.01)
    out = tmp_path / 'result.csv'
    write_csv(out, report)
    assert out.is_file()
    assert 'gallery_people' in out.read_text(encoding='utf-8-sig')
    assert '3' in out.read_text(encoding='utf-8-sig')
    assert json.loads(json.dumps(report))['stage'] == 'test'
