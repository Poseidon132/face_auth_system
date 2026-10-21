"""Tests for read-only aggregate report dashboard, without real face data."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from faceauth.dashboard_data import (
    ReportDataError, load_ident_test, load_pair_test, locate_reports,
)
from scripts.install_v16_dashboard import install


def _write(tmp_path: Path, name: str, data: dict) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(data), encoding='utf-8')
    return path


def _pair_report(skipped=0):
    count = 1000
    tp, tn, fp, fn = 491, 495, 5, 9
    return {
        'split': 'test', 'threshold': .3028, 'pairs_requested': count,
        'pairs_skipped': skipped, 'accuracy': .986, 'roc_auc': .9937,
        'evaluation_detector': {'yunet_score_threshold': .8,
                                'min_face_size_px': 55,
                                'face_selection_policy': 'lfw-largest'},
        'end_to_end_reject_on_failure': {
            'total_pairs': count, 'accuracy': .986, 'tar': .982,
            'far': .01, 'frr': .018, 'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        },
    }


def _ident_row(size, known_count, accepted_count, unknown_count=300, false_accept=1):
    return {
        'gallery_people': size, 'known_query_count': known_count,
        'unknown_query_count': unknown_count, 'known_correct_accept': accepted_count,
        'unknown_false_accept': false_accept,
        'gallery_templates_success': 2 * size, 'known_coverage': 1.0,
        'unknown_coverage': 1.0,
        'closed_set_rank1_end_to_end': 1.,
        'dir_at_threshold': accepted_count / known_count,
        'fpir': false_accept / unknown_count, 'threshold': .42,
        'timing_ms': {
            'vector_match_only': {'mean_ms': .018, 'p95_ms': .03},
            'estimated_query_model_plus_match': {'mean_ms': 11.8, 'p95_ms': 13.0},
        },
    }


def _ident_report():
    return {
        'protocol': 'lfw-identity-disjoint-1n-v1', 'stage': 'test',
        'face_selection': 'lfw-largest', 'protocol_fingerprint': 'abc123',
        'seed': 20261021, 'enroll_per_identity': 2,
        'target_validation_fpir': .01,
        'gallery_results': {
            '100': _ident_row(100, 171, 170),
            '10': _ident_row(10, 16, 16, false_accept=2),
            '50': _ident_row(50, 84, 84, false_accept=0),
        },
    }


def test_lfw_pairs_100_percent_coverage(tmp_path):
    value = load_pair_test(_write(tmp_path, 'pair.json', _pair_report()))
    assert value['processed'] == 1000
    assert value['coverage'] == 1.0
    assert value['accuracy_e2e'] == .986
    assert value['policy'] == 'lfw-largest'
    assert (value['tp'], value['fp'], value['fn'], value['tn']) == (491, 5, 9, 495)


def test_reject_inconsistent_confusion_matrix(tmp_path):
    bad = _pair_report()
    bad['end_to_end_reject_on_failure']['tp'] += 2
    with pytest.raises(ReportDataError, match='confusion'):
        load_pair_test(_write(tmp_path, 'bad.json', bad))


def test_pairs_wrong_split(tmp_path):
    bad = _pair_report()
    bad['split'] = 'val'
    with pytest.raises(ReportDataError, match='test'):
        load_pair_test(_write(tmp_path, 'val.json', bad))


def test_ident_report_rows_sorted_and_rates_from_report(tmp_path):
    output = load_ident_test(_write(tmp_path, 'ident.json', _ident_report()))
    assert [r['gallery_people'] for r in output['rows']] == [10, 50, 100]
    assert output['rows'][-1]['dir'] == pytest.approx(170 / 171)
    assert output['rows'][-1]['fpir'] == pytest.approx(1 / 300)
    assert output['rows'][0]['unknown_false_accept'] == 2
    assert output['rows'][0]['match_mean_ms'] == .018
    assert output['rows'][0]['model_search_est_mean_ms'] == 11.8


def test_ident_rejects_inconsistent_fpir(tmp_path):
    bad = _ident_report()
    bad['gallery_results']['10']['fpir'] = 1.
    with pytest.raises(ReportDataError, match='FPIR'):
        load_ident_test(_write(tmp_path, 'bad.json', bad))


def test_ident_rejects_inconsistent_dir(tmp_path):
    bad = _ident_report()
    bad['gallery_results']['100']['dir_at_threshold'] = 1.
    with pytest.raises(ReportDataError, match='DIR'):
        load_ident_test(_write(tmp_path, 'bad.json', bad))


def test_ident_rejects_validation_file(tmp_path):
    bad = _ident_report()
    bad['stage'] = 'val'
    with pytest.raises(ReportDataError, match='test'):
        load_ident_test(_write(tmp_path, 'val.json', bad))


def test_no_reports_no_fabricated_results(tmp_path):
    assert locate_reports(tmp_path) == (None, None)


def test_prefer_lfw_primary_report(tmp_path):
    _write(tmp_path, 'lfw_metrics_test.json', _pair_report())
    _write(tmp_path, 'lfw_metrics_test_lfwprimary.json', _pair_report())
    _write(tmp_path, 'lfw_1n_test.json', _ident_report())
    a,b=locate_reports(tmp_path)
    assert a.name == 'lfw_metrics_test_lfwprimary.json'
    assert b.name == 'lfw_1n_test.json'


def test_reject_json_arrays(tmp_path):
    (tmp_path/'array.json').write_text('[]',encoding='utf-8')
    with pytest.raises(ReportDataError, match='object'):
        load_ident_test(tmp_path/'array.json')


def test_reject_nonfinite_timing(tmp_path):
    bad = _ident_report()
    bad['gallery_results']['100']['timing_ms']['vector_match_only']['mean_ms'] = -1
    with pytest.raises(ReportDataError, match='timing'):
        load_ident_test(_write(tmp_path, 'bad.json', bad))


def test_installer_preserves_old_app_and_is_idempotent(tmp_path):
    original = (
        'from pathlib import Path\n'
        'import streamlit as st\n'
        'with eval_tab:\n'
        '    st.subheader("Placeholder")\n'
    )
    (tmp_path/'app.py').write_text(original,encoding='utf-8')
    assert install(tmp_path) == 'INSTALLED'
    new=(tmp_path/'app.py').read_text(encoding='utf-8')
    assert 'render_experiment_dashboard' in new
    assert 'st.subheader("Placeholder")' in new
    assert (tmp_path/'app.py.before_v16_dashboard.bak').read_text(encoding='utf-8') == original
    assert install(tmp_path) == 'ALREADY_INSTALLED'
    assert (tmp_path/'app.py').read_text(encoding='utf-8') == new


def test_installer_refuses_unknown_layout_without_changing_it(tmp_path):
    original='import streamlit as st\nprint("another application")\n'
    (tmp_path/'app.py').write_text(original,encoding='utf-8')
    with pytest.raises(RuntimeError, match='Unrecognized'):
        install(tmp_path)
    assert (tmp_path/'app.py').read_text(encoding='utf-8') == original
    assert not (tmp_path/'app.py.before_v16_dashboard.bak').exists()


def test_render_standalone_with_mock_streamlit_reports(tmp_path, monkeypatch):
    import sys
    from faceauth.dashboard import render_experiment_dashboard
    _write(tmp_path, 'lfw_metrics_test_lfwprimary.json', _pair_report())
    _write(tmp_path, 'lfw_1n_test.json', _ident_report())

    class FakeStreamlit:
        def __init__(self):
            self.calls = []

        def __getattr__(self, method):
            def fake(*args, **kwargs):
                self.calls.append((method, args, kwargs))
                return None
            return fake

        def columns(self, count, **kwargs):
            return [self] * count

        def expander(self, *args, **kwargs):
            return self

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    fake = FakeStreamlit()
    monkeypatch.setitem(sys.modules, 'streamlit', fake)
    render_experiment_dashboard(tmp_path)
    called = [method for method, _, _ in fake.calls]
    assert called.count('metric') == 9
    assert called.count('dataframe') == 3
    assert called.count('bar_chart') == 2
    assert called.count('download_button') == 1
    assert called.count('warning') == 0


def test_render_handles_missing_reports_without_fake_metrics(tmp_path, monkeypatch):
    import sys
    from faceauth.dashboard import render_experiment_dashboard

    class Fake:
        def __init__(self):
            self.metrics = []
            self.infos = []
        def metric(self, *a, **kw):
            self.metrics.append((a, kw))
        def info(self, *a, **kw):
            self.infos.append(a)
        def __getattr__(self, name):
            return lambda *a, **kw: None

    st = Fake()
    monkeypatch.setitem(sys.modules, 'streamlit', st)
    render_experiment_dashboard(tmp_path)
    assert len(st.metrics) == 0
    assert len(st.infos) >= 2
