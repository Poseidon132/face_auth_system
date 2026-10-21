"""Offline integrated validation/test pipeline with synthetic photos/model.
No public dataset download and no real biometric data involved.
"""
import json
import hashlib

import numpy as np
import pytest

from faceauth.vision import FaceObservation
from scripts import eval_identification as script


def test_val_then_test_cli_reports_and_rejects_protocol_drift(tmp_path, monkeypatch):
    root = tmp_path / 'lfw_funneled'
    root.mkdir()
    # 12 eligible identities, 32 out-of-gallery identities.
    for prefix, n_people, n_photos in (('member', 12, 4), ('nonmember', 32, 1)):
        for i in range(n_people):
            pid = f'{prefix}_{i:03d}'
            folder = root / pid
            folder.mkdir()
            for j in range(1, n_photos + 1):
                (folder / f'{pid}_{j:04d}.jpg').write_bytes(b'placeholder-no-identities')
    all_names = sorted(p.name for p in root.iterdir())
    number = {name: idx for idx, name in enumerate(all_names)}

    def read_stub(path):
        idx = number[path.parent.name]
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        image[0, 0, 0] = idx
        return image

    class FakeEngine:
        def __init__(self, score_threshold=.8, min_face_size=55):
            self.model_id = 'synthetic-only-for-tests'
        def extract(self, image, *, annotate=False, selection_policy='strict'):
            idx = int(image[0,0,0])
            feature = np.zeros(128, dtype=np.float32)
            feature[idx] = 1.0
            return FaceObservation(feature, .95, (1,1,60,60), 60,60,100,1.1,2.2)

    from faceauth import vision
    monkeypatch.setattr(vision, 'read_photo_file', read_stub)
    monkeypatch.setattr(script, 'FaceEngine', FakeEngine)
    parser = script.make_parser()
    out_val = tmp_path / 'val.json'
    out_test = tmp_path / 'test.json'
    thresholds = tmp_path / 'thresholds.json'
    shared = ['--images', str(root), '--sizes', '2', '4',
              '--unknown-queries', '8', '--no-chart', '--seed', '12']
    val = parser.parse_args(['--stage', 'val', *shared, '--calibration-out', str(thresholds),
                             '--out', str(out_val)])
    result_val = script.run(val)
    assert result_val['stage'] == 'val'
    assert result_val['unique_images_failed'] == 0
    assert result_val['gallery_results']['4']['known_coverage'] == 1
    assert out_val.with_suffix('.csv').is_file()
    assert all(0 <= value['threshold'] <= 1 for value in json.loads(thresholds.read_text())['per_gallery'].values())

    test = parser.parse_args(['--stage', 'test', *shared, '--threshold-file', str(thresholds),
                              '--out', str(out_test)])
    result_test = script.run(test)
    assert result_test['stage'] == 'test'
    assert result_test['protocol_fingerprint'] == result_val['protocol_fingerprint']
    assert result_test['gallery_results']['4']['closed_set_rank1_end_to_end'] == 1.0
    assert result_test['gallery_results']['4']['fpir'] == 0
    assert out_test.with_suffix('.csv').is_file()

    broken = parser.parse_args(['--stage', 'test', *shared, '--seed', '99',
                                '--threshold-file', str(thresholds), '--out', str(tmp_path / 'broken.json')])
    with pytest.raises(ValueError, match='protocol_fingerprint'):
        script.run(broken)


def test_val_test_must_use_appropriate_calibration_flags():
    parser = script.make_parser()
    args = parser.parse_args(['--stage', 'test', '--out', 'foo.json'])
    with pytest.raises(SystemExit):
        script.run(args)
