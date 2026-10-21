import pytest
from scripts.preflight_github import validate_path

@pytest.mark.parametrize('filename', [
    'models/face_recognition_sface.onnx', 'data/faceauth.sqlite3',
    'reports/lfw_1n_test.json', 'docs/screenshots/private_person.png',
    'app.py.before_github_release.bak', 'scripts/eval_pairs_v11_backup.py', '.streamlit/secrets.toml',
])
def test_publication_rejects_sensitive_files(filename):
    assert validate_path(filename) is not None

@pytest.mark.parametrize('filename', [
    'docs/screenshots/dashboard.png', 'docs/benchmarks/lfw_1ton_test.csv',
    'data/.gitkeep', 'models/.gitkeep', 'docs/PAIR_CSV_EXAMPLE.csv', 'app.py', 'README.md',
])
def test_publication_allows_safe_files(filename):
    assert validate_path(filename) is None
