"""Insert a read-only evaluation dashboard into the existing app's evaluation tab.

The installer never modifies biometric files, models, the database, or reports.
It creates a one-time backup of app.py and refuses unknown app layouts.
"""
from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

TAB_ANCHOR = 'with eval_tab:\n'
MARKER = 'render_experiment_dashboard(Path(__file__).resolve().parent / "reports")'
INSERT = (
    TAB_ANCHOR
    + '    from faceauth.dashboard import render_experiment_dashboard\n'
    f'    {MARKER}\n'
    '    st.divider()\n'
)


def install(project: Path) -> str:
    project = Path(project).resolve()
    app = project / 'app.py'
    if not app.is_file() or app.is_symlink():
        raise RuntimeError('Missing or unsafe app.py path')
    text = app.read_text(encoding='utf-8')
    if MARKER in text:
        return 'ALREADY_INSTALLED'
    if text.count(TAB_ANCHOR) != 1:
        raise RuntimeError('Unrecognized app.py; use standalone experiment_dashboard.py')
    if 'from pathlib import Path' not in text or 'import streamlit as st' not in text:
        raise RuntimeError('Required app imports not found; refusing to modify')
    changed = text.replace(TAB_ANCHOR, INSERT, 1)
    compile(changed, str(app), 'exec')
    backup = project / 'app.py.before_v16_dashboard.bak'
    if backup.exists() and backup.is_symlink():
        raise RuntimeError('Unsafe backup file is a symlink')
    if not backup.exists():
        shutil.copy2(app, backup)
    fd, temp_name = tempfile.mkstemp(prefix='faceauth_v16_', suffix='.py', dir=str(project))
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='') as output:
            output.write(changed)
        os.replace(temp_name, app)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
    # Prevent the preserved source backup being committed to a public repository.
    ignore = project / '.gitignore'
    if ignore.is_file() and not ignore.is_symlink():
        existing = ignore.read_text(encoding='utf-8')
        if 'app.py.before_v16_dashboard.bak' not in existing.splitlines():
            with ignore.open('a', encoding='utf-8') as handle:
                handle.write('\n# FaceAuth local integration backup\napp.py.before_v16_dashboard.bak\n')
    return 'INSTALLED'


def main() -> None:
    parser = argparse.ArgumentParser(description='Install v1.6 read-only dashboard in existing FaceAuth UI')
    parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(install(args.project_root))
    print('Original app backup (on first install): app.py.before_v16_dashboard.bak')
    print('Restart: streamlit run app.py --server.fileWatcherType none')
    print('Standalone: streamlit run experiment_dashboard.py --server.fileWatcherType none')


if __name__ == '__main__':
    main()
