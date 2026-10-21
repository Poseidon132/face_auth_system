"""Fail safe before publishing a FaceAuth repo from a local Git stage.

Checks paths, sizes and obvious secret filename patterns in *all tracked files*.
Does not certify that source text is free of every possible secret.
"""
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent.parent
ALLOW_PUBLIC_PNG={'docs/screenshots/dashboard.png'}
BAD_SUFFIXES={'.jpeg','.jpg','.webp','.bmp','.tif','.tiff','.onnx','.sqlite','.sqlite3','.db','.npz','.npy','.pt','.pth','.safetensors','.pkl','.pickle','.mov','.mp4','.avi','.tgz','.zip'}
BAD_DIRECTORIES=('data/','models/','reports/','private_faces/','private_pairs/','outputs/')
MAX_SIZE=10*1024*1024


def validate_path(path:str)->str|None:
    key=path.replace('\\','/').lstrip('./')
    lower=key.lower()
    if lower not in {'data/.gitkeep','models/.gitkeep'} and any(lower.startswith(prefix) for prefix in BAD_DIRECTORIES):
        return 'private or generated directory'
    if '/__pycache__/' in f'/{lower}' or '.pytest_cache' in lower:
        return 'generated Python cache'
    if 'secrets' in lower or lower.startswith('.env') or '/.env' in lower:
        return 'potential secret'
    if lower.endswith(('.bak','.backup','.log','.sqlite3-shm','.sqlite3-wal')) or 'backup' in lower:
        return 'backup or log'
    p=Path(lower)
    if p.suffix in BAD_SUFFIXES:
        return 'blocked extension'
    if p.suffix=='.png' and lower not in ALLOW_PUBLIC_PNG:
        return 'only reviewed dashboard screenshot may be public'
    if key.startswith('/') or '..' in p.parts:
        return 'unsafe path'
    return None


def main()->int:
    try:
        result=subprocess.run(['git','ls-files','-z'],cwd=ROOT,capture_output=True,check=True)
    except (OSError, subprocess.CalledProcessError):
        print('ERROR: run git init and git add before preflight.')
        return 2
    names=[s.decode('utf-8','replace') for s in result.stdout.split(b'\0') if s]
    if not names:
        print('ERROR: no tracked/staged files were found. Run git add -A first.')
        return 2
    findings=[]
    for name in names:
        issue=validate_path(name)
        path=ROOT/name
        if path.is_file() and path.stat().st_size>MAX_SIZE:
            issue='file larger than 10 MiB'
        if issue:
            findings.append((name,issue))
    if findings:
        print('PUBLICATION BLOCKED - unsafe tracked paths:')
        for name,reason in findings:
            print('  ',name,':',reason)
        print('Remove from Git tracking (git rm --cached ...) and inspect Git history before publishing.')
        return 1
    print(f'PASS: {len(names)} tracked paths pass the FaceAuth filename/size privacy guard.')
    print('Manual privacy and license review is still required.')
    return 0


if __name__=='__main__':
    sys.exit(main())
