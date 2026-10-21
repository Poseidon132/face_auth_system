from pathlib import Path

path=Path(__file__).resolve().parent.parent/'app.py'
text=path.read_text(encoding='utf-8')
dot=chr(183)
old=f'FACE AUTHENTICATION {dot} LOCAL DEMO V1.0'
new=f'FACE AUTHENTICATION {dot} LOCAL DEMO V1.6'
if old in text:
    backup=path.with_name('app.py.before_github_release.bak')
    if not backup.exists():
        backup.write_text(text,encoding='utf-8')
    path.write_text(text.replace(old,new,1),encoding='utf-8')
    print('Updated app banner to v1.6; created ignored .bak backup.')
elif new in text:
    print('App banner already v1.6.')
else:
    print('Could not locate expected banner. Please verify app.py manually.')
