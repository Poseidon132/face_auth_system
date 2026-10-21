# GitHub publication checklist (Windows PowerShell)

These commands apply to the **existing local v1.6 project**. Never replace your working project directory with an old clean archive. This patch changes the README, public documentation and app banner only; personal SQLite data stays local.

1. Back up your current working code or make a Git commit before applying the publication patch.
2. Confirm the new README and dashboard-only screenshot are present. Do not publish third-party portrait screenshots, real user images, feature embeddings, SQLite, full reports or personal names.
3. Run `python -m pytest -q` and `python scripts/smoke_test.py` in `(faceauth)`.
4. On GitHub create an **empty repository** named `face_auth_system` under the account that you own. Choose Public only if you intend to publish the code; do not add README, license or gitignore on the GitHub page because your local project already supplies those.
5. Run these commands from the existing terminal and verify the output before committing.

```powershell
cd E:\AIdemoace_auth_system
conda activate faceauth

git init
git branch -M main
# If origin exists, verify it instead of overwriting it.
git remote -v
# Only if origin does not exist:
git remote add origin https://github.com/Poseidon132/face_auth_system.git

git add -A
python scripts/preflight_github.py
git diff --cached --stat
git diff --cached --name-only

# Only after confirming the staged paths contain no private data:
git commit -m "Initial release: FaceAuth v1.6"
git push -u origin main
```

If the repo already has an origin, do not blindly run `git remote add origin` or force push. For a new project with local Git initialization, create a fresh empty GitHub repository or reconcile histories before pushing. `git push` requires your own authenticated GitHub credentials. The assistant cannot access files on your Windows disk or complete a Git push without a connected authorized integration.

**Note:** `.gitignore` stops most private files from being staged now, but does not remove files already tracked in older Git history. `scripts/preflight_github.py` checks tracked/staged file names; inspect any older commits before making the repository public.
