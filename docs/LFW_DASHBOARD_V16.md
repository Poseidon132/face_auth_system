# FaceAuth v1.6 - LFW evaluation dashboard

This optional patch adds a **read-only** dashboard to the existing Streamlit
"Experiments and notes" tab. It does not change face enrollment, verification,
identification, model weights, thresholds, the SQLite database, or LFW evaluation code.

## Windows paths for this workstation

- Project: `E:\AIdemo\face_auth_system`
- Edge downloads: `D:\Edge`
- Report data: `E:\AIdemo\face_auth_system\reports`

Save `face_auth_lfw_v1_6_dashboard_patch.zip` in `D:\Edge`, then run:

```powershell
cd E:\AIdemo\face_auth_system
conda activate faceauth
$zip = "D:\Edge\face_auth_lfw_v1_6_dashboard_patch.zip"
Test-Path -LiteralPath $zip
Expand-Archive -LiteralPath $zip -DestinationPath . -Force
python -m pytest -q
python scripts/install_v16_dashboard.py
streamlit run app.py --server.fileWatcherType none
```

If an older Streamlit server is open, stop it with Ctrl+C first. No additional
model or dataset download is needed. The installer backs up the original app to
`app.py.before_v16_dashboard.bak`. Running the installer twice does not duplicate
the dashboard.

You may also run a completely separate page without modifying app.py:

```powershell
streamlit run experiment_dashboard.py --server.fileWatcherType none
```

## Required aggregate data files

The dashboard **reads** these files if present:

```
reports/lfw_metrics_test_lfwprimary.json   # preferred, LFW 1:1 full coverage
reports/lfw_metrics_test.json              # fallback, strict-mode 1:1
reports/lfw_1n_test.json                   # identity-disjoint 1:N test
```

It will show a missing-report message rather than invent accuracy numbers.
The 1:1 screen displays **end-to-end reject-on-detection-failure accuracy**, plus
coverage, FAR, FRR and conditional AUC. The 1:N screen shows registered-gallery
sizes, known and unknown query counts, Rank-1, thresholded DIR, FPIR and
per-gallery calibrated thresholds. It plots Rank-1/DIR separately from FPIR.

Timings include exact embedding-vector matching and an **offline estimate** of
model processing plus matching. They exclude image decoding, UI rendering,
SQLite I/O and actual Web latency. Do not quote them as Web latency.

## Experiment protocol and limitations

- Pretrained YuNet and SFace, not trained from scratch.
- LFW funneled, custom identity-disjoint validation/test split, 2 templates per
  identity by default; **not the official LFW ten-fold benchmark**.
- The benchmark uses `lfw-largest` to select the primary face in LFW crops;
  normal Web enrollment and authentication keep the `strict` single-face rule.
- Reported 1:N 10/50/100 galleries have limited **known** query counts;
  0/300 false unknown acceptances is not proof of zero real-world risk.
- No liveness/PAD or production-grade authentication hardening is implemented.

Only aggregated JSON values are read. No raw faces, embeddings, database,
authentication log, or individual identities are accessed by the dashboard.
Do not publicly upload a biometric SQLite database or real face photos.
