# FaceAuth v1.2: LFW failure diagnostics

This optional patch upgrades only `scripts/eval_pairs.py` and adds regression tests. It preserves the existing YuNet and SFace weights, Streamlit app, database, and the default detector settings (YuNet score 0.8, minimum face width/height 55 px).

## Installation (PowerShell)

The local archive directory here matches the actual Windows download directory reported in the project conversation (`D:\Edge`). Ensure the zip has first been downloaded there.

```powershell
cd E:\AIdemo\face_auth_system
conda activate faceauth
$zip = 'D:\Edge\face_auth_lfw_v1_2_diagnostics_patch.zip'
Test-Path -LiteralPath $zip
Copy-Item .\scripts\eval_pairs.py .\scripts\eval_pairs_v11_backup.py -Force
Expand-Archive -LiteralPath $zip -DestinationPath . -Force
python -m pytest -q
```

## Step 1. Diagnose the current baseline without modifying the threshold

```powershell
python scripts/eval_pairs.py `
  --split test `
  --csv data/lfw_pairs/test_pairs.csv `
  --images data/lfw_home/lfw_funneled `
  --threshold-file reports/lfw_threshold_val.json `
  --out reports/lfw_metrics_test_diagnostics.json
```

New fields include `coverage_diagnostics.failure_by_reason`, `coverage_diagnostics.failure_by_label`, and `end_to_end_reject_on_failure`. The latter counts images that fail detection/embedding as rejected verification attempts. It therefore lowers TAR when a positive pair is skipped and counts skipped negative pairs as correctly rejected. `accuracy` at the top level is intentionally unchanged and still refers to valid pairs only.

## Step 2. Optional candidate detector settings: validation first

Do not change the Web app detector settings or the original calibrated threshold. Use separate filenames for experimental results.

```powershell
python scripts/eval_pairs.py `
  --split val `
  --csv data/lfw_pairs/val_pairs.csv `
  --images data/lfw_home/lfw_funneled `
  --eval-yunet-score 0.65 `
  --eval-min-face-size 35 `
  --target-far 0.01 `
  --calibration-out reports/lfw_threshold_val_candidate.json `
  --out reports/lfw_metrics_val_candidate.json
```

Examine coverage, end-to-end TAR, and false accept rate on `val`. Only if the candidate is worthwhile, evaluate `test` with **the same detection settings and newly calibrated threshold**:

```powershell
python scripts/eval_pairs.py `
  --split test `
  --csv data/lfw_pairs/test_pairs.csv `
  --images data/lfw_home/lfw_funneled `
  --eval-yunet-score 0.65 `
  --eval-min-face-size 35 `
  --threshold-file reports/lfw_threshold_val_candidate.json `
  --out reports/lfw_metrics_test_candidate.json
```

A mismatched detector setting between calibration and test is rejected. The old threshold JSON is recognized only with the original detector defaults. A test set evaluated multiple times is exploratory, not a fresh blind test.

## Interpretation and limits

- The data are the LFW development verification pairs, not the official 10-fold 6,000-pair protocol.
- `accuracy`, `far`, `frr` and `roc_auc` apply to valid pairs only. `end_to_end_reject_on_failure` covers all attempted pairs under an explicit reject-on-failure rule.
- The validation-calibrated verification threshold does not establish security guarantees for open-set 1:N identification or for multi-template matching.
- Original user photos and any biometric embeddings must not be pushed to GitHub.
- Streamlit remains at its original, conservative demonstration settings until a separate usability/security review.
