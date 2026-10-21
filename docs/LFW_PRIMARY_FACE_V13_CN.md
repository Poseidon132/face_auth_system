# FaceAuth v1.3: LFW multi-box analysis and offline face selection

The v1.2 test run reported 228 of 1000 LFW development test pairs skipped; all reported the `multiple_faces_detected` category. The original FaceEngine rejects any image with more than one YuNet detection **before evaluating box size**. Lowering YuNet score to 0.65 on validation increased the skipped-pair count to 656/2200 (29.82%). This may indicate false detections or duplicated boxes; examine the saved diagnostic images before drawing a conclusion.

This patch adds an **LFW-only offline evaluation option** `--face-selection lfw-largest` that picks the largest detector box in the already face-centered LFW funneled crop. It does not change the original live authentication default (`strict`) and does not claim that choosing the biggest face is safe for real multiple-person photos. There is no threshold change unless you separately calibrate on validation.

## Windows paths

- Project: `E:\AIdemo\face_auth_system`
- Download directory: `D:\Edge`
- Archive: `D:\Edge\face_auth_lfw_v1_3_primary_face_patch.zip`
- LFW images: `E:\AIdemo\face_auth_system\data\lfw_home\lfw_funneled`

## Install

```powershell
cd E:\AIdemo\face_auth_system
conda activate faceauth
$zip = 'D:\Edge\face_auth_lfw_v1_3_primary_face_patch.zip'
Test-Path -LiteralPath $zip
# Continue only if True:
Copy-Item .\faceauth\vision.py .\faceauth\vision_v12_backup.py -Force
Copy-Item .\scripts\eval_pairs.py .\scripts\eval_pairs_v12_backup.py -Force
Expand-Archive -LiteralPath $zip -DestinationPath . -Force
python -m pytest -q
```

## Step 1: Look at detected face boxes from validation images

```powershell
python scripts/inspect_lfw_detections.py `
  --csv data/lfw_pairs/val_pairs.csv `
  --images data/lfw_home/lfw_funneled `
  --yunet-score 0.8 `
  --min-face-size 55 `
  --limit 12 `
  --output-dir data/lfw_detection_audit
```

Review `data\lfw_detection_audit\face_01.jpg` and the other annotated images, plus `audit.json`. Green is the largest box; orange is another candidate. Compare box size, center position, and the depicted person. The audit files contain real face photos and are intentionally saved under `data/`, excluded from Git. Do not upload them publicly.

## Step 2: Re-evaluate validation using the LFW-only policy

```powershell
python scripts/eval_pairs.py `
  --split val `
  --csv data/lfw_pairs/val_pairs.csv `
  --images data/lfw_home/lfw_funneled `
  --eval-yunet-score 0.8 `
  --eval-min-face-size 55 `
  --face-selection lfw-largest `
  --target-far 0.01 `
  --calibration-out reports/lfw_threshold_val_lfwprimary.json `
  --out reports/lfw_metrics_val_lfwprimary.json
```

Compare validation **coverage, number of multi-box images resolved, end-to-end TAR/FAR, and successful-pair ROC-AUC**. Do not choose based only on valid-pair accuracy. The resulting `lfw_threshold_val_lfwprimary.json` records the new policy and cannot be used with strict-mode evaluation.

## Step 3: Test only after selecting the policy on validation

```powershell
python scripts/eval_pairs.py `
  --split test `
  --csv data/lfw_pairs/test_pairs.csv `
  --images data/lfw_home/lfw_funneled `
  --eval-yunet-score 0.8 `
  --eval-min-face-size 55 `
  --face-selection lfw-largest `
  --threshold-file reports/lfw_threshold_val_lfwprimary.json `
  --out reports/lfw_metrics_test_lfwprimary.json
```

The original test set has already been examined during development. Subsequent comparisons on that set are exploratory and must not be described as a new blinded benchmark. This is the **LFW development pairs protocol**, not the official LFW 6000-pair 10-fold protocol.

## Privacy and deployment separation

- All original photos remain locally in `data/`; no face photos or feature databases belong in a public GitHub repository.
- `FaceAuthService` and Streamlit continue using `FaceEngine.extract(... selection_policy='strict')` by default; `lfw-largest` is available only via the offline evaluation flag and is restricted to LFW funneled crop paths.
- The Web application's default 1:N threshold should **not** automatically be replaced by the 1:1 LFW evaluation threshold; a separate open-set identification study is required.
- This system has no reliable presentation-attack detection; it is an offline research and educational prototype.

## Verification status

- Offline code tests pass; real YuNet and SFace inference on Windows and the real LFW images must be executed on the user's local machine.
- No new validation/test accuracy results have been measured by this patch.
