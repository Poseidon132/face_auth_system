# Reproduced LFW evaluation summaries (v1.6)

The values in this folder were transcribed from the **actual Windows CLI outputs supplied by the project developer**. They are **aggregate numbers only**. The underlying pair images, identity labels, feature vectors, raw scores, private SQLite database and full `reports/` contents are not published.

## Custom 1:1 verification

LFW `pairsDevTrain` (2,200 development pairs) was used for threshold calibration; `pairsDevTest` (1,000 development pairs) for evaluation. The threshold was 0.3028 and the test yielded Accuracy 0.9860, TAR 0.9820, FAR 0.0100, FRR 0.0180, ROC-AUC 0.9937, coverage 1.0. From the balanced test split, counts are TP 491, FN 9, FP 5, TN 495. This uses `lfw-largest` subject-face selection. It is **not** the official LFW ten-fold benchmark.

## Custom 1:N identification

The 1:N evaluator used identity-disjoint validation/test pools, two enrollment images per person, up to two held-out known queries per identity and 300 unknown identity queries at each gallery size. Thresholds were calibrated on validation by gallery size. See the CSV in this folder for observed metrics. For 100 enrolled people, 170/171 known queries were correctly accepted (DIR 99.4152%) and 1/300 unknown query was incorrectly accepted (FPIR 0.3333%). A low FPIR measured on 300 queries does not prove sufficient resistance to fraud or impersonation attacks.

## Generalization and timing caveats

- Live Web verification **rejects multi-face images**; LFW subject-centered evaluation may select the largest face. These are different operating policies.
- The 1:N gallery samples are relatively small and drawn from the same source dataset; robustness under demographic, camera, lighting and presentation-attack changes has not been measured.
- SFace is a pretrained model; this is a model-integration and matching-system project, not newly trained recognition-network research.
- The Web 1:N default threshold is a demo setting; the 1:N benchmark thresholds are not transferable without independent local validation.
- Offline model-plus-search timing is a pipeline estimate, not the browser end-to-end latency. Refer to `scripts/run_lfw_1n_windows.ps1` for repeatable reporting.

Files: `lfw_1to1_test.csv`, `lfw_1ton_test.csv`.
