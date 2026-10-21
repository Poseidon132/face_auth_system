# FaceAuth - Local Deep-Learning Face Identification Demo

**基于深度学习的人脸识别与身份认证系统**

**Stack:** Python 3.11, OpenCV DNN (YuNet / SFace), NumPy, SQLite, Streamlit. **Release:** v1.6.

This is a locally run, consent-based **research and portfolio prototype** for face matching. It includes registration, 1:1 verification, 1:N identification with unknown-person rejection, local audit records, and an experiment dashboard. The YuNet detector and SFace embedding model are **pretrained third-party models**, not architectures trained from scratch in this project. This repository implements image handling, feature extraction, multi-template matching, data management, and evaluation tooling.

**Security boundary:** A photo upload can pass the demo; it does not include reliable liveness detection, database encryption, administrator access control, or robust anti-spoofing. Do not use this prototype for real door access, attendance, payment or high-stakes authentication.

## Demonstration

The screenshot below intentionally includes **only aggregate experimental metrics**. It does not contain biometric photos or personal identity information.

![LFW results dashboard](docs/screenshots/dashboard.png)

The live app has five tabs: registration, 1:1 verification, 1:N identification, database and logs, and experiments. Please use **your own or explicitly consented images** for local trials. Registration and live verification reject multiple detected faces; the LFW benchmark has a separate documented subject-face selection mode.

## Evaluation results (observed, not simulated)

The following values came from local Windows runs on LFW using pretrained YuNet + SFace. Numeric source summaries are in [docs/benchmarks](docs/benchmarks/LFW_RESULTS.md); full local JSON reports remain in `reports/` and are deliberately excluded from Git to avoid inadvertently publishing sensitive material.

### 1:1 verification - custom LFW development test pairs

| Split | Pair count | Coverage | Accuracy | FAR | FRR | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Test | 1,000 | 100% | **98.60%** | 1.00% | 1.80% | 0.9937 |

The similarity threshold **0.3028** was determined on the development validation pairs and reused on the test pairs. This is **not** the official LFW 10-fold protocol. LFW evaluation uses a subject-centered `lfw-largest` detector policy; the live UI retains the stricter single-face policy. The reported accuracy is not an estimate of operational access-control security.

### 1:N identification - custom identity-disjoint LFW experiment

| Gallery identities | Known queries | Unknown queries | Rank-1 | DIR at threshold | Unknown FPIR |
|---:|---:|---:|---:|---:|---:|
| 10 | 16 | 300 | 100.00% | 100.00% | 0.67% |
| 50 | 84 | 300 | 100.00% | 100.00% | 0.00% |
| 100 | 171 | 300 | 100.00% | **99.42%** | 0.33% |

Each identity had two enrollment images and up to two separate known-query images. Unknown queries came from non-enrolled identities. Validation and test identity pools were disjoint; 1:N rejection thresholds were tuned separately at each gallery size using the validation split. In the 100-person test there were 170 correctly accepted known queries out of 171 and one falsely accepted unknown query out of 300. **Zero observed errors in 300 trials does not establish a zero real-world error rate.**

The 1:N evaluation uses the `lfw-largest` face-crop policy, not the live system's strict multi-face rejection rule. Offline model-plus-search timing in the experiment is an **estimate**, not Web request latency; the UI measures a separate local request timing.

## Core functionality and implementation

| Component | Implementation |
|---|---|
| Face detection | YuNet ONNX, face bounding box and 5 landmarks |
| Embeddings | SFace ONNX, aligned crop, feature normalization |
| Enrollment | Consent confirmation, person record, multiple embeddings |
| Verification (1:1) | Maximum cosine similarity among the claimed person's enrolled templates |
| Identification (1:N) | Exact cosine matching over the local gallery; threshold-based unknown rejection |
| Persistence | SQLite person codes, feature vectors and audit metadata; no original photo storage by default |
| Interface | Streamlit Web UI and optional OpenCV local camera demo |
| Experiments | 1:1 ROC/Accuracy/TAR/FAR/FRR; 1:N Rank-1/DIR/FPIR, coverage, timing |

No FAISS, distributed database, proprietary pretrained network, or production-ready anti-spoofing is claimed.

## Windows setup (VSCode / PowerShell)

```powershell
cd E:\AIdemoace_auth_system
conda create -n faceauth python=3.11 -y
conda activate faceauth
python -m pip install -r requirements-dev.txt
python scripts/download_models.py
python scripts/verify_env.py
python -m pytest -q
python scripts/smoke_test.py
streamlit run app.py --server.fileWatcherType none
```

Open **http://127.0.0.1:8501**. The app binds to localhost via `.streamlit/config.toml`. You need to download the public ONNX weights once; they are not packaged with this repository. The scripts verify model SHA-256 checksums.

For an existing `faceauth` Conda environment, skip environment creation and activate it. Start the read-only dashboard without loading the biometric models or database using:

```powershell
streamlit run experiment_dashboard.py --server.fileWatcherType none
```

### Reproduce the benchmark (downloads public LFW dataset)

```powershell
python scripts/prepare_lfw_dataset.py

python scripts/eval_pairs.py `
  --split val `
  --csv data/lfw_pairs/val_pairs.csv `
  --images data/lfw_home/lfw_funneled `
  --face-selection lfw-largest `
  --target-far 0.01 `
  --calibration-out reports/lfw_threshold_val_lfwprimary.json `
  --out reports/lfw_metrics_val_lfwprimary.json

python scripts/eval_pairs.py `
  --split test `
  --csv data/lfw_pairs/test_pairs.csv `
  --images data/lfw_home/lfw_funneled `
  --face-selection lfw-largest `
  --threshold-file reports/lfw_threshold_val_lfwprimary.json `
  --out reports/lfw_metrics_test_lfwprimary.json

# IMPORTANT: invoke this in the CURRENT activated (faceauth) PowerShell.
& .\scriptsun_lfw_1n_windows.ps1
```

The last script independently calibrates the 1:N thresholds on the validation identity pool and evaluates them on held-out test identities. Running `powershell -File ...` in a fresh shell can select the wrong Python environment; the call operator `&` in the activated shell avoids that issue. The LFW dataset is not in this repository; check its origin, license and usage restrictions separately.

## Project structure

```text
face_auth_system/
  app.py, experiment_dashboard.py
  faceauth/       YuNet/SFace inference, matching, SQLite, dashboard, metrics
  scripts/        model download, dataset preparation, evaluation and camera demo
  tests/          pytest tests (including GitHub privacy checks)
  docs/benchmarks/   non-biometric metric summaries only
  docs/screenshots/  approved dashboard-only public screenshot
  docs/         protocols, security/privacy notes and release instructions
  data/          local dataset and face-template storage (ignored by Git)
  models/        local ONNX weights (ignored by Git)
  reports/       local detailed evaluation outputs (ignored by Git)
```

## Privacy, limitations and credit

All face photos, extracted biometric vectors, local person IDs, SQLite files, full local reports and model weights are **excluded from the public repository**. Do not add face photographs to documentation or use portraits from the internet without permission and a compatible license. Embeddings are sensitive biometric information even without photos. See [Security and Privacy](docs/SECURITY_AND_PRIVACY_CN.md).

Because only face matching is implemented, this prototype is vulnerable to printed-photo and replay attacks. It does not provide verified liveness, production audit authorization, multi-factor identity proofing, encryption-at-rest or independent clinical/operational validation.

Third-party pretrained networks: OpenCV Zoo **YuNet** and **SFace**. Their provenance and separate licenses are in [THIRD_PARTY_MODELS.md](THIRD_PARTY_MODELS.md). No overall software license has been selected for this repository; model licenses do not automatically license this source code.

[Windows quickstart](QUICKSTART_WINDOWS_CN.md) · [1:N evaluation protocol](docs/LFW_IDENTIFICATION_V15_CN.md) · [GitHub publication guide](docs/GITHUB_PUBLICATION.md)
