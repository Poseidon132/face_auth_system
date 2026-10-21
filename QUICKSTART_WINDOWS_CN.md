# Windows 11 + VSCode 运行说明

假定项目解压路径：`E:\AIdemo\face_auth_system`。项目可以放在其他目录，修改 `cd` 即可。使用新的 Conda 环境，**不会影响原来的 CLIP × BLIP 工程**。

## 首次配置

```powershell
cd E:\AIdemo\face_auth_system
conda create -n faceauth python=3.11 -y
conda activate faceauth
python -m pip install -r requirements-dev.txt
python scripts/download_models.py
python scripts/verify_env.py
python -m pytest -q
python scripts/smoke_test.py
```

下载 SFace 初次约需 39 MB。模型从 OpenCV 官方仓库下载并校验 SHA-256。若下载失败，阅读根目录 README 的“模型下载失败”部分。

## 启动 Streamlit

```powershell
cd E:\AIdemo\face_auth_system
conda activate faceauth
streamlit run app.py --server.fileWatcherType none
```

浏览器打开 `http://127.0.0.1:8501`，依次进行“人员注册”“身份验证 1:1”“身份识别 1:N”。请先注册两位本人同意参与实验的人员。普通照片认证不具备活体防伪能力。

## 测试实际 DNN 推理

```powershell
python scripts/verify_env.py --image "E:\MyPhotos\selfie1.jpg"
python scripts/demo_pair.py "E:\MyPhotos\selfie1.jpg" "E:\MyPhotos\selfie2.jpg"
```

同一人的两张照片应通常具有更高余弦相似度，具体是否通过取决于质量和阈值。此处**不要填入事先假定的准确率**。

## 本地摄像头实时预览

```powershell
python scripts/webcam_demo.py --mode identify --camera 0
```

按 `Q` 或 `ESC` 结束。Windows 需要允许摄像头访问。不要把这个演示作为门禁/支付的身份认证依据。

## 常见问题

- **`ModuleNotFoundError`**：确认终端显示 `(faceauth)`，再执行 `python -m pip install -r requirements-dev.txt`。
- **`FaceDetectorYN` 或 `FaceRecognizerSF` 不存在**：检查 `python -c "import cv2;print(cv2.__version__)"`，建议使用 `opencv-python` 4.8 以上，避免同环境装多个冲突的 OpenCV wheel。
- **模型解析失败**：不能把 Git LFS 的指针文本当 ONNX。运行 `python scripts/download_models.py --verify-only` 核查哈希。
- **`SSL` / HF 下载失败**：浏览器分别从 YuNet/SFace 官方页点击 Download 下载两个 `.onnx`，手动保存到 `models/`，然后 SHA 校验。不要设置 `sslVerify=false` 或跳过证书验证。
- **检测不到人脸**：换光线充足、正对镜头、清晰的人脸图片。检测框太小会拒绝。
- **多人照片失败**：为了避免选错人，必须只有一张可检测人脸。
- **摄像头被占用**：关闭微信/会议软件或 Windows 摄像头应用，检查隐私权限。
- **`torch.classes` Streamlit 文件监控异常**：本项目 `.streamlit/config.toml` 已关闭自动文件监控，也可显式添加 `--server.fileWatcherType none`。
- **数据删除**：在 Web 的“数据库与日志”删除个人模板；整个实验结束后也可关闭应用并删除 `data/faceauth.sqlite3`，删除前请注意备份需求。

## 上传 GitHub

项目的 `.gitignore` 已排除 `models/`、`data/`、个人照片及数据库，正式发布时仍应运行 `git status --short` 检查暂存文件。上传前务必确认没有包含未经同意的生物特征数据。

## v1.6 dashboard

After the LFW 1:1 / 1:N tests, the normal Web application has a read-only
aggregate-metrics dashboard inside the original Experiments tab. It will
load `reports/lfw_metrics_test_lfwprimary.json` and
`reports/lfw_1n_test.json`, or show a missing-report message if absent.
It never opens face images or SQLite.

```powershell
cd E:\AIdemo\face_auth_system
conda activate faceauth
streamlit run app.py --server.fileWatcherType none
```

Read `docs/LFW_DASHBOARD_V16.md` for all details. Reported model-plus-search
latency is an offline estimate and **not** Web end-to-end latency.
