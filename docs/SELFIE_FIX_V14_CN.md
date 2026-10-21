# v1.4：大幅竖屏自拍检测修复（Web 端）

## 原因

用户上传的实例图片为 921×2048；`read_photo()` 处理后为 720×1600。旧版 Web `FaceEngine.extract()` 将整张图以 720×1600 输入 YuNet。人脸尺寸占比大时可能漏检，因此出现“未检测到人脸”的提示。这不是相似度阈值 0.45 的问题，也不是 LFW 的多脸选择问题。

## 改动

- 在严格单人认证模式下，将大图片按长边 960、640、480 像素依次尝试检测（先用最大的有效输入）。
- 检测框以及五个关键点全部映射到原图像素坐标，SFace 继续对原图执行 `alignCrop()`，避免错误坐标或把特征图降采样到检测分辨率。
- 任一已使用的检测尺度发现两张或更多人脸时，仍拒绝认证；不会套用 `lfw-largest` 策略。
- LFW `lfw-largest` 模式原样保留，历史实验指标不因 Web 修复改变。
- `scripts/diagnose_photo_detection.py` 可以打印不同缩放尺度下 YuNet 的人脸检测数量和得分；默认不保存照片。

## 安装

在 `E:\AIdemo\face_auth_system` 使用：

```powershell
cd E:\AIdemo\face_auth_system
conda activate faceauth
$zip = "D:\Edge\face_auth_selfie_v1_4_patch.zip"
Test-Path -LiteralPath $zip
Copy-Item .\faceauth\vision.py .\faceauth\vision_v13_backup.py -Force
Expand-Archive -LiteralPath $zip -DestinationPath . -Force
python -m pytest -q
streamlit run app.py --server.fileWatcherType none
```

注意：如果 Streamlit 已在运行，请先 Ctrl+C 停止旧进程，再启动（`@st.cache_resource` 会缓存旧模型对象）。浏览器重新加载后，重新上传自拍。

## 诊断

```powershell
python scripts/diagnose_photo_detection.py "D:\Edge\1792477312233.jpg"
```

把该命令里的图片路径替换为照片的**真实路径**。如果原照片不在 `D:\Edge`，请不要假设它在那里。脚本输出原图缩放后的尺寸、原始和 960/640/480/320 等尺度检测框数量、实际检测用尺寸和 SFace 耗时。**不要将本人自拍、注册数据库或 embedding 上传公开 GitHub。**

## 限制

- 模型权重未随补丁提供。离线测试为模拟 YuNet/SFace，真实检测表现需要本机模型权重实测。
- 多尺度能缓解大图尺度问题，但不能保证任何照片都检出；如仍失败，可尝试移除耳机、保持镜头与眼睛同一高度、减少过曝或阴影。
- 人脸相似度阈值、反欺骗与活体检测问题与检测尺度独立。此工程只用于本地授权演示，不应直接部署成高安全身份认证。
