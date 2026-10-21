# 自动化测试记录

## 已验证（构建环境）

- `python -m pytest -q`：**32 passed**（本地依赖 OpenCV 4.13.0 / NumPy 2.3.5 / Python 3.13.5）。
- `python scripts/smoke_test.py`：使用可预测的合成向量模拟注册、1:1 验证、1:N 检索、未知拒识、审计与删除。
- `python -m compileall -q app.py faceauth scripts`：检查 Python 源文件语法。
- 样本输入解析、异常图片检查、数据库完整性、特征版本隔离、阈值计算与向量相似度具有独立测试。

## 未验证（需要用户本地执行）

- 构建环境无法从 Hugging Face / OpenCV Zoo 取回真实 ONNX 权重，未完成本环境的 YuNet+SFace 真正推理。
- 构建环境未安装 Streamlit，因此页面布局尚未经过浏览器端实测。
- 没有经过许可的真实人脸验证/测试配对集，不会伪造准确率、FAR、FRR 或处理时延。
- 摄像头驱动、Windows 图形窗口与 GPU/CPU 使用情况需在本地 Windows 机器验证。

## Windows 真正验收步骤

```powershell
python scripts/download_models.py
python scripts/verify_env.py
python scripts/verify_env.py --image "E:\MyPhotos\selfie1.jpg"
python scripts/demo_pair.py "E:\MyPhotos\selfie1.jpg" "E:\MyPhotos\selfie2.jpg"
python -m pytest -q
python scripts/smoke_test.py
streamlit run app.py --server.fileWatcherType none
```

界面验收：完成注册 A、注册 B，A 通过自己身份的验证、A 不通过 B 的验证、已注册的 A 可检索、未知人员拒识、导出日志、删除 A 后其模板无法再检索。浏览器摄像头采集和 OpenCV webcam 预览各验证一次。
