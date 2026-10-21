# FaceAuth 人脸数据集准备：LFW

前一个 FaceAuth 工程包只有代码，没有真实人脸数据集。本补丁提供**下载与转换脚本**，图片由使用者从 LFW 数据来源自行下载，不包含在 GitHub 或 ZIP 中。

## 选择 LFW 的原因

LFW（Labeled Faces in the Wild）约有 13,233 张、5,749 个身份的人脸图像，原始图片为 250×250 JPEG。工程使用 YuNet + SFace 做真实照片检测和特征提取，需要原始分辨率或 funneled 对齐版；不要使用默认 62×47 的 sklearn 裁剪缩小图像。 

- `lfw-funneled.tgz`：对齐版照片，约 200 MB，校验后解压到 `data/lfw_home/lfw_funneled/`。
- `pairsDevTrain.txt`：开发训练配对，转换为 `data/lfw_pairs/val_pairs.csv`，用于选择余弦阈值。
- `pairsDevTest.txt`：开发测试配对，转换为 `data/lfw_pairs/test_pairs.csv`，用于以固定阈值评估。
- `data/lfw_pairs/debug_val_30_pairs.csv`：调试用 30 对样本，**不用于正式准确率报告**。

下载源和 SHA-256 与 scikit-learn `sklearn/datasets/_lfw.py` 使用的 Figshare 下载源一致：

- `https://ndownloader.figshare.com/files/5976015` → `lfw-funneled.tgz`
- `https://ndownloader.figshare.com/files/5976012` → `pairsDevTrain.txt`
- `https://ndownloader.figshare.com/files/5976009` → `pairsDevTest.txt`

此处的 `val/test` 对应 **LFW 开发配对文件**，**不是**官方 LFW 6000 对、10 折交叉验证基准测试结果。不能直接将本工程跑出的开发配对分数宣称为“LFW 标准 benchmark”。

## Windows 11 + VSCode

请先解压主项目，再将本补丁覆盖到项目根目录：

```powershell
cd E:\AIdemo\face_auth_system
conda activate faceauth
python scripts/download_models.py
python scripts/prepare_lfw_dataset.py
```

下载过程约 200 MB，具体耗时依赖网络。脚本自动校验 SHA-256、拒绝不安全 tar 路径、生成评估 CSV。成功会输出 `val`/`test` 的正负配对数。

### 网络无法下载时

可用浏览器打开上述三个 Figshare 链接，将文件分别存储为：

```text
data/lfw_home/lfw-funneled.tgz
data/lfw_home/pairsDevTrain.txt
data/lfw_home/pairsDevTest.txt
```

然后运行（自动校验哈希并解压）：

```powershell
python scripts/prepare_lfw_dataset.py --no-download
```

**不要关闭 HTTPS 证书校验。** 如果文件被浏览器命名为数字或其他扩展名，请重命名为上述名字。哈希校验失败意味着文件错误或下载不完整，不应忽略。

## 1. 先测少量样本（只检查模型推理是否正常）

```powershell
python scripts/eval_pairs.py `
  --split val `
  --csv data/lfw_pairs/debug_val_30_pairs.csv `
  --images data/lfw_home/lfw_funneled `
  --target-far 0.01 `
  --calibration-out reports/threshold_debug_only.json `
  --out reports/metrics_debug_only.json
```

这个输出仅用于检查代码和人脸检测是否正常，样本极少，不要用于项目的准确率结论。

## 2. 用开发验证配对选择阈值

```powershell
python scripts/eval_pairs.py `
  --split val `
  --csv data/lfw_pairs/val_pairs.csv `
  --images data/lfw_home/lfw_funneled `
  --target-far 0.01 `
  --calibration-out reports/lfw_threshold_val.json `
  --out reports/lfw_metrics_val.json
```

## 3. 独立开发测试配对评估（不重新调参）

```powershell
python scripts/eval_pairs.py `
  --split test `
  --csv data/lfw_pairs/test_pairs.csv `
  --images data/lfw_home/lfw_funneled `
  --threshold-file reports/lfw_threshold_val.json `
  --out reports/lfw_metrics_test.json
```

报告会给出 `Accuracy`、`TAR`、`FAR`、`FRR`、`ROC-AUC`、`EER` 描述指标以及检测失败/跳过数量。**评估时不能忽略失败样本；应同时报告成功处理比例。**

## Web 注册与 1:N 识别数据

LFW 是来源于网络的研究用人脸数据，不建议拿来做真实人员注册与访问授权演示。Web 演示请使用自己的照片或明确授权的实验参与者照片，每人至少两张、不同光线角度；不要将照片、SQLite 特征数据库、数据集或生物特征文件提交至 GitHub。相关数据处理应遵循数据集条款及当地隐私规定。

## 来源与参考

- [LFW 数据集介绍](https://complexity.cecs.ucf.edu/lfw-labeled-faces-in-the-wild/)
- [scikit-learn LFW 原始下载源与 SHA-256](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/datasets/_lfw.py)
- [scikit-learn LFW 人脸配对数据说明](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_lfw_pairs.html)
- Huang et al. (2007), *Labeled Faces in the Wild: A Database for Studying Face Recognition in Unconstrained Environments*.
