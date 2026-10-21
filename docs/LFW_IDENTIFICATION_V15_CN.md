# FaceAuth v1.5：LFW 1:N 身份识别与未知身份拒识评估

这份补丁只增加离线评估程序，不修改 `app.py`、SQLite、人脸注册、Web 1:1/1:N 身份认证和原有 `eval_pairs.py`。

## 1. 评估内容

- 数据：已解压的 LFW funneled，路径 `data/lfw_home/lfw_funneled`，**无需重新下载**。
- 算法：预训练 YuNet + SFace，L2 归一化；每个身份用 `2` 张照片注册，至多 `2` 张不重复的照片查询，按模板的**最大余弦相似度**作为该身份得分。这与 Web 检索的多模板得分规则相同。
- 规模：默认 `10/50/100` 个已注册身份（100 是本补丁的主要规模上限）。同一阶段中的身份库按 10 ⊂ 50 ⊂ 100 扩展。
- 已知身份：验证阶段和测试阶段使用**不同人员**，并保证同一人的注册照片与查询照片不重复。
- 未知身份：验证、测试分别选取默认 `300` 个独立的、没有进入任何注册库的人员（每人 1 张查询照片）；不同阶段身份互不重叠。
- 校准：只用**验证阶段未知人员最大相似度**，为每个身份库规模选择独立的阈值。目标验证 FPIR 默认为 `0.01`，并不代表真实应用中的安全保证。测试阶段固定使用阈值文件，不重新调参。
- 检测策略：默认 `lfw-largest`（仅用于以主体人脸为中心裁剪的 LFW 基准）；线上身份注册/验证仍使用 `strict` 多人脸拒绝规则，**两类结果不可直接混用**。可在离线评估时指定 `--face-selection strict` 对比，但需用这个策略重新校准阈值。
- 测试协议：这是**自定义身份不重叠 1:N 划分**，不是 LFW 官方 10 折 1:1 测试。SFace 是预训练模型，不是在本工程里从零训练；此前已使用同一 LFW 数据族调试检测策略，本次属于开发性评估，不应当作全新、封闭的外部测试。

## 2. 安装补丁（Windows / VSCode）

你的本机实际路径：

- 工程：`E:\AIdemo\face_auth_system`
- Edge 下载：`D:\Edge`

把 `face_auth_lfw_v1_5_identification_patch.zip` 下载到 `D:\Edge`，运行：

```powershell
cd E:\AIdemo\face_auth_system
conda activate faceauth
$zip = 'D:\Edge\face_auth_lfw_v1_5_identification_patch.zip'
Test-Path -LiteralPath $zip
Expand-Archive -LiteralPath $zip -DestinationPath . -Force
python -m pytest -q
python scripts/eval_identification.py --help
```

这次压缩包只新增文件，**不需要再次解压 v1.1～v1.4，也不需要重新建立环境和下载 LFW**。如果 `matplotlib` 没安装，指标 JSON/CSV 仍能正常生成，只是跳过 PNG 图；可按需安装：`python -m pip install matplotlib`。

### 一键执行方式（推荐）

如果不需要自定义参数，可直接执行：

```powershell
cd E:\AIdemo\face_auth_system
conda activate faceauth
powershell -ExecutionPolicy Bypass -File .\scripts\run_lfw_1n_windows.ps1
```

它会先在验证集选择 10/50/100 人规模的阈值，成功后自动执行测试集，生成 JSON、CSV 和（安装 matplotlib 后）PNG。
`-ExecutionPolicy Bypass` 只作用于这个新启动的 PowerShell 进程，不会改变你电脑的全局执行策略。如果喜欢手动一步一步测试，请使用下面的两条 Python 命令。

## 3. 先在验证集校准不同库规模阈值

```powershell
cd E:\AIdemo\face_auth_system
conda activate faceauth

python scripts/eval_identification.py `
  --stage val `
  --images data/lfw_home/lfw_funneled `
  --sizes 10 50 100 `
  --enroll-per-person 2 `
  --queries-per-person 2 `
  --unknown-queries 300 `
  --seed 20261021 `
  --target-fpir 0.01 `
  --face-selection lfw-largest `
  --calibration-out reports/lfw_1n_thresholds_val.json `
  --out reports/lfw_1n_val.json
```

验证阶段的已注册人员与测试阶段不同；该命令针对库规模 10、50、100 各保存一个验证集阈值。

## 4. 在测试集上固定阈值验证

```powershell
python scripts/eval_identification.py `
  --stage test `
  --images data/lfw_home/lfw_funneled `
  --sizes 10 50 100 `
  --enroll-per-person 2 `
  --queries-per-person 2 `
  --unknown-queries 300 `
  --seed 20261021 `
  --target-fpir 0.01 `
  --face-selection lfw-largest `
  --threshold-file reports/lfw_1n_thresholds_val.json `
  --out reports/lfw_1n_test.json
```

如果测试命令报告“校准文件与当前测试配置不匹配”，请检查 `--sizes`、`--seed`、检测参数、图片目录及 LFW 图片清单是否发生变化；**不要删除检查或使用测试集重选阈值**。

## 5. 输出文件与指标解释

```text
reports/
  lfw_1n_thresholds_val.json  # 校准阈值，供 test 阶段读取
  lfw_1n_val.json             # 验证集完整结果
  lfw_1n_val.csv              # 表格型验证结果
  lfw_1n_val.png              # 可选：不同注册人数的指标曲线
  lfw_1n_val_timing.png       # 可选：耗时曲线
  lfw_1n_test.json            # 测试集完整结果
  lfw_1n_test.csv             # 表格型测试结果
  lfw_1n_test.png             # 可选：测试指标曲线
  lfw_1n_test_timing.png      # 可选：测试耗时曲线
```

JSON/CSV **不会写入人脸图片、原始特征向量、人员真实姓名、完整人员名单或单条查询明细**，仅包含每个库规模的聚合指标和实验配置哈希。原有 Web SQLite 库不会被访问。生成结果均在 `.gitignore` 忽略的 `reports/` 下，需要公开时请单独确认内容。

| 指标字段 | 含义 |
|---|---|
| `closed_set_rank1_end_to_end` | 所有已知身份查询中（含检测失败），最近邻身份正确的比例；**未应用拒识阈值** |
| `dir_at_threshold` | 已知身份查询中，“最近邻身份正确且得分达到阈值”的比例（DIR） |
| `known_wrong_id_rate` | 已知身份查询中，错误识别成其他已注册人员且超过阈值的比例 |
| `known_rejection_rate` | 已知身份查询中，被拒绝的比例，含检测失败 |
| `fpir` | 未注册身份查询中，被误接受为任一已注册人员的比例（FPIR）；检测失败按拒绝处理 |
| `unknown_correct_reject_rate` | 所有未知身份查询中的正确拒识率，等于 `1-FPIR` |
| `known_coverage` / `unknown_coverage` | 两类查询中成功提取人脸特征的比例（别让失败图片藏在指标分母之外） |
| `vector_match_only` | 在 CPU 上对已提取向量进行 NumPy 精确检索的耗时，不含模型运行 |
| `detection` / `recognition` | YuNet 检测、SFace 对齐及提特征的计时 |
| `estimated_query_model_plus_match` | 测得模型运行时间与对应规模向量检索时间相加的**离线估算**；不含图片读取、浏览器渲染、SQLite 或缓存操作，**并非 Web 端到端延迟** |

每个身份库规模的验证 FPIR 可与 `target-fpir=0.01` 对照；测试 FPIR 会自然波动，**可能超过 1%**。300 张未知查询只能提供有限统计精度，且人物来源与场景并不覆盖真实业务部署条件，不能以此宣称实际系统具备“1% 的安全误识率”。

## 6. 后续工作

1. 把 `lfw_1n_test.csv` 的真实指标整理成 10/50/100 人性能对比表。
2. 对比 `lfw-largest` 与 `strict` 的检测覆盖率，解释基准评估与 Web 实际运行策略的差异。
3. 使用本人或知情同意照片验收 Streamlit 已注册/未注册人员、摄像头、删除数据、日志，并注意未知身份阈值在不同注册库规模下的变化。
4. 若需要真实认证安全评估，还应额外建立授权采集、反照片攻击/活体检测、隐私访问控制及跨环境采样方案。

**不要声称 1:N 代码中的模型是从零训练；也不要拿此前 1:1 的 98.60% 代替这次的 1:N Rank-1/DIR。**
