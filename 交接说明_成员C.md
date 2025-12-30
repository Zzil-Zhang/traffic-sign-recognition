# 飞桨交通标志项目说明（评估与超参调优）

主要功能：
- 评估已训练的飞桨 CNN 模型，并与基准 SVM+HOG 模型对比（含混淆矩阵、预测样本、ROC 对比、F1 指标等）
- 对 CNN 模型进行超参数调优与 K 折交叉验证（包含过拟合/欠拟合分析和可视化）

---

## 目录结构与关键文件

请确保以下文件存在于同一项目根目录中（或在 README 指定的路径下）：

- 评估脚本与 Notebook
  - `evaluate_model_paddle.py`（命令行脚本，直接运行）
  - `evaluate-model.ipynb`（Jupyter 版本，分步骤执行）
- 超参数调优
  - `hyperparameter_tuning_final_paddle.py`（命令行脚本）
  - `hyperparameter_tuning_final_paddle.ipynb`（Jupyter 版本）
- 训练/数据/基准依赖
  - `data_utils.py`（数据加载与预处理）
  - `cnn_model_paddle.py`（模型结构定义：SimpleCNNPaddle / TrafficCNNPaddle 等）
  - `baseline_model_fixed.py`（基准 SVM+HOG 模型与评估工具）
  - 处理好的数据集（必须存在）
    - `/home/aistudio/work/processed_data/` 目录下包含：
      - `X_train.npy`, `y_train.npy`
      - `X_val.npy`, `y_val.npy`
      - `X_test.npy`, `y_test.npy`
  - 训练好的 CNN 参数文件（至少存在一个）
    - 常见路径（脚本会自动查找）
      - `trained_models/traffic_sign_cnn_paddle_final_*.pdparams`
      - `trained_models/traffic_sign_cnn_paddle_best_*.pdparams`
      - `trained_models/my_traffic_classifier_paddle.pdparams`
      - 或项目根目录的 `my_traffic_classifier_paddle.pdparams`, `model_final.pdparams`, `my_traffic_classifier.pdparams`
  - 基准模型文件（可选，如不存在自动跳过基准评估）
    - `/home/aistudio/work/fixed_baseline_model/model.pkl`

---

## 环境要求

- Python 3.8+（建议 >=3.10）
- PaddlePaddle
  - CPU 环境（通用）：`pip install paddlepaddle`
  - GPU 环境（若为依图加速卡 iluvatar）：需安装 Paddle 自定义设备插件（环境可能已预装）
    - 参考你所在平台的安装说明；脚本会自动尝试设置 `iluvatar_gpu:0`，失败则回退到 CPU
- 其他依赖包
  - `pip install numpy matplotlib seaborn scikit-learn ipython`
  - 如需使用中文字体显示（可选）：系统安装 `WenQuanYi` 或 `SimHei` 等字体

注意：
- 若你不使用依图加速卡，可将脚本中的 `paddle.set_device('iluvatar_gpu:0')` 改为 `paddle.set_device('gpu')` 或保持自动回退到 `cpu`。
- Jupyter Notebook 中使用了 `IPython.display` 的 `Markdown`，确保 `ipython` 已安装。

---

## 快速开始（命令行）

1. 安装依赖（示例）
   ```bash
   python -m venv .venv && source .venv/bin/activate  # Windows 用 .venv\Scripts\activate
   pip install paddlepaddle numpy matplotlib seaborn scikit-learn ipython
   ```
2. 准备数据与模型（确保路径与文件存在）
   - 数据：`/home/aistudio/work/processed_data/` 下的 `X_*`、`y_*` `.npy`
   - 基准：`/home/aistudio/work/fixed_baseline_model/model.pkl`（可选）
   - CNN 参数：参考前述模型文件路径
3. 评估 CNN 与（可选）基准 SVM+HOG
   ```bash
   python evaluate_model_paddle.py
   ```
   成功运行后，将在 `evaluation_results/` 目录下生成：
   - `confusion_matrix.png`（混淆矩阵）
   - `prediction_samples.png`（预测样本示例）
   - `model_comparison_metrics.png`（指标对比：Accuracy / F1）
   - `roc_overlay.png`（CNN vs 基准微平均 ROC）
   - `baseline_evaluation_results/`（若基准存在，生成其详细评估输出）

4. 超参数调优（交互式菜单）
   ```bash
   python hyperparameter_tuning_final_paddle.py
   ```
   按提示选择：
   - `K折交叉验证`
   - `随机搜索`
   - `分层搜索`
   输出保存到：
   - `/home/aistudio/work/hyperparameter_tuning_result/`（各类图表与 JSON/CSV 结果）
   - `/home/aistudio/work/recommended_parameters_for_memberB.txt`（给成员B的参数建议）

---

## 快速开始（Jupyter Notebook）

### 评估 Notebook：`evaluate-model.ipynb`

按顺序运行，确保不遗漏定义与依赖：

1. 导入和全局配置（字体/设备设置，依赖导入）
2. 导入项目模块（`data_utils.py`、`cnn_model_paddle.py`、`baseline_model_fixed.py`）
3. 数据加载与预处理（加载 `processed_data/*.npy` 并应用与训练一致的预处理）
4. 基础工具函数（务必运行，确保后续函数可用）
   - 包含：`compute_multiclass_auc`、`RenameUnpickler`、`prepare_pickle_class_aliases`、`to_hwc_batch`、`ensure_unit_range` 等
   - 注意：若你将工具函数单元格标注为“注释示例”，请取消注释或保证其它单元格有相同函数定义，并在执行 `evaluate_cnn_model` 之前已经运行
5. 加载模型（自动搜索 CNN 参数文件）
6. 评估 CNN（打印损失/准确率/分类报告，生成图表）
7. 基准评估（若基准存在：加载 model.pkl 并评估/生成图表）
8. 指标对比与 ROC 曲线
9. 一致性验证与汇总文件（保存 `evaluation_results/metrics_summary.json`）

运行建议：
- 使用 `Kernel -> Restart & Run All`，确保函数定义的单元格先于使用位置执行
- 若出现 `NameError: compute_multiclass_auc not defined`，说明未运行包含该函数的单元格。请在“评估 CNN 模型”单元格之前执行“基础工具函数”（或任何包含该函数定义的单元格）

### 调优 Notebook：`hyperparameter_tuning_final_paddle.ipynb`

按顺序运行：
1. 设备设置（若有依图 GPU，将自动使用；否则回退到 CPU）
2. 超参数调优器定义（包含 K 折交叉验证、随机搜索、分层搜索等）
3. 执行相应函数进行调优
4. 输出图表与结果文件位于 `/home/aistudio/work/hyperparameter_tuning_result/`

---

## 路径与参数约定

- 数据目录（必须）：`/home/aistudio/work/processed_data/`
  - 若你的数据目录不同，请修改脚本/Notebook 中相应路径（例如 `data_path` 变量）
- 基准模型（可选）：`/home/aistudio/work/fixed_baseline_model/model.pkl`
  - 若文件不存在，会自动跳过基准评估，不影响 CNN 评估流程
- 模型参数文件（至少存在一个）
  - 脚本/Notebook 会自动搜索常见文件名；也可在 `evaluate_model()` 中显式传入路径
- 输出目录
  - 评估：`evaluation_results/`（脚本自动创建）
  - 调优：`/home/aistudio/work/hyperparameter_tuning_result/`（脚本自动创建）

---

## 常见问题与解决

- 设备错误（iluvatar GPU 未找到）
  - 日志类似：`无法设置依图加速卡设备，使用CPU`
  - 说明环境未安装或未识别自定义设备插件；脚本已自动回退 CPU，可正常运行
  - 若希望使用 GPU：请按平台要求安装 `paddle_custom_device` 并配置 `CUSTOM_DEVICE_ROOT`
- `NameError: compute_multiclass_auc not defined`
  - 说明未执行包含该函数定义的单元格（或该单元格被注释）
  - 解决：在 Notebook 中，确保“基础工具函数”单元格已执行；或将该函数定义上移到更早的单元格，并执行
- 找不到模型参数文件
  - 确保至少存在一个 `.pdparams` 文件，并在约定路径中；或修改评估函数里 `possible_models` 列表以匹配你的文件名
- 找不到基准模型（不会阻塞）
  - 若日志提示：`未找到基准模型文件，跳过基准评估`，仅影响与 SVM+HOG 的对比，不影响 CNN 评估
- 字体与中文显示异常（图表中中文乱码）
  - 脚本会自动选择已安装字体；若无中文字体，则使用 `DejaVu Sans`
  - 可在系统安装 `WenQuanYi` 或 `SimHei` 字体以获得更佳显示

---