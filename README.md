# GTSRB 交通标志识别（一键复现指南）

本仓库实现了德国交通标志识别（GTSRB）的基准模型（HOG + SVM）与基于 PaddlePaddle 的轻量级 CNN 模型，并提供数据预处理、数据增强、超参数调优和评估脚本 / Notebook。  
仓库中已有更详细的模块说明，请参阅：`README_baseline.md`（基准模型）、`README_CNN.md`（CNN 模型）与 `README_evaluate.md`（评估与调优）。

---

## 快速前提
- 已获取原始 GTSRB 数据集：https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign
- 建议 Python >= 3.8（推荐 3.10+）。

---

## 一键复现（Unix / WSL / macOS）
在项目根目录下运行下列命令（将创建虚拟环境、安装依赖、执行预处理、可选数据增强、运行训练 Notebook 并评估）：

```bash
# 一键复现（bash）
python -m venv .venv && source .venv/bin/activate && \
pip install -U pip && pip install -r requirements.txt && \
# 数据预处理（必须）
python data_preprocessing.py && \
# 可选：批量数据增强（如果你想扩增数据）
python data_augment_run.py || true && \
# 运行训练 Notebook（需安装 jupyter）
jupyter nbconvert --to notebook --execute train_cnn_paddle.ipynb --ExecutePreprocessor.timeout=3600 --output executed_train.ipynb && \
# 五折交叉验证 + 超参数调优（交互或脚本/Notebook）
python hyperparameter_tuning_final_paddle.py && \
# 评估（依赖训练好的模型或执行上一步生成的模型文件）
python evaluate_model_paddle.py
```

说明：
- `data_preprocessing.py` 会生成 `processed_data/` 下的 `X_*.npy` / `y_*.npy` 等文件。
- `data_augment_run.py` 为批量数据增强脚本（可选）。
- `train_cnn_paddle.ipynb` 是主训练 Notebook；上面使用 `nbconvert` 执行它以实现非交互式训练。训练时间取决于机器与参数。
- `evaluate_model_paddle.py` 会在 `evaluation_results/` 生成评估图表与报告。

Windows（PowerShell）近似步骤：
- 创建虚拟环境并激活：python -m venv .venv ; .\.venv\Scripts\Activate.ps1
- pip install -r requirements.txt
- 依次运行：python data_preprocessing.py ; python data_augment_run.py ; 使用 Jupyter Notebook GUI 或 nbconvert 执行 notebook ; python evaluate_model_paddle.py

---

## 逐步说明（如果你需要更可控的流程）

1. 克隆仓库
   ```bash
   git clone https://github.com/Zzil-Zhang/traffic-sign-recognition.git
   cd traffic-sign-recognition
   ```

2. 创建并激活虚拟环境，安装依赖
   ```bash
   python -m venv .venv
   source .venv/bin/activate    # Windows: .\.venv\Scripts\activate
   pip install -U pip
   pip install -r requirements.txt
   ```

3. 数据准备（重要）
   - 下载 GTSRB 数据集（German Traffic Sign Recognition Benchmark）。
   - 将数据按仓库约定放置（参见 README_baseline.md）：
     - data/Train/ （训练图片按类别文件夹）
     - data/Test/
     - 以及对应的 Train.csv / Test.csv（若有）
   - 运行预处理：
     ```bash
     python data_preprocessing.py
     ```
     运行后会在 `processed_data/` 下生成：
     - X_train.npy, y_train.npy, X_val.npy, y_val.npy, X_test.npy, y_test.npy
     - train_list.csv / val_list.csv / test_list.csv
     - 若使用 Z-score，还会输出 scaler.pkl

4. （可选）批量数据增强
   ```bash
   python data_augment_run.py
   ```
   - 增强脚本会调用 `data_augmentation.py` 并输出新的 `.npy` 文件到 `processed_data/`（合并或单独保存，取决脚本参数）。

5. 训练模型
  **执行方式**：
   - 方式 A（Notebook，建议交互式观察训练过程）：
     打开 `train_cnn_paddle.ipynb` 在 Jupyter Lab/Notebook 中运行全部单元。
   - 方式 B（无头执行 Notebook）：
     ```bash
     jupyter nbconvert --to notebook --execute train_cnn_paddle.ipynb --ExecutePreprocessor.timeout=3600 --output executed_train.ipynb
     ```
   **超参调优**：
     ```bash
     python hyperparameter_tuning_final_paddle.py
     ```
     该脚本提供交互式菜单用于 K 折交叉验证 / 随机搜索 / 分层搜索，并将输出保存在 `hyperparameter_tuning_result/`。
6. 五折交叉验证 + 超参数调优（交互或脚本/Notebook）
  ```bash
   python hyperparameter_tuning_final_paddle.py
   ```
7. 评估模型
   ```bash
   python evaluate_model_paddle.py
   ```
   - 评估输出保存在 `evaluation_results/`（混淆矩阵、ROC、预测样本示例等）。
   - 若仓库中存在基准模型（`fixed_baseline_model/model.pkl`），脚本会自动进行 SVM+HOG 对比并把结果放在 `baseline_evaluation_results/`。

8. 可视化与结果
   - 训练曲线保存在 `training_curves/`（若训练脚本保存了曲线）。
   - 训练好的参数在 `trained_models/`（常见文件名：`*_final_*.pdparams` 等）。

---

## 重要路径 & 文件说明（快速索引）
- data_preprocessing.py — 数据预处理与保存 `.npy`
- data_augmentation.py / data_augment_run.py — 数据增强
- baseline_model.py — HOG + SVM 基准实现
- data_utils.py — 加载与数据工具函数
- cnn_model_paddle.py — PaddlePaddle 下的 CNN 模型定义
- train_cnn_paddle.ipynb — 训练主 Notebook
- evaluate_model_paddle.py — 评估脚本（生成混淆矩阵、ROC 等）
- evaluate_model_paddle.ipynb — 评估 Notebook（交互式）
  - 功能概述：Notebook 为 `evaluate_model_paddle.py` 的交互式版本，按步骤执行数据加载 → 模型加载 → 评估 → 可视化的流程。便于逐步查看中间结果、调试评估逻辑和即时生成图表。
- hyperparameter_tuning_final_paddle.py — 超参数调优脚本（支持 K 折 CV）
- hyperparameter_tuning_final_paddle.ipynb — 超参调优 Notebook（方便设置 k_folds=5）
- requirements.txt — 建议的依赖列表
- processed_data/ — 预处理后的数据（X_*.npy / y_*.npy）
- trained_models/ — 训练得到的模型参数
- evaluation_results/ — 评估输出
- hyperparameter_tuning_result/ — 超参搜索保存目录

想看每个模块的详细说明，请参考仓库中的：
- README_baseline.md
- README_CNN.md
- README_evaluate.md

---

## 常见问题与解决办法（摘录 / 常见错误）
- 找不到 processed_data/*.npy：请先运行 `python data_preprocessing.py`；确认你的原始数据已放在 `data/` 下且路径变量未修改。
- 找不到 .pdparams 文件：训练完成后文件保存在 `trained_models/`，脚本也会自动在常见名字中查找；如无请先完成训练。
- 设备错误（Paddle custom device）：脚本会尝试设置 `iluvatar_gpu:0`，若失败会回退到 `cpu`。如想使用 GPU，请安装对应的 PaddlePaddle GPU 版本并设置 `paddle.set_device('gpu')`。
- Notebook 无头执行失败：可改为手动在 Jupyter Lab/Notebook 中运行各单元，逐步排查错误并查看即时输出。

---



