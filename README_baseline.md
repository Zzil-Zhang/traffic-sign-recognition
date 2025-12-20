# GTSRB交通标志识别 - 基准模型（成员A）

本项目实现了基于HOG特征提取和SVM分类器的基准模型，用于德国交通标志识别任务。

## 项目结构

```
.
├── data/                          # GTSRB数据集目录
│   ├── Train/                     # 训练数据（按类别分文件夹）
│   ├── Train.csv                  # 训练数据索引文件
│   ├── Test/                      # 测试数据
│   └── Test.csv                   # 测试数据索引文件
├── data_analysis.py               # 数据分析脚本
├── data_preprocessing.py          # 数据预处理流水线
├── baseline_model.py              # 基准模型（HOG + SVM）
├── evaluate_model.py              # 模型评估脚本
├── run_baseline_pipeline.py       # 完整流水线主脚本
└── README_baseline.md             # 本文件
```

## 功能模块

### 1. 数据分析 (`data_analysis.py`)

- 分析数据集结构（样本数、类别数等）
- 统计类别分布，判断数据是否平衡
- 统计图像尺寸分布
- 绘制类别分布条形图
- 生成类别分布统计CSV文件

**运行方式：**
```bash
python data_analysis.py
```

**输出文件：**
- `class_distribution.png` - 类别分布条形图
- `image_size_distribution.png` - 图像尺寸分布图
- `class_distribution_stats.csv` - 类别分布统计

### 2. 数据预处理 (`data_preprocessing.py`)

- 图像加载和ROI裁剪
- 图像Resize到统一尺寸（默认64x64）
- 像素值归一化（Min-Max或Z-Score）
- 数据集划分（70%训练集，15%验证集，15%测试集），保持类别分布一致
- 几何数据增强（旋转、错切、透视变换）
- 保存预处理后的数据和数据列表CSV

**主要类：**
- `GTSRBDataLoader`: 数据加载和预处理类
- `DataAugmentation`: 数据增强类

**运行方式：**
```bash
python data_preprocessing.py
```

**输出文件：**
- `processed_data/` - 预处理后的数据目录
  - `X_train.npy`, `X_val.npy`, `X_test.npy` - 图像数据
  - `y_train.npy`, `y_val.npy`, `y_test.npy` - 标签数据
  - `train_list.csv`, `val_list.csv`, `test_list.csv` - 数据列表
  - `scaler.pkl` - 归一化器（如果使用Z-Score）

### 3. 基准模型 (`baseline_model.py`)

- HOG特征提取（可调整参数：orientations, pixels_per_cell, cells_per_block）
- SVM分类器训练
- 超参数调优（使用随机搜索或网格搜索）
- 模型保存和加载

**主要类：**
- `HOGFeatureExtractor`: HOG特征提取器
- `BaselineSVMClassifier`: 基准SVM分类器

**运行方式：**
```bash
python baseline_model.py
```

**输出文件：**
- `baseline_model/` - 模型目录
  - `svm_model.pkl` - 训练好的SVM模型
  - `hog_extractor.pkl` - HOG特征提取器
  - `training_log.csv` - 训练日志

### 4. 模型评估 (`evaluate_model.py`)

- 计算整体性能指标（准确率、精确率、召回率、F1-score）
- 计算每个类别的详细指标
- 生成分类报告
- 绘制混淆矩阵（原始和归一化）
- 绘制每个类别的性能指标图

**运行方式：**
```bash
python evaluate_model.py
```

**输出文件：**
- `evaluation_results/` - 评估结果目录
  - `classification_report.csv` - 分类报告
  - `per_class_metrics.csv` - 每个类别的指标
  - `evaluation_summary.csv` - 评估摘要
  - `confusion_matrix.png` - 混淆矩阵图
  - `confusion_matrix_normalized.png` - 归一化混淆矩阵图
  - `per_class_metrics.png` - 每个类别的性能指标图

### 5. 完整流水线 (`run_baseline_pipeline.py`)

一键运行所有步骤：数据分析 → 数据预处理 → 模型训练 → 模型评估

**运行方式：**
```bash
python run_baseline_pipeline.py
```

## 快速开始

### 1. 环境要求

确保已安装以下Python包：
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- scikit-image
- opencv-python
- pillow

安装命令：
```bash
pip install numpy pandas matplotlib seaborn scikit-learn scikit-image opencv-python pillow
```

或使用requirements.txt：
```bash
pip install -r requirements.txt
```

### 2. 数据准备

确保GTSRB数据集已解压到 `data/` 目录下，目录结构如下：
```
data/
├── Train/
│   ├── 0/
│   ├── 1/
│   └── ...
├── Train.csv
├── Test/
└── Test.csv
```

### 3. 运行完整流水线

```bash
python run_baseline_pipeline.py
```

或者分步运行：

```bash
# 步骤1: 数据分析
python data_analysis.py

# 步骤2: 数据预处理
python data_preprocessing.py

# 步骤3: 模型训练
python baseline_model.py

# 步骤4: 模型评估
python evaluate_model.py
```

## 参数配置

### 数据预处理参数

在 `data_preprocessing.py` 中可以调整：
- `image_size`: 统一图像尺寸，默认 `(64, 64)`，可改为 `(32, 32)`
- `normalize`: 归一化方式，`'minmax'` 或 `'zscore'` 或 `None`

### HOG特征参数

在 `baseline_model.py` 中可以调整：
- `orientations`: 方向梯度直方图的bin数量，默认 `9`
- `pixels_per_cell`: 每个cell的像素数，默认 `(8, 8)`
- `cells_per_block`: 每个block的cell数，默认 `(2, 2)`

### SVM超参数

在 `baseline_model.py` 中可以调整超参数搜索空间：
```python
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
    'kernel': ['rbf', 'linear', 'poly']
}
```

## 使用预处理数据（供成员B和C调用）

其他成员可以使用 `data_preprocessing.py` 中的 `GTSRBDataLoader` 类来加载预处理后的数据：

```python
from data_preprocessing import GTSRBDataLoader

# 初始化加载器
loader = GTSRBDataLoader(
    data_root='data',
    image_size=(64, 64),
    normalize='minmax'
)

# 加载预处理后的数据
X_train, X_val, X_test, y_train, y_val, y_test = loader.load_processed_data('processed_data')

# 使用数据...
```

## 交付产出

1. **清洗后的标准化数据集**
   - `processed_data/` 目录下的所有文件
   - `processed_data/train_list.csv`, `val_list.csv`, `test_list.csv`

2. **基准模型代码**
   - `baseline_model.py` - 完整的基准模型实现
   - `baseline_model/` - 训练好的模型文件

3. **训练日志**
   - `baseline_model/training_log.csv` - 包含超参数搜索结果和验证集性能

4. **评估报告**
   - `evaluation_results/classification_report.csv` - 分类报告
   - `evaluation_results/per_class_metrics.csv` - 每个类别的指标
   - `evaluation_results/confusion_matrix.png` - 混淆矩阵图
   - `evaluation_results/per_class_metrics.png` - 每个类别的性能指标图

## 注意事项

1. **数据不平衡**：GTSRB数据集存在类别不平衡问题，评估时会计算每个类别的指标
2. **内存占用**：如果内存不足，可以减小 `image_size` 或使用数据生成器
3. **训练时间**：SVM超参数搜索可能需要较长时间，可以通过减少 `n_iter` 或使用更小的参数网格来加速
4. **数据增强**：默认情况下数据增强是关闭的（`augment_factor=1`），如需启用可以修改 `data_preprocessing.py` 中的参数

## 联系方式

如有问题，请联系成员A。


