# GTSRB交通标志识别 - 基准模型

本项目实现了基于HOG特征提取和SVM分类器的基准模型，用于德国交通标志识别任务。

## 项目主要结构

```
.
├── data/                          # GTSRB数据集目录
│   ├── Train/                     # 训练数据（按类别分文件夹）
│   ├── Train.csv                  # 训练数据索引文件
│   ├── Test/                      # 测试数据
│   └── Test.csv                   # 测试数据索引文件
├── data_analysis.py               # 数据分析脚本
├── data_preprocessing.py          # 数据预处理流水线
├── baseline_model.py              # 基准模型
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

- HOG特征提取
- SVM分类器训练
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
  - `model.pkl` - 训练好的模型
  - `training_log.csv` - 训练日志

### 4. 模型评估 (包含在基准模型文件中)

- 计算整体性能指标（准确率、精确率、召回率、F1-score）
- 计算每个类别的详细指标
- 生成分类报告
- 绘制混淆矩阵（原始和归一化）
- 绘制每个类别的性能指标图

**输出文件：**
- `baseline_baseline_evaluation_results/` - 评估结果目录
  - `per_class_metrics.csv` - 每个类别的指标
  - `confusion_matrix.png` - 混淆矩阵图
  - `confusion_matrix_normalized.png` - 归一化混淆矩阵图
  - `per_class_metrics.png` - 每个类别的性能指标图

### 5. 完整流水线 (`baseline_model.py`)

一键运行所有步骤：数据分析 → 数据预处理 → 模型训练 → 模型评估

**运行方式：**
```bash
python baseline_model.py
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

分步运行：

```bash
# 步骤1: 数据分析
python data_analysis.py

# 步骤2: 数据预处理
python data_preprocessing.py

# 步骤3: 模型训练
python baseline_model.py
```

## 参数配置

### 数据预处理参数

在 `data_preprocessing.py` 中可以调整：
- `image_size`: 统一图像尺寸，默认 `(64, 64)`，可改为 `(32, 32)`
- `normalize`: 归一化方式，`'minmax'` 或 `'zscore'` 或 `None`

### HOG特征参数

在 `baseline_model.py` 中可以调整：
- `orientations`: 方向梯度直方图的bin数量，默认 `9`
- `pixels_per_cell`: 每个cell的像素数，默认 `(16, 16)`
- `cells_per_block`: 每个block的cell数，默认 `(2, 2)`

```

## 使用预处理数据（供后续成员调用）

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
   - `baseline_evaluation_results/classification_report.csv` - 分类报告
   - `baseline_evaluation_results/per_class_metrics.csv` - 每个类别的指标
   - `baseline_evaluation_results/confusion_matrix.png` - 混淆矩阵图
   - `baseline_evaluation_results/per_class_metrics.png` - 每个类别的性能指标图



