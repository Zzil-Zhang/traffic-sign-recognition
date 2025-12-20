# 成员A交接说明文档

## 📋 项目状态说明

### 原有代码 vs 新代码

**原有代码（GitHub项目）：**
- `traffic_sign.py` - 简单的CNN训练脚本（**可以保留作为参考，但不建议直接使用**）
- `gui.py` - GUI界面（**可以保留，不影响新代码**）
- 其他辅助文件（`image_cutting.py`, `BDimgSpyder.py` 等）- **可以保留，不影响新代码**

**新代码（成员A的任务）：**
- `data_analysis.py` - 数据分析脚本 ✅
- `data_preprocessing.py` - 数据预处理流水线 ✅（**核心文件，成员B和C需要使用**）
- `baseline_model.py` - 基准模型（HOG+SVM）✅
- `evaluate_model.py` - 模型评估脚本 ✅
- `run_baseline_pipeline.py` - 完整流水线主脚本 ✅

### ⚠️ 重要说明

1. **新旧代码可以共存**：新代码不会覆盖或影响原有代码
2. **建议保留原有代码**：作为参考，但成员B和C应该使用新代码提供的数据加载器
3. **数据格式差异**：
   - 原有代码：简单的80/20划分，没有验证集，图像尺寸30x30
   - 新代码：70/15/15划分（训练/验证/测试），使用stratify保持类别分布，图像尺寸64x64（可配置）

---

## 🤝 与成员B的交接

### 成员B需要知道的信息

1. **数据加载器位置**：`data_preprocessing.py` 中的 `GTSRBDataLoader` 类
2. **预处理数据位置**：`processed_data/` 目录（运行 `data_preprocessing.py` 后生成）
3. **数据格式**：
   - 图像：numpy数组，形状 `(N, H, W, C)`，像素值已归一化到[0,1]
   - 标签：numpy数组，形状 `(N,)`，类别ID（0-42），**不是one-hot编码**
   - 数据集划分：训练集70%，验证集15%，测试集15%

### 成员B如何使用数据加载器

**方法1：直接加载预处理好的数据（推荐）**

```python
from data_preprocessing import GTSRBDataLoader

# 初始化加载器
loader = GTSRBDataLoader(
    data_root='data',
    image_size=(64, 64),  # 或 (32, 32)
    normalize='minmax'     # 像素值归一化到[0,1]
)

# 加载预处理后的数据
X_train, X_val, X_test, y_train, y_val, y_test = loader.load_processed_data('processed_data')

# 注意：标签是类别ID（0-42），不是one-hot编码
# 如果使用Keras/TensorFlow，需要转换为one-hot：
from keras.utils import to_categorical
y_train_onehot = to_categorical(y_train, 43)
y_val_onehot = to_categorical(y_val, 43)
y_test_onehot = to_categorical(y_test, 43)
```

**方法2：重新处理数据（如果需要不同的图像尺寸或归一化方式）**

```python
from data_preprocessing import GTSRBDataLoader

loader = GTSRBDataLoader(
    data_root='data',
    image_size=(32, 32),  # 自定义尺寸
    normalize='zscore'     # 或 'minmax' 或 None
)

# 从CSV加载并处理
train_csv_path = 'data/Train.csv'
images, labels = loader.load_images_from_csv(train_csv_path, use_roi=True)

# 归一化
images = loader.normalize_images(images)

# 划分数据集
X_train, X_val, X_test, y_train, y_val, y_test = loader.split_dataset(
    images, labels,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    random_state=42
)
```

### 成员B需要修改的地方

1. **不要使用 `traffic_sign.py` 中的数据加载方式**
2. **使用 `GTSRBDataLoader` 加载数据**
3. **注意标签格式**：新代码返回的是类别ID，不是one-hot编码
4. **使用验证集**：新代码提供了独立的验证集，用于模型选择和超参数调优

### 示例：成员B的CNN训练脚本框架

```python
from data_preprocessing import GTSRBDataLoader
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization, ReLU, MaxPool2D, Dropout, Dense, Flatten

# 1. 加载数据
loader = GTSRBDataLoader(data_root='data', image_size=(64, 64), normalize='minmax')
X_train, X_val, X_test, y_train, y_val, y_test = loader.load_processed_data('processed_data')

# 2. 转换为one-hot编码（如果需要）
y_train_onehot = to_categorical(y_train, 43)
y_val_onehot = to_categorical(y_val, 43)

# 3. 构建模型
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3)))
model.add(BatchNormalization())
model.add(ReLU())
model.add(MaxPool2D((2, 2)))
model.add(Dropout(0.25))
# ... 更多层

# 4. 训练
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(
    X_train, y_train_onehot,
    validation_data=(X_val, y_val_onehot),
    epochs=50,
    batch_size=32
)
```

---

## 🎨 与成员C的交接

### 成员C需要知道的信息

1. **数据增强类位置**：`data_preprocessing.py` 中的 `DataAugmentation` 类
2. **当前数据增强功能**：
   - 旋转（rotate）
   - 错切（shear）
   - 透视变换（perspective）
3. **成员C可以扩展的功能**：
   - 运动模糊
   - 雨滴噪声
   - 亮度/对比度调整
   - 使用Albumentations库

### 成员C如何使用数据增强

**方法1：使用现有的数据增强类**

```python
from data_preprocessing import DataAugmentation, GTSRBDataLoader
import numpy as np

# 加载数据
loader = GTSRBDataLoader(data_root='data', image_size=(64, 64), normalize='minmax')
X_train, X_val, X_test, y_train, y_val, y_test = loader.load_processed_data('processed_data')

# 应用数据增强（仅对训练集）
X_train_augmented = []
y_train_augmented = []

for i in range(len(X_train)):
    # 原始图像
    X_train_augmented.append(X_train[i])
    y_train_augmented.append(y_train[i])
    
    # 增强后的图像（50%概率应用增强）
    aug_img = DataAugmentation.apply_augmentation(
        X_train[i].copy(),
        aug_types=['rotate', 'shear', 'perspective'],
        prob=0.5
    )
    X_train_augmented.append(aug_img)
    y_train_augmented.append(y_train[i])

X_train_augmented = np.array(X_train_augmented)
y_train_augmented = np.array(y_train_augmented)
```

**方法2：扩展数据增强类（成员C可以添加新功能）**

```python
from data_preprocessing import DataAugmentation
import cv2

class AdvancedDataAugmentation(DataAugmentation):
    """扩展的数据增强类"""
    
    @staticmethod
    def motion_blur(image, kernel_size=15):
        """运动模糊"""
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
        kernel = kernel / kernel_size
        blurred = cv2.filter2D(image, -1, kernel)
        return blurred
    
    @staticmethod
    def adjust_brightness_contrast(image, brightness=0, contrast=0):
        """调整亮度和对比度"""
        alpha = 1 + contrast / 100.0
        beta = brightness
        adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        return adjusted
    
    @staticmethod
    def apply_augmentation(image, aug_types=['rotate', 'shear', 'perspective', 'motion_blur'], prob=0.5):
        """应用增强（包含新功能）"""
        if np.random.random() > prob:
            return image
        
        aug_type = np.random.choice(aug_types)
        
        if aug_type == 'motion_blur':
            return AdvancedDataAugmentation.motion_blur(image)
        elif aug_type == 'brightness_contrast':
            return AdvancedDataAugmentation.adjust_brightness_contrast(image)
        else:
            # 调用父类的方法
            return super().apply_augmentation(image, [aug_type], prob=1.0)
```

**方法3：使用Albumentations（推荐给成员C）**

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

# 定义增强管道
transform = A.Compose([
    A.Rotate(limit=15, p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.MotionBlur(blur_limit=3, p=0.3),
    A.RandomRain(p=0.2),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 应用增强
augmented = transform(image=image)['image']
```

### 成员C需要修改的地方

1. **可以扩展 `DataAugmentation` 类**，添加新的增强方法
2. **建议使用Albumentations库**，功能更强大
3. **数据增强应该只应用于训练集**，不要对验证集和测试集应用
4. **与成员B协作**：确定哪些增强策略效果最好

---

## 📊 基准模型对比

### 基准模型性能

成员B和C可以参考基准模型（HOG+SVM）的性能作为对比：

- **基准模型位置**：`baseline_model/` 目录
- **评估结果**：`evaluation_results/` 目录
- **预期性能**：基准模型准确率通常在60-75%左右（取决于HOG参数和SVM超参数）

**深度学习模型应该超过基准模型**，目标：
- 训练集准确率 > 95%
- 验证集准确率 > 90%
- 测试集准确率 > 85%

---

## 📁 文件组织结构建议

```
traffic-sign-recognition/
├── data/                          # 数据集（所有成员共享）
├── processed_data/                # 预处理数据（成员A生成，成员B和C使用）
├── baseline_model/                # 基准模型（成员A）
├── evaluation_results/            # 基准模型评估结果（成员A）
│
├── data_analysis.py               # 数据分析（成员A）
├── data_preprocessing.py          # 数据预处理（成员A，成员B和C使用）
├── baseline_model.py              # 基准模型（成员A）
├── evaluate_model.py              # 评估脚本（成员A）
│
├── cnn_model.py                   # CNN模型定义（成员B）
├── train_cnn.py                   # CNN训练脚本（成员B）
├── trained_models/                # 训练好的模型（成员B）
│
├── data_augmentation.py           # 高级数据增强（成员C）
├── hyperparameter_tuning.py       # 超参数调优（成员C）
├── visualization.py               # 可视化脚本（成员C）
├── final_evaluation.py            # 最终评估（成员C）
│
└── traffic_sign.py                # 原有代码（保留作为参考）
```

---

## ✅ 交接检查清单

### 成员A需要完成：

- [x] 数据分析脚本（`data_analysis.py`）
- [x] 数据预处理流水线（`data_preprocessing.py`）
- [x] 基准模型实现（`baseline_model.py`）
- [x] 模型评估脚本（`evaluate_model.py`）
- [x] 运行完整流水线，生成预处理数据
- [x] 训练基准模型，生成评估报告
- [x] 创建交接文档

### 成员B需要确认：

- [ ] 能够成功加载预处理数据
- [ ] 理解数据格式（图像形状、标签格式）
- [ ] 能够使用数据加载器构建CNN模型
- [ ] 理解验证集的作用

### 成员C需要确认：

- [ ] 理解数据增强的接口
- [ ] 能够扩展数据增强功能
- [ ] 理解如何与成员B协作进行超参数调优

---

## 🚀 快速开始（给成员B和C）

### 第一步：运行成员A的代码生成预处理数据

```bash
# 如果还没有运行过，先运行数据预处理
python data_preprocessing.py
```

### 第二步：成员B开始构建CNN模型

参考上面的示例代码，使用 `GTSRBDataLoader` 加载数据。

### 第三步：成员C开始实现数据增强

参考上面的示例代码，扩展 `DataAugmentation` 类或使用Albumentations。

---

## 📞 联系方式

如有问题，请联系成员A。

**重要提醒：**
- 所有成员都应该使用 `processed_data/` 目录下的预处理数据，确保数据一致性
- 测试集（`X_test`, `y_test`）应该**只在最终评估时使用**，不要用于模型选择或超参数调优
- 验证集（`X_val`, `y_val`）用于模型选择和超参数调优

