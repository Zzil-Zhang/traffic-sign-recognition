# data_augment_run.py
import numpy as np
from data_augmentation import AdvancedDataAugmentation
import time

# 批量增强数据

# 1. 加载处理好的数据
try:
    X_train = np.load('processed_data/X_train.npy')
    y_train = np.load('processed_data/y_train.npy')
    print(f"✓ 加载成功！训练集: {len(X_train)} 张图片")
except:
    print("找不到数据！请先让成员A运行 data_preprocessing.py")
    exit()

# 2. 检查数据格式
print(f"数据格式: {X_train.dtype}, 数值范围: [{X_train.min():.2f}, {X_train.max():.2f}]")

# 3. 创建增强器
aug = AdvancedDataAugmentation()

# 4. 对训练集的每张图片应用增强
print("\n开始增强...")

# 存储增强后的所有数据
X_augmented = []  # 存储所有图像（原始 + 增强）
y_augmented = []  # 存储所有标签

# 首先保留所有原始图片
X_augmented.extend(X_train)
y_augmented.extend(y_train)

print(f"已添加原始数据: {len(X_train)} 张")

# 然后为每张原始图片生成增强版本（这会增加数据量）
for i, (img, label) in enumerate(zip(X_train, y_train)):
    if i % 500 == 0:
        print(f"  正在增强: {i}/{len(X_train)}")

    # 转换格式（如果数据是0~1的浮点数）
    if img.dtype != np.uint8 and img.max() <= 1.0:
        img_for_aug = (img * 255).astype(np.uint8)
    else:
        img_for_aug = img.astype(np.uint8)

    # 应用增强 - 生成新的增强图像
    aug_img = aug.apply_combined_augmentation(img_for_aug)

    # 转换回原来的格式
    if X_train.dtype != np.uint8 and X_train.max() <= 1.0:
        aug_img = aug_img.astype(np.float32) / 255.0

    # 将增强后的图片添加到数据集中（这会使数据量增加）
    X_augmented.append(aug_img)
    y_augmented.append(label)

# 转换为numpy数组
X_augmented = np.array(X_augmented)
y_augmented = np.array(y_augmented)

# 5. 保存
np.save('processed_data/X_train_augmented.npy', X_augmented)
np.save('processed_data/y_train_augmented.npy', y_augmented)

print(f"\n任务完成！")
print(f"原始数据: {len(X_train)} 张")
print(f"增强后: {len(X_augmented)} 张")
print(f"文件已保存: processed_data/X_train_augmented.npy")
print("请把这个文件交给成员B！")