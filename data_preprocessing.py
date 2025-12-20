"""
GTSRB数据预处理流水线
包括：图像resize、归一化、数据集划分、数据增强等功能
"""

import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import pickle

class GTSRBDataLoader:
    """GTSRB数据集加载和预处理类"""
    
    def __init__(self, data_root='data', image_size=(64, 64), normalize='zscore'):
        """
        初始化数据加载器
        
        Args:
            data_root: 数据根目录
            image_size: 统一图像尺寸 (height, width)
            normalize: 归一化方式，'zscore' 或 'minmax' 或 None
        """
        self.data_root = Path(data_root)
        self.image_size = image_size
        self.normalize = normalize
        self.scaler = None
        
    def load_images_from_csv(self, csv_path, use_roi=True):
        """
        从CSV文件加载图像数据
        
        Args:
            csv_path: CSV文件路径
            use_roi: 是否使用ROI区域（从CSV中的Roi坐标）
        
        Returns:
            images: 图像数组 (N, H, W, C)
            labels: 标签数组 (N,)
        """
        df = pd.read_csv(csv_path)
        images = []
        labels = []
        
        print(f"正在加载 {len(df)} 张图像...")
        for idx, row in df.iterrows():
            if idx % 1000 == 0:
                print(f"  已加载: {idx}/{len(df)}")
            
            img_path = self.data_root / row['Path']
            
            if not img_path.exists():
                print(f"警告: 图像文件不存在: {img_path}")
                continue
            
            # 读取图像
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"警告: 无法读取图像: {img_path}")
                continue
            
            # 转换为RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 如果使用ROI，裁剪图像
            if use_roi and 'Roi.X1' in row:
                x1, y1 = int(row['Roi.X1']), int(row['Roi.Y1'])
                x2, y2 = int(row['Roi.X2']), int(row['Roi.Y2'])
                if x2 > x1 and y2 > y1:
                    img = img[y1:y2, x1:x2]
            
            # Resize到统一尺寸
            img = cv2.resize(img, self.image_size)
            
            images.append(img)
            labels.append(int(row['ClassId']))
        
        images = np.array(images, dtype=np.float32)
        labels = np.array(labels, dtype=np.int32)
        
        print(f"✓ 加载完成: {len(images)} 张图像")
        return images, labels
    
    def normalize_images(self, images):
        """
        归一化图像数据
        
        Args:
            images: 图像数组 (N, H, W, C)
        
        Returns:
            normalized_images: 归一化后的图像数组
        """
        if self.normalize is None:
            return images
        
        # 将图像展平为 (N, H*W*C) 进行归一化
        original_shape = images.shape
        images_flat = images.reshape(images.shape[0], -1)
        
        if self.normalize == 'zscore':
            if self.scaler is None:
                self.scaler = StandardScaler()
                images_flat = self.scaler.fit_transform(images_flat)
            else:
                images_flat = self.scaler.transform(images_flat)
        elif self.normalize == 'minmax':
            # Min-Max归一化到[0, 1]
            images_flat = images_flat / 255.0
        
        return images_flat.reshape(original_shape)
    
    def split_dataset(self, images, labels, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
        """
        划分数据集，保持类别分布一致
        
        Args:
            images: 图像数组
            labels: 标签数组
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            random_state: 随机种子
        
        Returns:
            (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例之和必须为1"
        
        # 第一次划分：训练集 vs (验证集+测试集)
        X_train, X_temp, y_train, y_temp = train_test_split(
            images, labels, 
            test_size=(val_ratio + test_ratio),
            random_state=random_state,
            stratify=labels
        )
        
        # 第二次划分：验证集 vs 测试集
        val_size = val_ratio / (val_ratio + test_ratio)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=(1 - val_size),
            random_state=random_state,
            stratify=y_temp
        )
        
        print(f"\n数据集划分完成:")
        print(f"  训练集: {len(X_train)} 样本 ({len(X_train)/len(images)*100:.1f}%)")
        print(f"  验证集: {len(X_val)} 样本 ({len(X_val)/len(images)*100:.1f}%)")
        print(f"  测试集: {len(X_test)} 样本 ({len(X_test)/len(images)*100:.1f}%)")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def save_processed_data(self, X_train, X_val, X_test, y_train, y_val, y_test, save_dir='processed_data'):
        """
        保存预处理后的数据
        
        Args:
            X_train, X_val, X_test: 图像数据
            y_train, y_val, y_test: 标签数据
            save_dir: 保存目录
        """
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        # 保存数据
        np.save(save_path / 'X_train.npy', X_train)
        np.save(save_path / 'X_val.npy', X_val)
        np.save(save_path / 'X_test.npy', X_test)
        np.save(save_path / 'y_train.npy', y_train)
        np.save(save_path / 'y_val.npy', y_val)
        np.save(save_path / 'y_test.npy', y_test)
        
        # 保存scaler（如果使用）
        if self.scaler is not None:
            with open(save_path / 'scaler.pkl', 'wb') as f:
                pickle.dump(self.scaler, f)
        
        # 保存数据列表CSV
        self._save_data_list_csv(X_train, y_train, save_path / 'train_list.csv', 'train')
        self._save_data_list_csv(X_val, y_val, save_path / 'val_list.csv', 'val')
        self._save_data_list_csv(X_test, y_test, save_path / 'test_list.csv', 'test')
        
        print(f"\n✓ 预处理数据已保存到: {save_dir}")
    
    def _save_data_list_csv(self, images, labels, csv_path, split_name):
        """保存数据列表CSV文件"""
        data_list = []
        for i in range(len(images)):
            data_list.append({
                'Index': i,
                'ClassId': labels[i],
                'Split': split_name
            })
        df = pd.DataFrame(data_list)
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    def load_processed_data(self, load_dir='processed_data'):
        """加载预处理后的数据"""
        load_path = Path(load_dir)
        
        X_train = np.load(load_path / 'X_train.npy')
        X_val = np.load(load_path / 'X_val.npy')
        X_test = np.load(load_path / 'X_test.npy')
        y_train = np.load(load_path / 'y_train.npy')
        y_val = np.load(load_path / 'y_val.npy')
        y_test = np.load(load_path / 'y_test.npy')
        
        # 加载scaler
        scaler_path = load_path / 'scaler.pkl'
        if scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
        
        print(f"✓ 从 {load_dir} 加载预处理数据完成")
        return X_train, X_val, X_test, y_train, y_val, y_test


class DataAugmentation:
    """数据增强类 - 几何变换"""
    
    @staticmethod
    def rotate_image(image, angle_range=(-15, 15)):
        """旋转图像"""
        angle = np.random.uniform(angle_range[0], angle_range[1])
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        return rotated
    
    @staticmethod
    def shear_image(image, shear_range=(-0.2, 0.2)):
        """错切变换"""
        shear = np.random.uniform(shear_range[0], shear_range[1])
        h, w = image.shape[:2]
        
        # 创建错切矩阵
        M = np.float32([[1, shear, 0], [0, 1, 0]])
        sheared = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        return sheared
    
    @staticmethod
    def perspective_transform(image, perspective_range=0.1):
        """透视变换 - 模拟车载摄像头视角"""
        h, w = image.shape[:2]
        
        # 定义原始四个角点
        src_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        
        # 随机生成目标四个角点
        offset = perspective_range * min(w, h)
        dst_points = np.float32([
            [np.random.uniform(-offset, offset), np.random.uniform(-offset, offset)],
            [w + np.random.uniform(-offset, offset), np.random.uniform(-offset, offset)],
            [w + np.random.uniform(-offset, offset), h + np.random.uniform(-offset, offset)],
            [np.random.uniform(-offset, offset), h + np.random.uniform(-offset, offset)]
        ])
        
        # 计算透视变换矩阵
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        transformed = cv2.warpPerspective(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        return transformed
    
    @staticmethod
    def apply_augmentation(image, aug_types=['rotate', 'shear', 'perspective'], prob=0.5):
        """
        应用数据增强
        
        Args:
            image: 输入图像
            aug_types: 增强类型列表
            prob: 应用增强的概率
        """
        if np.random.random() > prob:
            return image
        
        aug_type = np.random.choice(aug_types)
        
        if aug_type == 'rotate':
            return DataAugmentation.rotate_image(image)
        elif aug_type == 'shear':
            return DataAugmentation.shear_image(image)
        elif aug_type == 'perspective':
            return DataAugmentation.perspective_transform(image)
        else:
            return image
    
    @staticmethod
    def augment_dataset(X, y, augment_factor=1):
        """
        对数据集进行增强
        
        Args:
            X: 图像数组
            y: 标签数组
            augment_factor: 增强倍数（1表示不增强，2表示数据量翻倍）
        
        Returns:
            X_augmented, y_augmented
        """
        if augment_factor <= 1:
            return X, y
        
        X_augmented = [X]
        y_augmented = [y]
        
        for _ in range(augment_factor - 1):
            X_new = []
            y_new = []
            for i in range(len(X)):
                aug_img = DataAugmentation.apply_augmentation(X[i].copy())
                X_new.append(aug_img)
                y_new.append(y[i])
            X_augmented.append(np.array(X_new))
            y_augmented.append(np.array(y_new))
        
        X_final = np.concatenate(X_augmented, axis=0)
        y_final = np.concatenate(y_augmented, axis=0)
        
        print(f"✓ 数据增强完成: {len(X)} -> {len(X_final)} 样本")
        return X_final, y_final


def preprocess_pipeline():
    """完整的数据预处理流水线"""
    print("=" * 60)
    print("GTSRB数据预处理流水线")
    print("=" * 60)
    
    # 初始化数据加载器
    loader = GTSRBDataLoader(
        data_root='data',
        image_size=(64, 64),  # 可以改为32x32或64x64
        normalize='minmax'  # 'zscore' 或 'minmax'
    )
    
    # 加载训练数据
    print("\n1. 加载训练数据...")
    train_csv_path = Path('data/Train.csv')
    images, labels = loader.load_images_from_csv(train_csv_path, use_roi=True)
    
    # 归一化
    print("\n2. 归一化图像数据...")
    images = loader.normalize_images(images)
    
    # 数据集划分
    print("\n3. 划分数据集...")
    X_train, X_val, X_test, y_train, y_val, y_test = loader.split_dataset(
        images, labels,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42
    )
    
    # 可选：数据增强（仅对训练集）
    print("\n4. 应用数据增强（可选）...")
    use_augmentation = True  # 设置为False跳过增强
    if use_augmentation:
        X_train, y_train = DataAugmentation.augment_dataset(
            X_train, y_train, augment_factor=1  # 1表示不增强，2表示翻倍
        )
    
    # 保存预处理后的数据
    print("\n5. 保存预处理数据...")
    loader.save_processed_data(X_train, X_val, X_test, y_train, y_val, y_test)
    
    print("\n" + "=" * 60)
    print("数据预处理完成！")
    print("=" * 60)
    
    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == '__main__':
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_pipeline()

