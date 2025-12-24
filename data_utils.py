# data_utils.py - 增强版数据加载工具
import numpy as np
import os
import cv2
from paddle.io import Dataset, DataLoader
from paddle.vision.transforms import Compose, RandomHorizontalFlip, RandomRotation, ColorJitter, Normalize, Resize
import random
import paddle  # 添加这行！！！

def load_data_from_npy():
    """从npy文件加载数据"""
    print("📂 从processed_data加载预处理数据...")
    
    try:
        X_train = np.load('processed_data/X_train.npy')
        X_val = np.load('processed_data/X_val.npy')
        X_test = np.load('processed_data/X_test.npy')
        y_train = np.load('processed_data/y_train.npy')
        y_val = np.load('processed_data/y_val.npy')
        y_test = np.load('processed_data/y_test.npy')
        
        # 检查数据形状
        print(f"✅ 数据加载成功！")
        print(f"  训练集形状: {X_train.shape} - 标签: {len(y_train)}")
        print(f"  验证集形状: {X_val.shape} - 标签: {len(y_val)}")
        print(f"  测试集形状: {X_test.shape} - 标签: {len(y_test)}")
        
        # 检查数据范围
        print(f"\n🔍 数据统计:")
        print(f"  训练集范围: [{X_train.min():.3f}, {X_train.max():.3f}]")
        print(f"  训练集均值: {X_train.mean():.3f}, 标准差: {X_train.std():.3f}")
        print(f"  标签唯一值: {len(np.unique(y_train))}, 范围: [{y_train.min()}, {y_train.max()}]")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        print("请确保已运行data_preprocessing.py预处理数据")
        return None, None, None

class GTSRBDatasetPaddle(Dataset):
    """改进的PaddlePaddle数据集类，支持数据增强"""
    def __init__(self, images, labels, is_training=False, augment=False):
        """
        参数:
            images: 图像数据
            labels: 标签数据
            is_training: 是否为训练集
            augment: 是否进行数据增强
        """
        # 确保图像数据是float32
        self.images = images.astype('float32')
        self.labels = labels.astype('int64')
        self.is_training = is_training
        self.augment = augment
        
        # 获取图像尺寸信息
        if len(self.images.shape) == 4:
            self.num_samples, self.height, self.width, self.channels = self.images.shape
        else:
            # 如果已经是CHW格式
            self.num_samples, self.channels, self.height, self.width = self.images.shape
        
        # 创建数据增强变换 - 降低增强强度，防止过拟合
        if augment and is_training:
            self.transform = Compose([
                Resize((64, 64)),  # 确保尺寸一致
                RandomHorizontalFlip(prob=0.2),  # 降低翻转概率
                RandomRotation(degrees=5),  # 降低旋转角度
                ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),  # 降低扰动强度
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], data_format='CHW')
            ])
        else:
            self.transform = Compose([
                Resize((64, 64)),  # 确保尺寸一致
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], data_format='CHW')
            ])
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # 获取图像数据
        img = self.images[idx]
        
        # 如果图像在[0, 255]范围内，归一化到[0, 1]
        if img.max() > 1.0:
            img = img / 255.0
        
        # 确保图像是CHW格式 (PaddlePaddle期望的格式)
        if len(img.shape) == 3 and img.shape[2] == 3:  # HWC格式
            img = img.transpose(2, 0, 1)  # 转换为CHW
        
        # 转换为paddle tensor
        img = paddle.to_tensor(img)
        
        # 应用变换
        img = self.transform(img)
        
        # 获取标签
        label = self.labels[idx]
        
        return img, label
    
    def visualize_sample(self, idx=0):
        """可视化样本"""
        try:
            import matplotlib.pyplot as plt
            
            img, label = self.__getitem__(idx)
            
            # 转换回HWC格式显示
            img_np = img.numpy()
            if len(img_np.shape) == 3 and img_np.shape[0] == 3:  # CHW格式
                img_np = img_np.transpose(1, 2, 0)  # 转换为HWC
            
            # 反标准化用于显示
            mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
            std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
            img_display = (img_np * std) + mean
            img_display = np.clip(img_display, 0, 1)
            
            # 获取原始图像（未处理）
            raw_img = self.images[idx]
            if raw_img.max() > 1.0:
                raw_img_display = raw_img / 255.0
            else:
                raw_img_display = raw_img
            
            # 如果原始图像是CHW格式，转换为HWC
            if len(raw_img_display.shape) == 3 and raw_img_display.shape[0] == 3:
                raw_img_display = raw_img_display.transpose(1, 2, 0)
            
            plt.figure(figsize=(8, 4))
            plt.subplot(1, 2, 1)
            plt.imshow(raw_img_display)
            plt.title(f"原始样本 {idx} - 标签: {self.labels[idx]}")
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.imshow(img_display)
            plt.title(f"处理后样本 {idx} - 标签: {label}")
            plt.axis('off')
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("matplotlib未安装，无法显示图像")
        except Exception as e:
            print(f"可视化失败: {e}")

def create_data_loaders(batch_size=32, augment_train=True):
    """创建数据加载器"""
    print("🔄 正在加载数据...")
    
    # 尝试从npy文件加载数据
    train_data, val_data, test_data = load_data_from_npy()
    
    # 如果加载失败，创建示例数据用于测试
    if train_data is None:
        print("⚠️  无法从文件加载数据，创建示例数据用于测试...")
        
        # 创建示例数据
        num_train = 500
        num_val = 100
        num_test = 100
        
        # 随机生成图像数据
        X_train = np.random.randn(num_train, 64, 64, 3).astype('float32') * 0.1 + 0.5
        X_train = np.clip(X_train, 0, 1) * 255  # 模拟[0,255]范围
        y_train = np.random.randint(0, 43, num_train).astype('int64')
        
        X_val = np.random.randn(num_val, 64, 64, 3).astype('float32') * 0.1 + 0.5
        X_val = np.clip(X_val, 0, 1) * 255
        y_val = np.random.randint(0, 43, num_val).astype('int64')
        
        X_test = np.random.randn(num_test, 64, 64, 3).astype('float32') * 0.1 + 0.5
        X_test = np.clip(X_test, 0, 1) * 255
        y_test = np.random.randint(0, 43, num_test).astype('int64')
        
        train_data = (X_train, y_train)
        val_data = (X_val, y_val)
        test_data = (X_test, y_test)
    
    # 解包数据
    X_train, y_train = train_data
    X_val, y_val = val_data
    X_test, y_test = test_data
    
    # 检查数据分布
    print(f"\n📊 数据分布统计:")
    for name, X, y in [("训练集", X_train, y_train), 
                       ("验证集", X_val, y_val), 
                       ("测试集", X_test, y_test)]:
        unique, counts = np.unique(y, return_counts=True)
        print(f"  {name}: {len(y)} 样本, {len(unique)} 个类别")
    
    # 创建数据集
    print("\n🛠️  创建数据集...")
    train_dataset = GTSRBDatasetPaddle(X_train, y_train, is_training=True, augment=augment_train)
    val_dataset = GTSRBDatasetPaddle(X_val, y_val, is_training=False)
    test_dataset = GTSRBDatasetPaddle(X_test, y_test, is_training=False)
    
    # 创建数据加载器
    print("🔄 创建数据加载器...")
    
    # 训练集使用更随机的shuffle
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        drop_last=True,  # 丢弃最后不完整的批次，稳定训练
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # 显示一个样本用于验证
    if len(train_dataset) > 0:
        print("\n👁️  显示一个训练样本用于验证:")
        train_dataset.visualize_sample(0)
    
    return train_loader, val_loader, test_loader, (X_train, y_train, X_val, y_val, X_test, y_test)

def verify_data_consistency():
    """验证数据一致性"""
    print("\n🔍 数据一致性检查...")
    
    try:
        # 检查是否有重复数据
        from sklearn.metrics.pairwise import cosine_similarity
        
        X_train = np.load('processed_data/X_train.npy')
        X_val = np.load('processed_data/X_val.npy')
        
        # 随机检查几个样本
        n_check = min(10, len(X_train), len(X_val))
        
        for i in range(n_check):
            train_img = X_train[i].flatten()
            val_img = X_val[i].flatten()
            
            if np.array_equal(train_img, val_img):
                print(f"⚠️  发现重复数据: 训练集样本{i}和验证集样本{i}相同！")
        
        print("✅ 数据一致性检查完成")
        
    except Exception as e:
        print(f"❌ 数据一致性检查失败: {e}")

# 在文件末尾添加测试代码
if __name__ == "__main__":
    print("测试数据加载器...")
    
    # 测试数据加载
    train_loader, val_loader, test_loader, data_info = create_data_loaders(batch_size=16)
    
    if train_loader is not None:
        X_train, y_train, X_val, y_val, X_test, y_test = data_info
        
        print(f"\n✅ 数据加载成功:")
        print(f"  训练批次数量: {len(train_loader)}")
        print(f"  验证批次数量: {len(val_loader)}")
        print(f"  测试批次数量: {len(test_loader)}")
        
        # 验证一个批次
        for images, labels in train_loader:
            print(f"\n一个批次的形状:")
            print(f"  图像: {images.shape}, 范围: [{images.min().item():.3f}, {images.max().item():.3f}]")
            print(f"  标签: {labels.shape}, 范围: {labels.min().item()}到{labels.max().item()}")
            break