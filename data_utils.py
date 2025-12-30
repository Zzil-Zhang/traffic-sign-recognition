import numpy as np
import os
from paddle.io import Dataset, DataLoader
from paddle.vision.transforms import Compose, RandomHorizontalFlip, RandomRotation, ColorJitter, Normalize, Resize
import random
import paddle

def load_data_from_npy():
    
    try:
        X_train = np.load('processed_data/X_train.npy')
        X_val = np.load('processed_data/X_val.npy')
        X_test = np.load('processed_data/X_test.npy')
        y_train = np.load('processed_data/y_train.npy')
        y_val = np.load('processed_data/y_val.npy')
        y_test = np.load('processed_data/y_test.npy')
        
        print(f"数据加载成功")
        print(f"训练集形状: {X_train.shape} - 标签: {len(y_train)}")
        print(f"验证集形状: {X_val.shape} - 标签: {len(y_val)}")
        print(f"测试集形状: {X_test.shape} - 标签: {len(y_test)}")
        
        print(f"训练集范围: [{X_train.min():.3f}, {X_train.max():.3f}]")
        print(f"训练集均值: {X_train.mean():.3f}, 标准差: {X_train.std():.3f}")
        print(f"标签唯一值: {len(np.unique(y_train))}, 范围: [{y_train.min()}, {y_train.max()}]")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    except Exception as e:
        print(f"数据加载失败: {e}")
        return None, None, None

class GTSRBDatasetPaddle(Dataset):
    def __init__(self, images, labels, is_training=False, augment=False):
        self.images = images.astype('float32')
        self.labels = labels.astype('int64')
        self.is_training = is_training
        self.augment = augment
        
        if len(self.images.shape) == 4:
            self.num_samples, self.height, self.width, self.channels = self.images.shape
        else:
            self.num_samples, self.channels, self.height, self.width = self.images.shape
        
        if augment and is_training:
            self.transform = Compose([
                Resize((64, 64)),
                RandomHorizontalFlip(prob=0.2),
                RandomRotation(degrees=5),
                ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], data_format='CHW')
            ])
        else:
            self.transform = Compose([
                Resize((64, 64)),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], data_format='CHW')
            ])
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        img = self.images[idx]
        
        if img.max() > 1.0:
            img = img / 255.0
        
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = img.transpose(2, 0, 1)
        
        img = paddle.to_tensor(img)
        img = self.transform(img)
        label = self.labels[idx]
        
        return img, label
    
    def visualize_sample(self, idx=0):
        try:
            import matplotlib.pyplot as plt
            
            img, label = self.__getitem__(idx)
            img_np = img.numpy()
            if len(img_np.shape) == 3 and img_np.shape[0] == 3:
                img_np = img_np.transpose(1, 2, 0)
            
            mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
            std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
            img_display = (img_np * std) + mean
            img_display = np.clip(img_display, 0, 1)
            
            raw_img = self.images[idx]
            if raw_img.max() > 1.0:
                raw_img_display = raw_img / 255.0
            else:
                raw_img_display = raw_img
            
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
            print("matplotlib未安装")
        except Exception as e:
            print(f"可视化失败: {e}")

def create_data_loaders(batch_size=32, augment_train=True):
    print("正在加载数据")
    
    train_data, val_data, test_data = load_data_from_npy()
    
    if train_data is None:
        print("无法从文件加载数据，创建示例数据用于测试")
        
        num_train = 500
        num_val = 100
        num_test = 100
        
        X_train = np.random.randn(num_train, 64, 64, 3).astype('float32') * 0.1 + 0.5
        X_train = np.clip(X_train, 0, 1) * 255
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
    
    X_train, y_train = train_data
    X_val, y_val = val_data
    X_test, y_test = test_data
    
    print(f"训练集: {len(y_train)} 样本, {len(np.unique(y_train))} 个类别")
    print(f"验证集: {len(y_val)} 样本, {len(np.unique(y_val))} 个类别")
    print(f"测试集: {len(y_test)} 样本, {len(np.unique(y_test))} 个类别")
    
    print("创建数据集")
    train_dataset = GTSRBDatasetPaddle(X_train, y_train, is_training=True, augment=augment_train)
    val_dataset = GTSRBDatasetPaddle(X_val, y_val, is_training=False)
    test_dataset = GTSRBDatasetPaddle(X_test, y_test, is_training=False)
    
    print("创建数据加载器")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        drop_last=True,
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
    
    if len(train_dataset) > 0:
        train_dataset.visualize_sample(0)
    
    return train_loader, val_loader, test_loader, (X_train, y_train, X_val, y_val, X_test, y_test)

def verify_data_consistency():
    print("数据一致性检查")
    
    try:
        X_train = np.load('processed_data/X_train.npy')
        X_val = np.load('processed_data/X_val.npy')
        
        n_check = min(10, len(X_train), len(X_val))
        
        for i in range(n_check):
            train_img = X_train[i].flatten()
            val_img = X_val[i].flatten()
            
            if np.array_equal(train_img, val_img):
                print(f"重复数据，训练集样本{i}和验证集样本{i}相同")
        
        print("数据一致性检查完成")
        
    except Exception as e:
        print(f"数据一致性检查失败: {e}")

if __name__ == "__main__":
    print("测试数据加载器")
    
    train_loader, val_loader, test_loader, data_info = create_data_loaders(batch_size=16)
    
    if train_loader is not None:
        X_train, y_train, X_val, y_val, X_test, y_test = data_info
        
        print(f"训练批次数量: {len(train_loader)}")
        print(f"验证批次数量: {len(val_loader)}")
        print(f"测试批次数量: {len(test_loader)}")
        
        for images, labels in train_loader:
            print(f"图像: {images.shape}, 范围: [{images.min().item():.3f}, {images.max().item():.3f}]")
            print(f"标签: {labels.shape}, 范围: {labels.min().item()}到{labels.max().item()}")
            break