"""
修复版基准模型：HOG特征提取 + SVM分类器
针对GTSRB交通标志识别优化
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from skimage.feature import hog
from skimage import color, exposure
import pickle
import time
import matplotlib.pyplot as plt
import seaborn as sns
import os
from data_preprocessing import GTSRBDataLoader

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class EnhancedHOGFeatureExtractor:
    """增强版HOG特征提取器"""

    def __init__(self, orientations=9, pixels_per_cell=(16, 16),
                 cells_per_block=(2, 2), use_color_hog=False):
        """
        初始化增强版HOG特征提取器

        Args:
            orientations: 方向梯度直方图的bin数量
            pixels_per_cell: 每个cell的像素数，建议(16,16)对于64x64图像
            cells_per_block: 每个block的cell数
            use_color_hog: 是否使用颜色HOG（更慢但可能更准确）
        """
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.use_color_hog = use_color_hog

    def extract_features(self, images):
        """提取HOG特征"""
        features = []

        print(f"正在提取增强版HOG特征...")
        print(f"参数: orientations={self.orientations}, "
              f"pixels_per_cell={self.pixels_per_cell}, "
              f"cells_per_block={self.cells_per_block}")

        for i, img in enumerate(images):
            if i % 1000 == 0:
                print(f"  已处理: {i}/{len(images)}")

            # 确保图像是float32
            if img.dtype != np.float32:
                img = img.astype(np.float32)

            if self.use_color_hog and len(img.shape) == 3:
                # 颜色HOG：分别提取RGB通道的HOG特征
                channel_features = []
                for channel in range(3):
                    channel_img = img[:, :, channel]
                    # 添加直方图均衡化增强对比度
                    channel_img = exposure.equalize_hist(channel_img)

                    hog_feature = hog(
                        channel_img,
                        orientations=self.orientations,
                        pixels_per_cell=self.pixels_per_cell,
                        cells_per_block=self.cells_per_block,
                        block_norm='L2-Hys',
                        visualize=False,
                        feature_vector=True,
                        transform_sqrt=True
                    )
                    channel_features.append(hog_feature)

                # 合并所有通道特征
                final_feature = np.concatenate(channel_features)
            else:
                # 标准HOG：转换为灰度图
                if len(img.shape) == 3:
                    # 使用OpenCV的灰度转换公式
                    gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
                else:
                    gray = img

                # 直方图均衡化增强对比度
                gray = exposure.equalize_hist(gray)

                final_feature = hog(
                    gray,
                    orientations=self.orientations,
                    pixels_per_cell=self.pixels_per_cell,
                    cells_per_block=self.cells_per_block,
                    block_norm='L2-Hys',
                    visualize=False,
                    feature_vector=True,
                    transform_sqrt=True
                )

            features.append(final_feature)

        features = np.array(features, dtype=np.float32)
        print(f"✓ HOG特征提取完成: {features.shape}")

        # 计算并显示特征统计
        print(f"  特征维度: {features.shape[1]}")
        print(f"  特征均值: {features.mean():.4f}, 标准差: {features.std():.4f}")

        return features

    def get_feature_dimension(self, image_shape):
        """计算HOG特征的维度"""
        if len(image_shape) == 3:
            h, w = image_shape[:2]
        else:
            h, w = image_shape

        n_cells_h = h // self.pixels_per_cell[0]
        n_cells_w = w // self.pixels_per_cell[1]

        n_blocks_h = n_cells_h - self.cells_per_block[0] + 1
        n_blocks_w = n_cells_w - self.cells_per_block[1] + 1

        feature_dim = (n_blocks_h * n_blocks_w *
                       self.cells_per_block[0] * self.cells_per_block[1] *
                       self.orientations)

        if self.use_color_hog:
            feature_dim *= 3  # RGB三个通道

        return feature_dim


class OptimizedSVMClassifier:
    """优化版SVM分类器"""

    def __init__(self, hog_extractor=None):
        self.hog_extractor = hog_extractor
        self.svm_model = None
        self.scaler = StandardScaler()
        self.training_log = []

    def train_with_validation(self, X_train, y_train, X_val, y_val,
                              use_sample=False, sample_size=2000):
        """带验证的训练"""
        print("=" * 60)
        print("优化版SVM训练")
        print("=" * 60)

        # 可选：使用部分样本加速调试
        if use_sample and len(X_train) > sample_size:
            print(f"使用 {sample_size} 个样本进行训练...")
            indices = np.random.choice(len(X_train), sample_size, replace=False)
            X_train = X_train[indices]
            y_train = y_train[indices]

        print(f"训练数据: {X_train.shape[0]} 样本")
        print(f"验证数据: {X_val.shape[0]} 样本")
        print(f"类别数: {len(np.unique(y_train))}")

        # 1. 提取HOG特征
        print("\n1. 提取HOG特征...")
        start_time = time.time()

        X_train_hog = self.hog_extractor.extract_features(X_train)
        X_val_hog = self.hog_extractor.extract_features(X_val)

        hog_time = time.time() - start_time
        print(f"  ✓ HOG特征提取完成 (耗时: {hog_time:.2f}秒)")

        # 2. 特征标准化
        print("\n2. 特征标准化...")
        X_train_scaled = self.scaler.fit_transform(X_train_hog)
        X_val_scaled = self.scaler.transform(X_val_hog)

        # 3. 分析类别分布
        print("\n3. 分析类别分布...")
        unique, counts = np.unique(y_train, return_counts=True)
        print(f"  总类别数: {len(unique)}")
        print(f"  最小类别样本数: {counts.min()}")
        print(f"  最大类别样本数: {counts.max()}")

        # 4. 训练SVM（使用优化参数）
        print("\n4. 训练优化版LinearSVC...")
        print("  参数配置:")
        print("  - dual=True (对于高维特征)")
        print("  - C=0.1 (中等正则化强度)")
        print("  - tol=1e-3 (收敛条件)")
        print("  - max_iter=5000")
        print("  - class_weight='balanced' (处理类别不平衡)")

        train_start = time.time()

        # 根据特征维度选择dual参数
        n_features = X_train_scaled.shape[1]
        n_samples = X_train_scaled.shape[0]

        # 规则：当 n_features > n_samples 时使用dual=True
        use_dual = n_features > n_samples

        svm = LinearSVC(
            dual=use_dual,
            C=0.1,
            max_iter=5000,
            tol=1e-3,
            random_state=42,
            class_weight='balanced',
            verbose=1
        )

        svm.fit(X_train_scaled, y_train)

        train_time = time.time() - train_start
        print(f"\n  ✓ SVM训练完成 (耗时: {train_time:.2f}秒)")
        print(f"    迭代次数: {svm.n_iter_}")

        # 5. 验证集评估
        print("\n5. 验证集评估...")
        y_val_pred = svm.predict(X_val_scaled)
        val_accuracy = accuracy_score(y_val, y_val_pred)

        print(f"  验证集准确率: {val_accuracy:.4f}")
        print(f"  随机基线准确率: {1 / len(np.unique(y_val)):.4f}")

        # 显示每个类别的准确率
        print("\n  各类别准确率（前10类）:")
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_val, y_val_pred)
        class_accuracies = cm.diagonal() / cm.sum(axis=1)

        for i in range(min(10, len(class_accuracies))):
            print(f"    类别 {i}: {class_accuracies[i]:.4f}")

        # 保存模型
        self.svm_model = svm

        # 记录日志
        self.training_log.append({
            'n_samples': X_train.shape[0],
            'n_features': n_features,
            'n_classes': len(unique),
            'val_accuracy': val_accuracy,
            'train_time': train_time,
            'hog_time': hog_time,
            'dual_used': use_dual,
            'iterations': svm.n_iter_
        })

        return svm, val_accuracy

    def predict(self, X):
        """预测"""
        X_hog = self.hog_extractor.extract_features(X)
        X_scaled = self.scaler.transform(X_hog)
        return self.svm_model.predict(X_scaled)

    def evaluate(self, X_test, y_test):
        """评估模型"""
        print("\n" + "=" * 60)
        print("模型测试集评估")
        print("=" * 60)

        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"测试集准确率: {accuracy:.4f}")
        print("\n分类报告:")
        print(classification_report(y_test, y_pred, digits=4))

        return accuracy

    def save_model(self, save_path='optimized_model'):
        """保存模型"""
        save_dir = Path(save_path)
        save_dir.mkdir(exist_ok=True)

        model_data = {
            'svm_model': self.svm_model,
            'hog_extractor': self.hog_extractor,
            'scaler': self.scaler,
            'training_log': self.training_log
        }

        with open(save_dir / 'model.pkl', 'wb') as f:
            pickle.dump(model_data, f)

        # 保存训练日志
        log_df = pd.DataFrame(self.training_log)
        log_df.to_csv(save_dir / 'training_log.csv', index=False, encoding='utf-8-sig')

        print(f"✓ 模型已保存到: {save_path}")


def run_fixed_pipeline():
    """运行修复版训练流程"""
    print("=" * 60)
    print("GTSRB交通标志识别 - 修复版训练")
    print("=" * 60)

    # 1. 加载数据
    print("\n1. 加载数据...")
    loader = GTSRBDataLoader(data_root='data', image_size=(64, 64), normalize='minmax')

    try:
        X_train, X_val, X_test, y_train, y_val, y_test = loader.load_processed_data('processed_data')
        print(f"✓ 数据加载成功")
        print(f"  训练集: {X_train.shape[0]} 样本")
        print(f"  验证集: {X_val.shape[0]} 样本")
        print(f"  测试集: {X_test.shape[0]} 样本")
    except Exception as e:
        print(f"✗ 数据加载失败: {e}")
        return None

    # 2. 创建增强版HOG提取器
    print("\n2. 创建增强版HOG特征提取器...")

    # 实验不同的参数组合
    hog_configs = [
        {
            'name': '标准HOG',
            'params': {
                'orientations': 9,
                'pixels_per_cell': (16, 16),
                'cells_per_block': (2, 2),
                'use_color_hog': False
            }
        },
        {
            'name': '颜色HOG',
            'params': {
                'orientations': 9,
                'pixels_per_cell': (16, 16),
                'cells_per_block': (2, 2),
                'use_color_hog': True
            }
        }
    ]

    best_accuracy = 0
    best_classifier = None

    for config in hog_configs:
        print(f"\n尝试配置: {config['name']}")
        print(f"参数: {config['params']}")

        hog_extractor = EnhancedHOGFeatureExtractor(**config['params'])

        # 计算特征维度
        feature_dim = hog_extractor.get_feature_dimension(X_train[0].shape)
        print(f"  预计特征维度: {feature_dim}")

        # 3. 创建分类器
        classifier = OptimizedSVMClassifier(hog_extractor=hog_extractor)

        # 4. 训练（使用部分样本加速）
        print("\n开始训练...")
        try:
            svm, val_accuracy = classifier.train_with_validation(
                X_train, y_train, X_val, y_val,
                use_sample=True,  # 使用部分样本加速
                sample_size=2000
            )

            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_classifier = classifier
                best_config = config['name']

            print(f"\n配置 {config['name']} 完成，验证准确率: {val_accuracy:.4f}")

        except Exception as e:
            print(f"✗ 配置 {config['name']} 训练失败: {e}")
            continue

    if best_classifier is not None:
        print(f"\n✓ 最佳配置: {best_config}, 验证准确率: {best_accuracy:.4f}")

        # 5. 使用最佳配置在全量数据上训练（可选）
        print("\n是否使用全量数据训练最终模型？")
        choice = input("输入 'y' 使用全量数据训练，或直接测试 (y/n): ").strip().lower()

        if choice == 'y':
            print("\n使用全量数据训练最终模型...")
            final_hog_extractor = EnhancedHOGFeatureExtractor(**best_classifier.hog_extractor.__dict__)
            final_classifier = OptimizedSVMClassifier(hog_extractor=final_hog_extractor)

            final_svm, final_val_acc = final_classifier.train_with_validation(
                X_train, y_train, X_val, y_val,
                use_sample=False  # 使用全部数据
            )

            best_classifier = final_classifier
        else:
            print("跳过全量训练，直接测试...")

        # 6. 在测试集上评估
        print("\n6. 在测试集上评估...")
        test_accuracy = best_classifier.evaluate(X_test, y_test)

        # 7. 保存模型
        print("\n7. 保存模型...")
        best_classifier.save_model('fixed_baseline_model')

        print("\n" + "=" * 60)
        print("训练完成！")
        print(f"最终测试准确率: {test_accuracy:.4f}")
        print("=" * 60)

        return best_classifier
    else:
        print("\n✗ 所有配置都失败了")
        return None


def quick_test():
    """快速测试"""
    print("快速测试修复版模型...")

    # 使用小样本快速测试
    loader = GTSRBDataLoader(data_root='data', image_size=(64, 64), normalize='minmax')

    try:
        X_train, X_val, X_test, y_train, y_val, y_test = loader.load_processed_data('processed_data')
    except:
        print("数据加载失败")
        return

    # 只取500个样本测试
    sample_size = 500
    X_train_small = X_train[:sample_size]
    y_train_small = y_train[:sample_size]
    X_val_small = X_val[:100]
    y_val_small = y_val[:100]

    # 使用标准HOG配置
    hog_extractor = EnhancedHOGFeatureExtractor(
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        use_color_hog=False
    )

    classifier = OptimizedSVMClassifier(hog_extractor=hog_extractor)

    print(f"使用 {sample_size} 个样本进行快速测试...")
    svm, val_acc = classifier.train_with_validation(
        X_train_small, y_train_small, X_val_small, y_val_small,
        use_sample=False  # 已经采样了
    )

    print(f"\n快速测试完成，验证准确率: {val_acc:.4f}")

    if val_acc > 0.3:  # 应该明显高于随机猜测
        print("✓ 修复版模型工作正常，可以运行完整训练")
        return True
    else:
        print("⚠ 准确率仍然偏低，需要进一步调试")
        return False


if __name__ == '__main__':
    print("GTSRB交通标志识别 - 修复版")
    print("=" * 60)

    print("选择模式:")
    print("1. 快速测试 (使用500个样本)")
    print("2. 完整训练 (尝试不同配置)")

    choice = input("\n请输入选择 (1/2): ").strip()

    if choice == '1':
        success = quick_test()
        if success:
            print("\n快速测试成功！建议运行完整训练:")
            print("python baseline_model_fixed.py")
            input("按Enter键运行完整训练...")
            run_fixed_pipeline()
    else:
        run_fixed_pipeline()