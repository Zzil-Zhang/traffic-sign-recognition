"""
hyperparameter_tuning_final.py - 超参数调优与交叉验证（复用A&B代码）
成员C任务：基于成员A的数据预处理和成员B的CNN模型进行超参数调优
"""

import numpy as np
import tensorflow as tf
from keras.utils import to_categorical
from keras.optimizers import Adam, SGD, RMSprop
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False
import pandas as pd
import time
import os
import json
import warnings

warnings.filterwarnings('ignore')

print("=" * 60)
print("🚦 德国交通标志识别 - 超参数调优系统")
print("✅ 复用：成员A的数据预处理 + 成员B的CNN模型")
print("=" * 60)

# ==================== 强制复用现有代码 ====================
try:
    # 1. 复用成员A的数据预处理模块
    from data_preprocessing import GTSRBDataLoader

    print("✅ 成功导入 GTSRBDataLoader (成员A)")

    # 2. 复用成员B的CNN模型模块
    from cnn_model import create_traffic_cnn_model, create_simple_cnn_model, create_reference_model

    print("✅ 成功导入 CNN模型函数 (成员B)")

except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print("\n请确保以下文件存在：")
    print("  1. data_preprocessing.py - 成员A的数据预处理")
    print("  2. cnn_model.py - 成员B的CNN模型")
    print("  3. processed_data/ - 预处理数据目录")
    exit(1)

print("=" * 60)


class HyperparameterOptimizer:
    """超参数优化器 - 完全复用现有代码"""

    def __init__(self, model_type='standard'):
        """
        初始化优化器
        model_type: 'standard' (标准CNN), 'simple', 'reference'
        """
        self.model_type = model_type
        self.best_params = None
        self.best_score = 0
        self.results = []

        # 创建结果目录 - 修改为 hyperparameter_tuning_result
        os.makedirs('hyperparameter_tuning_result', exist_ok=True)

        print(f"🎯 使用模型类型: {model_type}")

    def load_data_from_processed(self):
        """
        复用成员A的预处理数据
        直接加载 processed_data 目录中的数据
        """
        print("\n📂 加载预处理数据...")

        try:
            # 直接加载 .npy 文件（最快最简单）
            X_train = np.load('processed_data/X_train.npy')
            X_val = np.load('processed_data/X_val.npy')
            X_test = np.load('processed_data/X_test.npy')
            y_train = np.load('processed_data/y_train.npy')
            y_val = np.load('processed_data/y_val.npy')
            y_test = np.load('processed_data/y_test.npy')

            print(f"✅ 数据加载成功！")
            print(f"  训练集: {X_train.shape} - {len(y_train)} 样本")
            print(f"  验证集: {X_val.shape} - {len(y_val)} 样本")
            print(f"  测试集: {X_test.shape} - {len(y_test)} 样本")

            # 合并训练集和验证集用于交叉验证
            X_full = np.concatenate([X_train, X_val], axis=0)
            y_full = np.concatenate([y_train, y_val], axis=0)

            return X_full, y_full, X_test, y_test

        except Exception as e:
            print(f"❌ 数据加载失败: {e}")
            print("请确保 processed_data/ 目录包含所需文件")
            exit(1)

    def create_model_with_params(self, params):
        """
        复用成员B的CNN模型，但允许参数调整（包括Dropout率）
        params: 包含超参数的字典
        """
        from keras.models import Sequential
        from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation

        input_shape = (64, 64, 3)  # 固定，与预处理一致
        num_classes = 43

        # 提取Dropout参数（如果提供）
        conv_dropout = params.get('conv_dropout', 0.25)  # 卷积层Dropout
        fc_dropout = params.get('fc_dropout', 0.5)      # 全连接层Dropout

        # 根据模型类型创建模型架构（支持可调Dropout）
        if self.model_type == 'simple':
            model = self._create_simple_model_with_dropout(input_shape, num_classes, conv_dropout, fc_dropout)
        elif self.model_type == 'reference':
            model = self._create_reference_model_with_dropout(input_shape, num_classes, conv_dropout, fc_dropout)
        else:  # 'standard'
            model = self._create_standard_model_with_dropout(input_shape, num_classes, conv_dropout, fc_dropout)

        # 选择优化器
        optimizer = self._get_optimizer(params)

        # 编译模型
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def _create_standard_model_with_dropout(self, input_shape, num_classes, conv_dropout, fc_dropout):
        """创建标准模型（支持可调Dropout）"""
        from keras.models import Sequential
        from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation

        model = Sequential(name="TrafficSignCNN_Tunable")

        # 第一卷积块
        model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(conv_dropout))

        # 第二卷积块
        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(conv_dropout))

        # 第三卷积块
        model.add(Conv2D(128, (3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(conv_dropout))

        # 全连接层
        model.add(Flatten())
        model.add(Dense(512))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(fc_dropout))

        # 输出层
        model.add(Dense(num_classes, activation='softmax'))

        return model

    def _create_simple_model_with_dropout(self, input_shape, num_classes, conv_dropout, fc_dropout):
        """创建简单模型（支持可调Dropout）"""
        from keras.models import Sequential
        from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation

        model = Sequential(name="SimpleTrafficCNN_Tunable")

        model.add(Conv2D(32, (3, 3), input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(conv_dropout))

        model.add(Conv2D(64, (3, 3)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(conv_dropout))

        model.add(Flatten())
        model.add(Dense(128))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(fc_dropout))

        model.add(Dense(num_classes, activation='softmax'))

        return model

    def _create_reference_model_with_dropout(self, input_shape, num_classes, conv_dropout, fc_dropout):
        """创建参考模型（支持可调Dropout）"""
        from keras.models import Sequential
        from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation

        model = Sequential(name="ReferenceWithBatchNorm_Tunable")

        model.add(Conv2D(16, (3, 3), input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(Conv2D(32, (3, 3)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(conv_dropout))

        model.add(Conv2D(64, (3, 3)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(conv_dropout))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(fc_dropout))

        model.add(Dense(num_classes, activation='softmax'))

        return model

    def _get_optimizer(self, params):
        """获取优化器"""
        lr = params.get('learning_rate', 0.001)
        optimizer_type = params.get('optimizer_type', 'adam')

        if optimizer_type.lower() == 'sgd':
            return SGD(learning_rate=lr, momentum=0.9)
        elif optimizer_type.lower() == 'rmsprop':
            return RMSprop(learning_rate=lr)
        else:  # adam
            return Adam(learning_rate=lr)

    def kfold_cross_validation(self, n_splits=5, batch_size=32, epochs=10):
        """
        K折交叉验证 - 评估模型稳定性
        完全复用成员B的模型架构
        """
        print("\n" + "=" * 60)
        print(f"📊 {n_splits}-折交叉验证")
        print(f"评估模型: {self.model_type}")
        print(f"配置: epochs={epochs}, batch_size={batch_size}")
        print("=" * 60)

        # 加载数据
        X, y, X_test, y_test = self.load_data_from_processed()
        y_onehot = to_categorical(y, 43)

        # 创建KFold
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        fold_scores = []
        fold_histories = []

        for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
            print(f"\n🔄 Fold {fold + 1}/{n_splits}")
            start_time = time.time()

            # 分割数据
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y_onehot[train_idx], y_onehot[val_idx]

            # 使用默认参数创建模型（复用B的代码）
            model = self.create_model_with_params({
                'learning_rate': 0.001,
                'optimizer_type': 'adam'
            })

            # 训练
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                verbose=0
            )

            # 评估
            val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
            fold_time = time.time() - start_time

            fold_scores.append(val_acc)
            fold_histories.append(history.history)

            print(f"  ✅ 准确率: {val_acc:.4f} | 损失: {val_loss:.4f} | 时间: {fold_time:.1f}s")

        # 统计结果
        self._analyze_kfold_results(fold_scores, fold_histories, n_splits)

        return np.mean(fold_scores), np.std(fold_scores)

    def _analyze_kfold_results(self, fold_scores, fold_histories, n_splits):
        """分析K折结果"""
        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)

        print("\n" + "=" * 60)
        print(f"📈 {n_splits}-折交叉验证结果")
        print(f"平均准确率: {mean_score:.4f} ({mean_score:.2%})")
        print(f"标准差: {std_score:.4f}")
        print(f"各折准确率: {[f'{s:.4f}' for s in fold_scores]}")

        # 绘制结果
        self._plot_kfold_results(fold_scores, fold_histories)

        # 保存结果 - 修改路径
        kfold_results = {
            'n_splits': n_splits,
            'model_type': self.model_type,
            'mean_accuracy': float(mean_score),
            'std_accuracy': float(std_score),
            'fold_accuracies': [float(s) for s in fold_scores],
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }

        with open('hyperparameter_tuning_result/kfold_results.json', 'w', encoding='utf-8') as f:
            json.dump(kfold_results, f, indent=4, ensure_ascii=False)

    def systematic_search(self, n_folds=3, epochs=5):
        """
        系统化超参数搜索 - 测试关键参数组合
        设计实验：学习率、批大小、优化器、Dropout率
        """
        print("\n" + "=" * 60)
        print("🔬 系统化超参数搜索")
        print(f"交叉验证: {n_folds}折 | 每轮epochs: {epochs}")
        print(f"模型: {self.model_type}")
        print("=" * 60)

        # 加载数据
        X, y, X_test, y_test = self.load_data_from_processed()
        y_onehot = to_categorical(y, 43)

        # 设计系统化实验
        experiments = []
        exp_id = 1

        # 实验组1: 学习率调优（固定其他参数）
        print("\n[实验组1] 学习率调优")
        learning_rates = [0.0001, 0.0005, 0.001, 0.005, 0.01]
        for lr in learning_rates:
            experiments.append({
                'id': exp_id,
                'group': 'learning_rate',
                'learning_rate': lr,
                'batch_size': 32,
                'optimizer_type': 'adam',
                'conv_dropout': 0.25,
                'fc_dropout': 0.5,
                'note': f'学习率={lr}'
            })
            exp_id += 1

        # 实验组2: 批大小调优
        print("[实验组2] 批大小调优")
        batch_sizes = [16, 32, 64, 128]
        for bs in batch_sizes:
            experiments.append({
                'id': exp_id,
                'group': 'batch_size',
                'learning_rate': 0.001,
                'batch_size': bs,
                'optimizer_type': 'adam',
                'conv_dropout': 0.25,
                'fc_dropout': 0.5,
                'note': f'批大小={bs}'
            })
            exp_id += 1

        # 实验组3: 优化器对比
        print("[实验组3] 优化器对比")
        optimizers = ['adam', 'sgd', 'rmsprop']
        for opt in optimizers:
            experiments.append({
                'id': exp_id,
                'group': 'optimizer',
                'learning_rate': 0.001,
                'batch_size': 32,
                'optimizer_type': opt,
                'conv_dropout': 0.25,
                'fc_dropout': 0.5,
                'note': f'优化器={opt}'
            })
            exp_id += 1

        # 实验组4: Dropout率调优
        print("[实验组4] Dropout率调优")
        conv_dropouts = [0.15, 0.2, 0.25, 0.3, 0.35]
        fc_dropouts = [0.3, 0.4, 0.5, 0.6, 0.7]

        for cd in conv_dropouts:
            experiments.append({
                'id': exp_id,
                'group': 'dropout_conv',
                'learning_rate': 0.001,
                'batch_size': 32,
                'optimizer_type': 'adam',
                'conv_dropout': cd,
                'fc_dropout': 0.5,
                'note': f'卷积Dropout={cd}'
            })
            exp_id += 1

        for fd in fc_dropouts:
            experiments.append({
                'id': exp_id,
                'group': 'dropout_fc',
                'learning_rate': 0.001,
                'batch_size': 32,
                'optimizer_type': 'adam',
                'conv_dropout': 0.25,
                'fc_dropout': fd,
                'note': f'全连接Dropout={fd}'
            })
            exp_id += 1

        print(f"\n总共设计 {len(experiments)} 个实验")

        # 运行所有实验
        results = []
        for exp in experiments:
            print(f"\n{'='*60}")
            print(f"实验 {exp['id']}/{len(experiments)}: {exp['note']}")
            print(f"{'='*60}")

            # K折交叉验证
            fold_scores = []
            kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)

            for fold, (train_idx, val_idx) in enumerate(kfold.split(X), 1):
                print(f"  折 {fold}/{n_folds}...", end=' ')
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y_onehot[train_idx], y_onehot[val_idx]

                # 创建模型
                model = self.create_model_with_params(exp)

                # 训练
                model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=epochs,
                    batch_size=exp['batch_size'],
                    verbose=0
                )

                # 评估
                _, val_acc = model.evaluate(X_val, y_val, verbose=0)
                fold_scores.append(val_acc)

                # 清理内存
                del model
                tf.keras.backend.clear_session()

            # 计算统计
            mean_score = np.mean(fold_scores)
            std_score = np.std(fold_scores)

            exp['mean_accuracy'] = float(mean_score)
            exp['std_accuracy'] = float(std_score)
            exp['fold_accuracies'] = [float(s) for s in fold_scores]

            print(f"\n  ✅ 平均准确率: {mean_score:.4f} (±{std_score:.4f})")

            results.append(exp)

            # 更新最佳参数
            if mean_score > self.best_score:
                self.best_score = mean_score
                self.best_params = exp.copy()
                print(f"  🎉 新的最佳参数!")

        # 保存和分析结果
        self.results = results
        self._save_search_results()
        self._analyze_search_results()

        return self.best_params, self.best_score

    def random_search(self, n_iter=20, n_folds=3, epochs=5):
        """
        随机搜索超参数
        搜索空间：学习率、批大小、优化器、Dropout率
        """
        print("\n" + "=" * 60)
        print("🎲 随机搜索超参数调优")
        print(f"迭代: {n_iter}次 | 交叉验证: {n_folds}折")
        print(f"模型: {self.model_type} | 每轮epochs: {epochs}")
        print("=" * 60)

        # 定义超参数搜索空间（包含Dropout率）
        param_space = {
            'learning_rate': [0.0001, 0.0005, 0.001, 0.005, 0.01],
            'batch_size': [16, 32, 64, 128],
            'optimizer_type': ['adam', 'sgd', 'rmsprop'],
            'conv_dropout': [0.15, 0.2, 0.25, 0.3, 0.35],  # 卷积层Dropout率
            'fc_dropout': [0.3, 0.4, 0.5, 0.6, 0.7]        # 全连接层Dropout率
        }

        # 加载数据
        X, y, X_test, y_test = self.load_data_from_processed()
        y_onehot = to_categorical(y, 43)

        results = []

        for i in range(n_iter):
            print(f"\n🔍 迭代 {i + 1}/{n_iter}")

            # 随机选择参数
            params = {
                'learning_rate': np.random.choice(param_space['learning_rate']),
                'batch_size': np.random.choice(param_space['batch_size']),
                'optimizer_type': np.random.choice(param_space['optimizer_type']),
                'conv_dropout': np.random.choice(param_space['conv_dropout']),
                'fc_dropout': np.random.choice(param_space['fc_dropout']),
                'iteration': i + 1
            }

            print(f"  参数: LR={params['learning_rate']}, "
                  f"BS={params['batch_size']}, Opt={params['optimizer_type']}, "
                  f"ConvDrop={params['conv_dropout']}, FCDrop={params['fc_dropout']}")

            # 交叉验证评估
            fold_scores = []
            kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)

            for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y_onehot[train_idx], y_onehot[val_idx]

                # 创建模型
                model = self.create_model_with_params(params)

                # 训练
                model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=epochs,
                    batch_size=params['batch_size'],
                    verbose=0
                )

                # 评估
                _, val_acc = model.evaluate(X_val, y_val, verbose=0)
                fold_scores.append(val_acc)

            # 计算平均分数
            mean_score = np.mean(fold_scores)
            std_score = np.std(fold_scores)

            params['mean_accuracy'] = float(mean_score)
            params['std_accuracy'] = float(std_score)
            params['fold_accuracies'] = [float(s) for s in fold_scores]

            print(f"  📊 平均准确率: {mean_score:.4f} (±{std_score:.4f})")

            results.append(params)

            # 更新最佳参数
            if mean_score > self.best_score:
                self.best_score = mean_score
                self.best_params = params.copy()
                print(f"  🎉 新的最佳参数!")

        # 保存结果
        self.results = results
        self._save_search_results()

        # 分析结果
        self._analyze_search_results()

        return self.best_params, self.best_score

    def optimized_search(self, n_coarse=10, n_fine=3, n_folds=3, coarse_epochs=3, fine_epochs=10):
        """
        分层搜索策略：先快速筛选，再精细调优
        解决原系统性搜索计算成本过高的问题
        """
        print("\n" + "=" * 60)
        print("🏗️  分层搜索策略 (优化版)")
        print(f"阶段1: 快速筛选 {n_coarse} 个配置, {coarse_epochs} epochs")
        print(f"阶段2: 精细调优前 {n_fine} 个配置, {n_folds}折, {fine_epochs} epochs")
        print(f"模型: {self.model_type}")
        print("=" * 60)

        # 阶段1：快速筛选
        print("\n📋 阶段1: 快速筛选")
        coarse_results = self._coarse_search(n_iter=n_coarse, epochs=coarse_epochs)

        # 按准确率排序，选择最好的几个配置
        coarse_results_sorted = sorted(coarse_results,
                                       key=lambda x: x['mean_accuracy'],
                                       reverse=True)

        print(f"\n🏆 快速筛选结果 (前{n_fine}个):")
        for i, result in enumerate(coarse_results_sorted[:n_fine]):
            print(f"  {i + 1}. 准确率: {result['mean_accuracy']:.4f} | 参数: {result}")

        # 阶段2：精细调优
        print(f"\n🔬 阶段2: 精细调优 (前{n_fine}个配置)")
        fine_results = self._fine_search(coarse_results_sorted[:n_fine],
                                         n_folds=n_folds,
                                         epochs=fine_epochs)

        # 找出最佳配置
        best_fine_result = max(fine_results, key=lambda x: x['mean_accuracy'])

        # 更新最佳参数
        self.best_score = best_fine_result['mean_accuracy']
        self.best_params = {k: v for k, v in best_fine_result.items()
                            if k not in ['mean_accuracy', 'std_accuracy', 'fold_accuracies']}

        print(f"\n🎉 分层搜索完成!")
        print(f"最佳准确率: {self.best_score:.4f} ({self.best_score:.2%})")

        # 保存结果
        self.results = fine_results
        self._save_search_results()
        self._analyze_search_results()

        return self.best_params, self.best_score

    def _coarse_search(self, n_iter=10, epochs=3):
        """
        快速筛选阶段：单折验证，少量epochs
        """
        # 加载数据
        X, y, X_test, y_test = self.load_data_from_processed()
        y_onehot = to_categorical(y, 43)

        # 定义搜索空间（与random_search相同）
        param_space = {
            'learning_rate': [0.0001, 0.0005, 0.001, 0.005, 0.01],
            'batch_size': [16, 32, 64, 128],
            'optimizer_type': ['adam', 'sgd', 'rmsprop'],
            'conv_dropout': [0.15, 0.2, 0.25, 0.3, 0.35],
            'fc_dropout': [0.3, 0.4, 0.5, 0.6, 0.7]
        }

        coarse_results = []

        for i in range(n_iter):
            print(f"  快速测试 {i + 1}/{n_iter}...", end=' ')

            # 随机选择参数
            params = {
                'learning_rate': np.random.choice(param_space['learning_rate']),
                'batch_size': np.random.choice(param_space['batch_size']),
                'optimizer_type': np.random.choice(param_space['optimizer_type']),
                'conv_dropout': np.random.choice(param_space['conv_dropout']),
                'fc_dropout': np.random.choice(param_space['fc_dropout']),
                'iteration': i + 1
            }

            # 单折快速评估（训练集80%，验证集20%）
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(
                X, y_onehot, test_size=0.2, random_state=42
            )

            # 创建并训练模型
            model = self.create_model_with_params(params)
            model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=params['batch_size'],
                verbose=0
            )

            # 评估
            _, val_acc = model.evaluate(X_val, y_val, verbose=0)

            params['mean_accuracy'] = float(val_acc)
            params['std_accuracy'] = 0.0  # 单折没有标准差
            params['fold_accuracies'] = [float(val_acc)]

            coarse_results.append(params)

            print(f"准确率: {val_acc:.4f}")

            # 清理内存
            del model
            tf.keras.backend.clear_session()

        return coarse_results

    def _fine_search(self, coarse_results, n_folds=3, epochs=10):
        """
        精细调优阶段：多折交叉验证，更多epochs
        """
        # 加载数据
        X, y, X_test, y_test = self.load_data_from_processed()
        y_onehot = to_categorical(y, 43)

        fine_results = []

        for i, coarse_params in enumerate(coarse_results):
            print(f"  精细调优配置 {i + 1}/{len(coarse_results)}...")

            # 准备参数（移除迭代信息）
            params = {k: v for k, v in coarse_params.items()
                      if k not in ['mean_accuracy', 'std_accuracy', 'fold_accuracies', 'iteration']}

            # K折交叉验证
            fold_scores = []
            kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)

            for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y_onehot[train_idx], y_onehot[val_idx]

                # 创建模型
                model = self.create_model_with_params(params)

                # 训练
                model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=epochs,
                    batch_size=params['batch_size'],
                    verbose=0
                )

                # 评估
                _, val_acc = model.evaluate(X_val, y_val, verbose=0)
                fold_scores.append(val_acc)

                # 清理内存
                del model
                tf.keras.backend.clear_session()

            # 计算统计
            mean_score = np.mean(fold_scores)
            std_score = np.std(fold_scores)

            params['mean_accuracy'] = float(mean_score)
            params['std_accuracy'] = float(std_score)
            params['fold_accuracies'] = [float(s) for s in fold_scores]
            params['original_rank'] = i + 1

            print(f"    → 准确率: {mean_score:.4f} (±{std_score:.4f})")

            fine_results.append(params)

        return fine_results

    def _save_search_results(self):
        """保存搜索结果"""
        # 排序结果
        sorted_results = sorted(self.results, key=lambda x: x['mean_accuracy'], reverse=True)

        # 保存为JSON - 修改路径
        for result in sorted_results:
            for key, value in result.items():
                if isinstance(value, (np.integer, np.int64, np.int32)):
                    result[key] = int(value)
                elif isinstance(value, (np.floating, np.float64, np.float32)):
                    result[key] = float(value)
                elif isinstance(value, np.ndarray):
                    result[key] = value.tolist()

            # 同样处理最佳参数
        if self.best_params:
            for key, value in self.best_params.items():
                if isinstance(value, (np.integer, np.int64, np.int32)):
                    self.best_params[key] = int(value)
                elif isinstance(value, (np.floating, np.float64, np.float32)):
                    self.best_params[key] = float(value)
                elif isinstance(value, np.ndarray):
                    self.best_params[key] = value.tolist()

        self.best_score = float(self.best_score) if self.best_score else 0.0

        # 保存为JSON - 修改路径
        search_results = {
            'model_type': str(self.model_type),
            'best_params': self.best_params,
            'best_score': float(self.best_score) if self.best_score else 0.0,
            'all_results': sorted_results,
            'summary': {
                'total_iterations': int(len(self.results)),
                'mean_best_score': float(
                    np.mean([r['mean_accuracy'] for r in sorted_results[:5]]) if sorted_results else 0.0),
                'timestamp': str(time.strftime("%Y-%m-%d %H:%M:%S"))
            }
        }

        with open('hyperparameter_tuning_result/random_search_results.json', 'w', encoding='utf-8') as f:
            json.dump(search_results, f, indent=4, ensure_ascii=False)

        # 保存为CSV - 修改路径
        df = pd.DataFrame(sorted_results)
        df.to_csv('hyperparameter_tuning_result/search_results.csv', index=False)

        print(f"\n✅ 搜索结果已保存:")
        print(f"  hyperparameter_tuning_result/random_search_results.json")
        print(f"  hyperparameter_tuning_result/search_results.csv")

    def _analyze_search_results(self):
        """分析搜索结果"""
        if not self.results:
            return

        df = pd.DataFrame(self.results)

        print("\n" + "=" * 60)
        print("📊 超参数重要性分析")
        print("=" * 60)

        # 1. 最佳参数
        print(f"\n🏆 最佳超参数组合:")
        print(f"  学习率: {self.best_params['learning_rate']}")
        print(f"  批大小: {self.best_params['batch_size']}")
        print(f"  优化器: {self.best_params['optimizer_type']}")
        print(f"  卷积层Dropout: {self.best_params.get('conv_dropout', 0.25)}")
        print(f"  全连接层Dropout: {self.best_params.get('fc_dropout', 0.5)}")
        print(f"  准确率: {self.best_score:.4f} ({self.best_score:.2%})")
        print(f"  标准差: {self.best_params.get('std_accuracy', 0):.4f}")

        # 2. 参数影响分析
        print(f"\n📈 超参数影响分析:")

        # 学习率影响
        if 'learning_rate' in df.columns:
            lr_groups = df.groupby('learning_rate')['mean_accuracy'].agg(['mean', 'std', 'count'])
            print(f"  学习率影响:")
            for lr, stats in lr_groups.iterrows():
                print(f"    {lr}: {stats['mean']:.4f} (n={stats['count']})")

        # 批大小影响
        if 'batch_size' in df.columns:
            bs_groups = df.groupby('batch_size')['mean_accuracy'].agg(['mean', 'std', 'count'])
            print(f"  批大小影响:")
            for bs, stats in bs_groups.iterrows():
                print(f"    {bs}: {stats['mean']:.4f} (n={stats['count']})")

        # 优化器影响
        if 'optimizer_type' in df.columns:
            opt_groups = df.groupby('optimizer_type')['mean_accuracy'].agg(['mean', 'std', 'count'])
            print(f"  优化器影响:")
            for opt, stats in opt_groups.iterrows():
                print(f"    {opt}: {stats['mean']:.4f} (n={stats['count']})")

        # 绘制可视化
        self._plot_search_analysis(df)

    def _plot_kfold_results(self, fold_scores, fold_histories):
        """绘制K折结果"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 1. 各折准确率
        axes[0].bar(range(1, len(fold_scores) + 1), fold_scores, color='skyblue', alpha=0.8)
        axes[0].axhline(y=np.mean(fold_scores), color='red', linestyle='--',
                        label=f'平均: {np.mean(fold_scores):.4f}')
        axes[0].set_xlabel('Fold')
        axes[0].set_ylabel('验证准确率')
        axes[0].set_title(f'{len(fold_scores)}-折交叉验证准确率')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 2. 训练曲线
        axes[1].set_title('各折验证准确率曲线')
        colors = plt.cm.Set2(np.linspace(0, 1, len(fold_histories)))

        for i, history in enumerate(fold_histories):
            axes[1].plot(history['val_accuracy'], label=f'Fold {i + 1}',
                         color=colors[i], linewidth=1.5, alpha=0.7)

        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('验证准确率')
        axes[1].legend(loc='lower right', fontsize=9)
        axes[1].grid(True, alpha=0.3)

        plt.suptitle(f'K折交叉验证分析 - {self.model_type}模型', fontsize=14, fontweight='bold')
        plt.tight_layout()
        # 修改保存路径
        plt.savefig('hyperparameter_tuning_result/kfold_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()

    def _plot_search_analysis(self, df):
        """绘制搜索分析图"""
        if len(df) < 3:
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        # 1. 学习率 vs 准确率
        if 'learning_rate' in df.columns:
            axes[0].scatter(df['learning_rate'], df['mean_accuracy'], alpha=0.6)
            axes[0].set_xlabel('学习率')
            axes[0].set_ylabel('平均准确率')
            axes[0].set_title('学习率影响')
            axes[0].set_xscale('log')
            axes[0].grid(True, alpha=0.3)

        # 2. 批大小 vs 准确率
        if 'batch_size' in df.columns:
            axes[1].scatter(df['batch_size'], df['mean_accuracy'], alpha=0.6, color='green')
            axes[1].set_xlabel('批大小')
            axes[1].set_ylabel('平均准确率')
            axes[1].set_title('批大小影响')
            axes[1].grid(True, alpha=0.3)

        # 3. 优化器对比
        if 'optimizer_type' in df.columns:
            optimizer_means = df.groupby('optimizer_type')['mean_accuracy'].mean()
            axes[2].bar(range(len(optimizer_means)), optimizer_means.values,
                        tick_label=optimizer_means.index)
            axes[2].set_xlabel('优化器类型')
            axes[2].set_ylabel('平均准确率')
            axes[2].set_title('优化器性能对比')
            axes[2].grid(True, alpha=0.3)

        # 4. Dropout影响分析
        if 'conv_dropout' in df.columns and 'fc_dropout' in df.columns:
            scatter = axes[3].scatter(df['conv_dropout'], df['fc_dropout'],
                                     c=df['mean_accuracy'], cmap='viridis',
                                     s=100, alpha=0.6)
            axes[3].set_xlabel('卷积层Dropout率')
            axes[3].set_ylabel('全连接层Dropout率')
            axes[3].set_title('Dropout率组合影响（颜色=准确率）')
            axes[3].grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=axes[3], label='准确率')
        else:
            # 如果没有Dropout数据，显示迭代进度
            axes[3].plot(df.index, df['mean_accuracy'].sort_values(ascending=False).values,
                         marker='o', linewidth=1.5)
            axes[3].set_xlabel('配置排名')
            axes[3].set_ylabel('准确率')
            axes[3].set_title('超参数配置排序')
            axes[3].grid(True, alpha=0.3)

        plt.suptitle('超参数搜索结果分析', fontsize=14, fontweight='bold')
        plt.tight_layout()
        # 修改保存路径
        plt.savefig('hyperparameter_tuning_result/search_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()

    def train_final_model_with_best_params(self, epochs=30):
        """
        使用最佳参数训练最终模型
        与成员B协作：将最佳参数应用到最终训练
        """
        if not self.best_params:
            print("❌ 请先运行随机搜索找到最佳参数")
            return None

        print("\n" + "=" * 60)
        print("🚀 使用最佳参数训练最终模型")
        print("=" * 60)

        # 加载完整数据
        X_train = np.load('processed_data/X_train.npy')
        X_val = np.load('processed_data/X_val.npy')
        X_test = np.load('processed_data/X_test.npy')
        y_train = np.load('processed_data/y_train.npy')
        y_val = np.load('processed_data/y_val.npy')
        y_test = np.load('processed_data/y_test.npy')

        # 合并训练集和验证集
        X_full = np.concatenate([X_train, X_val], axis=0)
        y_full = np.concatenate([y_train, y_val], axis=0)
        y_full_onehot = to_categorical(y_full, 43)
        y_test_onehot = to_categorical(y_test, 43)

        print(f"训练数据: {X_full.shape}")
        print(f"测试数据: {X_test.shape}")

        print(f"\n🎯 最佳参数:")
        print(f"  学习率: {self.best_params['learning_rate']}")
        print(f"  批大小: {self.best_params['batch_size']}")
        print(f"  优化器: {self.best_params['optimizer_type']}")
        print(f"  卷积层Dropout: {self.best_params.get('conv_dropout', 0.25)}")
        print(f"  全连接层Dropout: {self.best_params.get('fc_dropout', 0.5)}")
        print(f"  模型类型: {self.model_type}")

        # 创建最终模型
        model = self.create_model_with_params(self.best_params)

        # 训练
        print(f"\n⏳ 开始训练 ({epochs} epochs)...")
        start_time = time.time()

        history = model.fit(
            X_full, y_full_onehot,
            validation_data=(X_test, y_test_onehot),
            epochs=epochs,
            batch_size=self.best_params['batch_size'],
            verbose=1
        )

        training_time = time.time() - start_time

        # 评估
        test_loss, test_acc = model.evaluate(X_test, y_test_onehot, verbose=0)

        print("\n" + "=" * 60)
        print(f"🎯 最终模型性能:")
        print(f"  测试准确率: {test_acc:.4f} ({test_acc:.2%})")
        print(f"  测试损失: {test_loss:.4f}")
        print(f"  训练时间: {training_time:.1f}s")

        # 保存模型
        model_name = f'traffic_sign_model_tuned_{self.model_type}.keras'
        model.save(model_name)
        print(f"✅ 模型已保存: {model_name}")

        # 保存训练报告 - 修改路径
        self._save_final_report(history, test_acc, training_time)

        return model, test_acc

    def _save_final_report(self, history, test_acc, training_time):
        """保存最终训练报告"""
        report = {
            'model_type': self.model_type,
            'best_params': self.best_params,
            'performance': {
                'test_accuracy': float(test_acc),
                'final_val_accuracy': float(history.history['val_accuracy'][-1]),
                'final_train_accuracy': float(history.history['accuracy'][-1]),
                'training_time_seconds': float(training_time)
            },
            'training_history': {
                'train_accuracy': [float(x) for x in history.history['accuracy']],
                'val_accuracy': [float(x) for x in history.history['val_accuracy']],
                'train_loss': [float(x) for x in history.history['loss']],
                'val_loss': [float(x) for x in history.history['val_loss']]
            },
            'recommendations_for_memberB': [
                f"使用学习率: {self.best_params['learning_rate']}",
                f"使用批大小: {self.best_params['batch_size']}",
                f"使用优化器: {self.best_params['optimizer_type']}",
                f"使用{self.model_type}模型架构"
            ],
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }

        # 修改保存路径
        with open('hyperparameter_tuning_result/final_tuning_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=4, ensure_ascii=False)

        print(f"📋 训练报告已保存: hyperparameter_tuning_result/final_tuning_report.json")

        # 绘制训练曲线
        self._plot_training_curves(history, test_acc)

    def _plot_training_curves(self, history, test_acc):
        """绘制训练曲线"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 准确率曲线
        axes[0].plot(history.history['accuracy'], label='训练准确率', linewidth=2)
        axes[0].plot(history.history['val_accuracy'], label='测试准确率', linewidth=2)
        axes[0].axhline(y=test_acc, color='green', linestyle='--',
                        label=f'最终测试: {test_acc:.3f}')
        axes[0].set_title('模型准确率')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 损失曲线
        axes[1].plot(history.history['loss'], label='训练损失', linewidth=2)
        axes[1].plot(history.history['val_loss'], label='测试损失', linewidth=2)
        axes[1].set_title('模型损失')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.suptitle(f'超参数调优后的模型训练 - {self.model_type}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        # 修改保存路径
        plt.savefig('hyperparameter_tuning_result/final_training_curves.png', dpi=150, bbox_inches='tight')
        plt.show()


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("🎯 德国交通标志识别 - 超参数调优系统")
    print("成员C任务：基于成员A&B的工作进行模型优化")
    print("=" * 60)

    # 选择模型类型
    print("\n📋 选择要优化的模型架构:")
    print("1. standard - 标准CNN模型（成员B的主要模型）")
    print("2. simple - 简单CNN模型")
    print("3. reference - 参考项目模型")

    choice = input("\n请选择模型类型 (1/2/3, 默认1): ").strip()

    model_types = {'1': 'standard', '2': 'simple', '3': 'reference'}
    model_type = model_types.get(choice, 'standard')

    # 创建优化器
    optimizer = HyperparameterOptimizer(model_type=model_type)

    while True:
        print("\n" + "=" * 60)
        print("📱 超参数调优菜单")
        print("=" * 60)
        print("1. 📊 K折交叉验证 (评估模型稳定性)")
        print("2. 🔍 随机搜索 (寻找最佳超参数)")
        print("3. 🔬 系统化搜索 (系统测试关键参数)")
        print("4. 🏗️  分层搜索 (推荐，更高效)")
        print("5. 🚀 使用最佳参数训练最终模型")
        print("6. 📈 查看当前最佳参数")
        print("7. 📤 生成给成员B的参数建议")
        print("8. 🚪 退出")
        print("=" * 60)

        choice = input("请选择 (1-8): ").strip()

        if choice == '1':
            # K折交叉验证
            n_folds = input("折数 (默认5): ").strip()
            n_folds = int(n_folds) if n_folds else 5

            epochs = input("每折训练轮数 (默认10): ").strip()
            epochs = int(epochs) if epochs else 10

            batch_size = input("批大小 (默认32): ").strip()
            batch_size = int(batch_size) if batch_size else 32

            optimizer.kfold_cross_validation(
                n_splits=n_folds,
                epochs=epochs,
                batch_size=batch_size
            )

        elif choice == '2':
            # 随机搜索
            n_iter = input("迭代次数 (默认20): ").strip()
            n_iter = int(n_iter) if n_iter else 20

            n_folds = input("交叉验证折数 (默认3): ").strip()
            n_folds = int(n_folds) if n_folds else 3

            epochs = input("每轮训练轮数 (默认5): ").strip()
            epochs = int(epochs) if epochs else 5

            best_params, best_score = optimizer.random_search(
                n_iter=n_iter,
                n_folds=n_folds,
                epochs=epochs
            )

            print(f"\n🏆 最佳参数找到! 准确率: {best_score:.4f}")

        elif choice == '3':
            # 系统化搜索
            n_folds = input("交叉验证折数 (默认3): ").strip()
            n_folds = int(n_folds) if n_folds else 3

            epochs = input("每轮训练轮数 (默认5): ").strip()
            epochs = int(epochs) if epochs else 5

            best_params, best_score = optimizer.systematic_search(
                n_folds=n_folds,
                epochs=epochs
            )

            print(f"\n🏆 系统化搜索完成! 最佳准确率: {best_score:.4f}")

        elif choice == '4':
            # 分层搜索
            print("\n🏗️  分层搜索策略 (推荐)")
            print("先快速筛选，再精细调优，节省时间")

            n_coarse = input("快速筛选配置数 (默认10): ").strip()
            n_coarse = int(n_coarse) if n_coarse else 10

            n_fine = input("精细调优配置数 (默认3): ").strip()
            n_fine = int(n_fine) if n_fine else 3

            n_folds = input("精细调优折数 (默认3): ").strip()
            n_folds = int(n_folds) if n_folds else 3

            coarse_epochs = input("快速筛选epochs (默认3): ").strip()
            coarse_epochs = int(coarse_epochs) if coarse_epochs else 3

            fine_epochs = input("精细调优epochs (默认10): ").strip()
            fine_epochs = int(fine_epochs) if fine_epochs else 10

            best_params, best_score = optimizer.optimized_search(
                n_coarse=n_coarse,
                n_fine=n_fine,
                n_folds=n_folds,
                coarse_epochs=coarse_epochs,
                fine_epochs=fine_epochs
            )

            print(f"\n🏆 分层搜索完成! 最佳准确率: {best_score:.4f}")

        elif choice == '5':
            # 使用最佳参数训练最终模型
            if not optimizer.best_params:
                print("请先运行随机搜索找到最佳参数")
                continue

            epochs = input("训练轮数 (默认30): ").strip()
            epochs = int(epochs) if epochs else 30

            model, test_acc = optimizer.train_final_model_with_best_params(epochs=epochs)

            if model:
                print(f"\n✅ 最终模型训练完成! 测试准确率: {test_acc:.2%}")

        elif choice == '6':
            # 查看最佳参数
            if optimizer.best_params:
                print("\n📋 当前最佳参数:")
                for key, value in optimizer.best_params.items():
                    if key not in ['mean_accuracy', 'std_accuracy', 'fold_accuracies', 'iteration']:
                        print(f"  {key}: {value}")
                print(f"  验证准确率: {optimizer.best_score:.4f}")
            else:
                print("还没有找到最佳参数，请先运行随机搜索")

        elif choice == '7':
            # 生成给成员B的建议
            if optimizer.best_params:
                print("\n📤 给成员B的参数建议:")
                print("=" * 40)
                print("建议在 train_cnn.py 中使用以下参数:")
                print("=" * 40)
                print(f"learning_rate = {optimizer.best_params['learning_rate']}")
                print(f"batch_size = {optimizer.best_params['batch_size']}")
                print(f"optimizer = '{optimizer.best_params['optimizer_type']}'")
                print(f"conv_dropout = {optimizer.best_params.get('conv_dropout', 0.25)}")
                print(f"fc_dropout = {optimizer.best_params.get('fc_dropout', 0.5)}")
                print(f"model_type = '{optimizer.model_type}'")
                print("=" * 40)
                print("说明: 这些参数在交叉验证中表现最佳")

                # 保存建议
                with open('recommended_parameters_for_memberB.txt', 'w', encoding='utf-8') as f:
                    f.write("# 成员C推荐的超参数（基于K折交叉验证）\n")
                    f.write("# ============================================\n\n")
                    f.write(f"learning_rate = {optimizer.best_params['learning_rate']}\n")
                    f.write(f"batch_size = {optimizer.best_params['batch_size']}\n")
                    f.write(f"optimizer = '{optimizer.best_params['optimizer_type']}'\n")
                    f.write(f"conv_dropout = {optimizer.best_params.get('conv_dropout', 0.25)}\n")
                    f.write(f"fc_dropout = {optimizer.best_params.get('fc_dropout', 0.5)}\n")
                    f.write(f"model_type = '{optimizer.model_type}'\n\n")
                    f.write(f"# 交叉验证结果:\n")
                    f.write(f"# 平均准确率: {optimizer.best_score:.4f} ({optimizer.best_score:.2%})\n")
                    f.write(f"# 标准差: {optimizer.best_params.get('std_accuracy', 0):.4f}\n")
                    f.write(f"# 各折准确率: {optimizer.best_params.get('fold_accuracies', [])}\n")

                print("✅ 建议已保存到 recommended_parameters_for_memberB.txt")
            else:
                print("请先运行随机搜索找到最佳参数")

        elif choice == '8':
            print("\n👋 退出程序")
            print("\n📁 生成的文件:")
            print("  hyperparameter_tuning_result/ - 所有结果文件")
            print("  recommended_parameters_for_memberB.txt - 给成员B的建议")
            break

        else:
            print("❌ 无效选择")


if __name__ == "__main__":
    # 检查环境
    try:
        print(f"TensorFlow版本: {tf.__version__}")
    except:
        print("请安装TensorFlow: pip install tensorflow")
        exit(1)

    # 检查数据目录
    if not os.path.exists('processed_data'):
        print("❌ 找不到 processed_data/ 目录")
        print("请先运行 data_preprocessing.py 生成预处理数据")
        exit(1)

    # 运行主程序
    main()