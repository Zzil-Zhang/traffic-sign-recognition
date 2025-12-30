"""
hyperparameter_tuning_final_paddle.py - 超参数调优与交叉验证（PaddlePaddle版本）
成员C任务：基于成员A的数据预处理和成员B的CNN模型进行超参数调优
PaddlePaddle版本 - 支持依图加速卡
只进行超参数调优，不训练最终模型
包含过拟合/欠拟合分析
"""

import numpy as np
import paddle
import paddle.nn as nn
import paddle.optimizer as opt
from paddle.io import DataLoader, TensorDataset
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import time
import os
import json
import warnings
import sys

warnings.filterwarnings('ignore')


# ==================== 中文字体设置 ====================
def setup_chinese_font():
    """设置中文字体"""
    try:
        import matplotlib.font_manager as fm
        matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
        matplotlib.rcParams['axes.unicode_minus'] = False
        return True
    except Exception as e:
        matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
        matplotlib.rcParams['axes.unicode_minus'] = False
        return False


print("德国交通标志识别 - 超参数调优系统 (PaddlePaddle)")
print("复用：成员A的数据预处理和成员B的模型结构")
print("包含过拟合/欠拟合分析")

# ==================== 导入成员B的模型 ====================
print("\n导入成员B的模型结构...")
try:
    from cnn_model_paddle import (
        TrafficCNNPaddle,
        SimpleCNNPaddle,
        ReferenceModelPaddle,
        create_model_by_type
    )

    print("成功导入成员B的模型结构")

    MODEL_CLASSES = {
        'standard': TrafficCNNPaddle,
        'simple': SimpleCNNPaddle,
        'reference': ReferenceModelPaddle
    }

    ModelImported = True

except ImportError as e:
    print(f"无法导入成员B的模型结构: {e}")
    print("请确保 cnn_model_paddle.py 文件在相同目录下")
    ModelImported = False
    exit(1)

# ==================== 设备设置 ====================
try:
    paddle.set_device('iluvatar_gpu:0')
    print(f"使用设备: {paddle.device.get_device()}")
except Exception as e:
    print(f"无法设置依图加速卡: {e}，使用CPU")
    paddle.set_device('cpu')
    print(f"使用设备: CPU")


# ==================== 超参数优化器 ====================
class HyperparameterOptimizer:
    """超参数优化器 - 包含过拟合/欠拟合分析"""

    def __init__(self, model_type='simple'):
        """
        初始化优化器
        model_type: 'standard' (标准CNN), 'simple', 'reference'
        """
        if not ModelImported:
            print("模型导入失败，无法继续")
            exit(1)

        self.model_type = model_type
        self.best_params = None
        self.best_score = 0
        self.results = []

        # 创建结果目录
        os.makedirs('/home/aistudio/work/hyperparameter_tuning_result', exist_ok=True)

        print(f"使用模型类型: {model_type}")

    def load_data_from_processed(self):
        """
        加载预处理数据
        """
        print("\n加载预处理数据...")

        try:
            data_path = '/home/aistudio/work/processed_data/'

            X_train = np.load(f'{data_path}/X_train.npy')
            X_val = np.load(f'{data_path}/X_val.npy')
            X_test = np.load(f'{data_path}/X_test.npy')
            y_train = np.load(f'{data_path}/y_train.npy')
            y_val = np.load(f'{data_path}/y_val.npy')
            y_test = np.load(f'{data_path}/y_test.npy')

            print(f"数据加载成功！")
            print(f"训练集: {X_train.shape} - {len(y_train)} 样本")
            print(f"验证集: {X_val.shape} - {len(y_val)} 样本")
            print(f"测试集: {X_test.shape} - {len(y_test)} 样本")

            # 合并训练集和验证集用于交叉验证
            X_full = np.concatenate([X_train, X_val], axis=0)
            y_full = np.concatenate([y_train, y_val], axis=0)

            return X_full, y_full, X_test, y_test

        except Exception as e:
            print(f"数据加载失败: {e}")
            print("请确保 /home/aistudio/work/processed_data/ 目录包含所需文件")
            exit(1)

    def create_model_with_params(self, params):
        """
        创建模型，允许参数调整
        使用成员B的模型结构
        """
        num_classes = 43

        try:
            model = create_model_by_type(
                model_type=self.model_type,
                num_classes=num_classes
            )
            return model
        except Exception as e:
            print(f"模型创建失败: {e}")
            model_class = MODEL_CLASSES.get(self.model_type, SimpleCNNPaddle)
            model = model_class(num_classes=num_classes)
            return model

    def create_optimizer(self, model, params):
        """
        创建优化器
        """
        lr = params.get('learning_rate', 0.001)
        optimizer_type = params.get('optimizer_type', 'adam')

        if optimizer_type.lower() == 'sgd':
            return opt.Momentum(
                learning_rate=lr,
                momentum=0.9,
                parameters=model.parameters()
            )
        elif optimizer_type.lower() == 'rmsprop':
            return opt.RMSProp(
                learning_rate=lr,
                parameters=model.parameters()
            )
        elif optimizer_type.lower() == 'adamw':
            return opt.AdamW(
                learning_rate=lr,
                parameters=model.parameters(),
                weight_decay=0.0005
            )
        else:
            return opt.Adam(
                learning_rate=lr,
                parameters=model.parameters()
            )

    def train_epoch(self, model, train_loader, optimizer, criterion):
        """
        训练一个epoch（用于交叉验证）
        """
        model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            output = model(data)
            loss = criterion(output, target)

            loss.backward()
            optimizer.step()
            optimizer.clear_grad()

            total_loss += loss.numpy() * data.shape[0]
            pred = output.argmax(axis=1)
            total_correct += (pred == target).sum().numpy()
            total_samples += data.shape[0]

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples

        return avg_loss, accuracy

    def evaluate(self, model, data_loader, criterion):
        """
        评估模型
        """
        model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        for data, target in data_loader:
            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.numpy() * data.shape[0]
            pred = output.argmax(axis=1)
            total_correct += (pred == target).sum().numpy()
            total_samples += data.shape[0]

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples

        return avg_loss, accuracy

    def kfold_cross_validation(self, n_splits=5, batch_size=32, epochs=10):
        """
        K折交叉验证 - 评估模型稳定性，包含过拟合/欠拟合分析
        """
        print(f"\n{n_splits}-折交叉验证")
        print(f"评估模型: {self.model_type}")
        print(f"配置: epochs={epochs}, batch_size={batch_size}")

        X, y, X_test, y_test = self.load_data_from_processed()

        X = paddle.to_tensor(X, dtype='float32').transpose([0, 3, 1, 2])
        y_tensor = paddle.to_tensor(y, dtype='int64')

        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        fold_scores = []
        fold_histories = []

        for fold, (train_idx, val_idx) in enumerate(kfold.split(range(len(X)))):
            print(f"Fold {fold + 1}/{n_splits}")
            start_time = time.time()

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y_tensor[train_idx], y_tensor[val_idx]

            train_dataset = TensorDataset([X_train, y_train])
            val_dataset = TensorDataset([X_val, y_val])

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)

            model = self.create_model_with_params({
                'learning_rate': 0.001,
                'optimizer_type': 'adam'
            })

            optimizer = self.create_optimizer(model, {
                'learning_rate': 0.001,
                'optimizer_type': 'adam'
            })

            criterion = nn.CrossEntropyLoss()

            history = {
                'train_loss': [], 'train_acc': [],
                'val_loss': [], 'val_acc': []
            }

            for epoch in range(epochs):
                train_loss, train_acc = self.train_epoch(model, train_loader, optimizer, criterion)
                val_loss, val_acc = self.evaluate(model, val_loader, criterion)

                history['train_loss'].append(train_loss)
                history['train_acc'].append(train_acc)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)

            fold_time = time.time() - start_time
            final_val_acc = history['val_acc'][-1]

            fold_scores.append(final_val_acc)
            fold_histories.append(history)

            print(f"准确率: {final_val_acc:.4f} | 损失: {history['val_loss'][-1]:.4f} | 时间: {fold_time:.1f}s")

        self._analyze_kfold_results_with_overfitting(fold_scores, fold_histories, n_splits)

        return np.mean(fold_scores), np.std(fold_scores)

    def _analyze_kfold_results_with_overfitting(self, fold_scores, fold_histories, n_splits):
        """分析K折结果，包含过拟合/欠拟合分析"""
        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)

        print(f"\n{n_splits}-折交叉验证结果")
        print(f"平均准确率: {mean_score:.4f} ({mean_score:.2%})")
        print(f"标准差: {std_score:.4f}")
        print(f"各折准确率: {[f'{s:.4f}' for s in fold_scores]}")

        self._analyze_overfitting(fold_histories)
        self._plot_kfold_results(fold_scores, fold_histories)
        self._plot_detailed_training_curves(fold_histories)

        kfold_results = {
            'n_splits': n_splits,
            'model_type': self.model_type,
            'mean_accuracy': float(mean_score),
            'std_accuracy': float(std_score),
            'fold_accuracies': [float(s) for s in fold_scores],
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }

        with open('/home/aistudio/work/hyperparameter_tuning_result/kfold_results.json', 'w', encoding='utf-8') as f:
            json.dump(kfold_results, f, indent=4, ensure_ascii=False)

    def _analyze_overfitting(self, fold_histories):
        """分析过拟合/欠拟合情况"""
        print("\n过拟合/欠拟合分析:")

        all_overfitting_gaps = []

        for fold_idx, history in enumerate(fold_histories, 1):
            final_train_acc = history['train_acc'][-1]
            final_val_acc = history['val_acc'][-1]
            accuracy_gap = final_train_acc - final_val_acc

            final_train_loss = history['train_loss'][-1]
            final_val_loss = history['val_loss'][-1]
            loss_gap = final_val_loss - final_train_loss

            all_overfitting_gaps.append(accuracy_gap)

            if accuracy_gap > 0.15:
                status = "严重过拟合"
                suggestion = "增加Dropout率、增加正则化、减少模型复杂度"
            elif accuracy_gap > 0.10:
                status = "过拟合"
                suggestion = "轻微增加Dropout率、尝试数据增强"
            elif accuracy_gap > 0.05:
                status = "轻微过拟合"
                suggestion = "可以接受，或微调Dropout率"
            elif accuracy_gap > 0.02:
                status = "正常"
                suggestion = "模型拟合良好"
            elif accuracy_gap > 0:
                status = "轻微欠拟合"
                suggestion = "可以接受"
            else:
                status = "欠拟合"
                suggestion = "增加模型复杂度、减少正则化、增加训练轮数"

            print(f"Fold {fold_idx}:")
            print(f"训练准确率: {final_train_acc:.4f}")
            print(f"验证准确率: {final_val_acc:.4f}")
            print(f"准确率差距: {accuracy_gap:.4f} ({status})")
            print(f"损失差距: {loss_gap:.4f}")
            print(f"建议: {suggestion}")
            print()

        avg_gap = np.mean(all_overfitting_gaps)
        if avg_gap > 0.15:
            print(f"整体情况: 严重过拟合 (平均差距: {avg_gap:.4f})")
            print("建议: 增加Dropout率、增加正则化、减少模型复杂度")
        elif avg_gap > 0.10:
            print(f"整体情况: 过拟合 (平均差距: {avg_gap:.4f})")
            print("建议: 轻微增加Dropout率、尝试数据增强")
        elif avg_gap > 0.05:
            print(f"整体情况: 轻微过拟合 (平均差距: {avg_gap:.4f})")
            print("建议: 可以接受，或微调Dropout率")
        elif avg_gap > 0.02:
            print(f"整体情况: 良好 (平均差距: {avg_gap:.4f})")
            print("建议: 模型拟合良好")
        else:
            print(f"整体情况: 欠拟合 (平均差距: {avg_gap:.4f})")
            print("建议: 增加模型复杂度、减少正则化、增加训练轮数")

    def _plot_detailed_training_curves(self, fold_histories):
        """绘制详细的训练曲线，分析过拟合/欠拟合"""
        n_folds = len(fold_histories)
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        colors = plt.cm.Set2(np.linspace(0, 1, n_folds))

        ax1 = axes[0, 0]
        for i, history in enumerate(fold_histories):
            epochs = range(1, len(history['train_acc']) + 1)
            ax1.plot(epochs, history['train_acc'],
                     label=f'Train Fold {i + 1}', color=colors[i], linestyle='-', alpha=0.7)
            ax1.plot(epochs, history['val_acc'],
                     label=f'Val Fold {i + 1}', color=colors[i], linestyle='--', alpha=0.7)

        ax1.set_title('训练 vs 验证准确率')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('准确率')
        ax1.legend(loc='best', fontsize=8)
        ax1.grid(True, alpha=0.3)

        ax2 = axes[0, 1]
        for i, history in enumerate(fold_histories):
            epochs = range(1, len(history['train_loss']) + 1)
            ax2.plot(epochs, history['train_loss'],
                     label=f'Train Fold {i + 1}', color=colors[i], linestyle='-', alpha=0.7)
            ax2.plot(epochs, history['val_loss'],
                     label=f'Val Fold {i + 1}', color=colors[i], linestyle='--', alpha=0.7)

        ax2.set_title('训练 vs 验证损失')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('损失')
        ax2.legend(loc='best', fontsize=8)
        ax2.grid(True, alpha=0.3)

        ax3 = axes[0, 2]
        for i, history in enumerate(fold_histories):
            epochs = range(1, len(history['train_acc']) + 1)
            gap = [train - val for train, val in zip(history['train_acc'], history['val_acc'])]
            ax3.plot(epochs, gap, label=f'Fold {i + 1}', color=colors[i], linewidth=1.5, alpha=0.7)

        ax3.axhline(y=0.15, color='r', linestyle='--', alpha=0.7, label='15%过拟合阈值')
        ax3.axhline(y=0.05, color='g', linestyle='--', alpha=0.7, label='5%正常范围')
        ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax3.set_title('训练-验证准确率差距（过拟合分析）')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('准确率差距')
        ax3.legend(loc='best', fontsize=8)
        ax3.grid(True, alpha=0.3)

        ax4 = axes[1, 0]
        final_val_accs = [history['val_acc'][-1] for history in fold_histories]
        final_train_accs = [history['train_acc'][-1] for history in fold_histories]

        x = range(1, n_folds + 1)
        ax4.bar(x, final_train_accs, width=0.4, label='训练准确率', alpha=0.7, color='skyblue')
        ax4.bar([i + 0.4 for i in x], final_val_accs, width=0.4, label='验证准确率', alpha=0.7, color='lightcoral')
        ax4.set_xlabel('Fold')
        ax4.set_ylabel('最终准确率')
        ax4.set_title('各折最终准确率对比')
        ax4.set_xticks([i + 0.2 for i in x])
        ax4.set_xticklabels([f'Fold {i}' for i in x])
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        ax5 = axes[1, 1]
        for i, history in enumerate(fold_histories):
            ax5.plot(history['val_acc'], label=f'Fold {i + 1}', color=colors[i])

        ax5.set_title('收敛速度分析')
        ax5.set_xlabel('Epoch')
        ax5.set_ylabel('验证准确率')
        ax5.legend(loc='best', fontsize=8)
        ax5.grid(True, alpha=0.3)

        ax6 = axes[1, 2]
        for i, history in enumerate(fold_histories):
            ax6.plot(history['val_loss'], label=f'Fold {i + 1}', color=colors[i])

        ax6.set_title('损失收敛分析')
        ax6.set_xlabel('Epoch')
        ax6.set_ylabel('验证损失')
        ax6.legend(loc='best', fontsize=8)
        ax6.grid(True, alpha=0.3)

        plt.suptitle(f'详细训练曲线分析 - {self.model_type}模型', fontsize=16, fontweight='bold')
        plt.tight_layout()

        plot_path = '/home/aistudio/work/hyperparameter_tuning_result/detailed_training_curves.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"详细训练曲线图已保存: {plot_path}")
        plt.show()

    def random_search(self, n_iter=20, n_folds=3, epochs=5):
        """
        随机搜索超参数
        搜索空间：学习率、批大小、优化器类型
        """
        print(f"\n随机搜索超参数调优")
        print(f"迭代: {n_iter}次 | 交叉验证: {n_folds}折")
        print(f"模型: {self.model_type} | 每轮epochs: {epochs}")

        param_space = {
            'learning_rate': [0.0001, 0.0005, 0.001, 0.005, 0.01],
            'batch_size': [16, 32, 64, 128],
            'optimizer_type': ['adam', 'sgd', 'rmsprop', 'adamw']
        }

        X, y, X_test, y_test = self.load_data_from_processed()
        X = paddle.to_tensor(X, dtype='float32').transpose([0, 3, 1, 2])
        y_tensor = paddle.to_tensor(y, dtype='int64')

        results = []

        for i in range(n_iter):
            print(f"迭代 {i + 1}/{n_iter}")

            params = {
                'learning_rate': np.random.choice(param_space['learning_rate']),
                'batch_size': int(np.random.choice(param_space['batch_size'])),
                'optimizer_type': np.random.choice(param_space['optimizer_type']),
                'iteration': i + 1
            }

            print(f"参数: LR={params['learning_rate']}, BS={params['batch_size']}, Opt={params['optimizer_type']}")

            fold_scores = []
            kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)

            for fold, (train_idx, val_idx) in enumerate(kfold.split(range(len(X)))):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y_tensor[train_idx], y_tensor[val_idx]

                batch_size = int(params['batch_size'])
                train_dataset = TensorDataset([X_train, y_train])
                val_dataset = TensorDataset([X_val, y_val])

                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=batch_size)

                model = self.create_model_with_params(params)
                optimizer = self.create_optimizer(model, params)
                criterion = nn.CrossEntropyLoss()

                for epoch in range(epochs):
                    train_loss, train_acc = self.train_epoch(model, train_loader, optimizer, criterion)

                val_loss, val_acc = self.evaluate(model, val_loader, criterion)
                fold_scores.append(val_acc)

                del model, optimizer, criterion
                paddle.device.cuda.empty_cache()

            mean_score = np.mean(fold_scores)
            std_score = np.std(fold_scores)

            params['mean_accuracy'] = float(mean_score)
            params['std_accuracy'] = float(std_score)
            params['fold_accuracies'] = [float(s) for s in fold_scores]

            print(f"平均准确率: {mean_score:.4f} (±{std_score:.4f})")

            results.append(params)

            if mean_score > self.best_score:
                self.best_score = mean_score
                self.best_params = params.copy()
                print(f"新的最佳参数!")

        self.results = results
        self._save_search_results()
        self._analyze_search_results()

        return self.best_params, self.best_score

    def optimized_search(self, n_coarse=10, n_fine=3, n_folds=3, coarse_epochs=3, fine_epochs=10):
        """
        分层搜索策略：先快速筛选，再精细调优
        """
        print(f"\n分层搜索策略 (优化版)")
        print(f"阶段1: 快速筛选 {n_coarse} 个配置, {coarse_epochs} epochs")
        print(f"阶段2: 精细调优前 {n_fine} 个配置, {n_folds}折, {fine_epochs} epochs")
        print(f"模型: {self.model_type}")

        print("\n阶段1: 快速筛选")
        coarse_results = self._coarse_search(n_iter=n_coarse, epochs=coarse_epochs)

        coarse_results_sorted = sorted(coarse_results,
                                       key=lambda x: x['mean_accuracy'],
                                       reverse=True)

        print(f"\n快速筛选结果 (前{n_fine}个):")
        for i, result in enumerate(coarse_results_sorted[:n_fine]):
            print(f"{i + 1}. 准确率: {result['mean_accuracy']:.4f}")

        print(f"\n阶段2: 精细调优 (前{n_fine}个配置)")
        fine_results = self._fine_search(coarse_results_sorted[:n_fine],
                                         n_folds=n_folds,
                                         epochs=fine_epochs)

        best_fine_result = max(fine_results, key=lambda x: x['mean_accuracy'])

        self.best_score = best_fine_result['mean_accuracy']
        self.best_params = {k: v for k, v in best_fine_result.items()
                            if k not in ['mean_accuracy', 'std_accuracy', 'fold_accuracies']}

        print(f"\n分层搜索完成!")
        print(f"最佳准确率: {self.best_score:.4f} ({self.best_score:.2%})")

        self.results = fine_results
        self._save_search_results()
        self._analyze_search_results()

        return self.best_params, self.best_score

    def _coarse_search(self, n_iter=10, epochs=3):
        """
        快速筛选阶段：单折验证，少量epochs
        """
        X, y, X_test, y_test = self.load_data_from_processed()
        X = paddle.to_tensor(X, dtype='float32').transpose([0, 3, 1, 2])
        y_tensor = paddle.to_tensor(y, dtype='int64')

        param_space = {
            'learning_rate': [0.0001, 0.0005, 0.001, 0.005, 0.01],
            'batch_size': [16, 32, 64, 128],
            'optimizer_type': ['adam', 'sgd', 'rmsprop', 'adamw']
        }

        coarse_results = []

        for i in range(n_iter):
            print(f"快速测试 {i + 1}/{n_iter}...", end=' ')

            params = {
                'learning_rate': np.random.choice(param_space['learning_rate']),
                'batch_size': int(np.random.choice(param_space['batch_size'])),
                'optimizer_type': np.random.choice(param_space['optimizer_type']),
                'iteration': i + 1
            }

            from sklearn.model_selection import train_test_split
            indices = list(range(len(X)))
            train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y_tensor[train_idx], y_tensor[val_idx]

            batch_size = int(params['batch_size'])
            train_dataset = TensorDataset([X_train, y_train])
            val_dataset = TensorDataset([X_val, y_val])

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)

            model = self.create_model_with_params(params)
            optimizer = self.create_optimizer(model, params)
            criterion = nn.CrossEntropyLoss()

            for epoch in range(epochs):
                train_loss, train_acc = self.train_epoch(model, train_loader, optimizer, criterion)

            val_loss, val_acc = self.evaluate(model, val_loader, criterion)

            params['mean_accuracy'] = float(val_acc)
            params['std_accuracy'] = 0.0
            params['fold_accuracies'] = [float(val_acc)]

            coarse_results.append(params)

            print(f"准确率: {val_acc:.4f}")

            del model, optimizer, criterion
            paddle.device.cuda.empty_cache()

        return coarse_results

    def _fine_search(self, coarse_results, n_folds=3, epochs=10):
        """
        精细调优阶段：多折交叉验证，更多epochs
        """
        X, y, X_test, y_test = self.load_data_from_processed()
        X = paddle.to_tensor(X, dtype='float32').transpose([0, 3, 1, 2])
        y_tensor = paddle.to_tensor(y, dtype='int64')

        fine_results = []

        for i, coarse_params in enumerate(coarse_results):
            print(f"精细调优配置 {i + 1}/{len(coarse_results)}...")

            params = {k: v for k, v in coarse_params.items()
                      if k not in ['mean_accuracy', 'std_accuracy', 'fold_accuracies', 'iteration']}

            fold_scores = []
            kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)

            for fold, (train_idx, val_idx) in enumerate(kfold.split(range(len(X)))):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y_tensor[train_idx], y_tensor[val_idx]

                train_dataset = TensorDataset([X_train, y_train])
                val_dataset = TensorDataset([X_val, y_val])

                train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=params['batch_size'])

                model = self.create_model_with_params(params)
                optimizer = self.create_optimizer(model, params)
                criterion = nn.CrossEntropyLoss()

                for epoch in range(epochs):
                    train_loss, train_acc = self.train_epoch(model, train_loader, optimizer, criterion)

                val_loss, val_acc = self.evaluate(model, val_loader, criterion)
                fold_scores.append(val_acc)

                del model, optimizer, criterion
                paddle.device.cuda.empty_cache()

            mean_score = np.mean(fold_scores)
            std_score = np.std(fold_scores)

            params['mean_accuracy'] = float(mean_score)
            params['std_accuracy'] = float(std_score)
            params['fold_accuracies'] = [float(s) for s in fold_scores]
            params['original_rank'] = i + 1

            print(f"准确率: {mean_score:.4f} (±{std_score:.4f})")

            fine_results.append(params)

        return fine_results

    def _save_search_results(self):
        """保存搜索结果"""
        if not self.results:
            return

        sorted_results = sorted(self.results, key=lambda x: x['mean_accuracy'], reverse=True)

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

        result_path = '/home/aistudio/work/hyperparameter_tuning_result/random_search_results.json'
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(search_results, f, indent=4, ensure_ascii=False)

        df = pd.DataFrame(sorted_results)
        csv_path = '/home/aistudio/work/hyperparameter_tuning_result/search_results.csv'
        df.to_csv(csv_path, index=False)

        print(f"\n搜索结果已保存:")
        print(f"{result_path}")
        print(f"{csv_path}")

    def _analyze_search_results(self):
        """分析搜索结果"""
        if not self.results:
            return

        df = pd.DataFrame(self.results)

        print(f"\n超参数重要性分析")

        print(f"\n最佳超参数组合:")
        print(f"学习率: {self.best_params['learning_rate']}")
        print(f"批大小: {self.best_params['batch_size']}")
        print(f"优化器: {self.best_params['optimizer_type']}")
        print(f"准确率: {self.best_score:.4f} ({self.best_score:.2%})")
        print(f"标准差: {self.best_params.get('std_accuracy', 0):.4f}")

        print(f"\n超参数影响分析:")

        if 'learning_rate' in df.columns:
            lr_groups = df.groupby('learning_rate')['mean_accuracy'].agg(['mean', 'std', 'count'])
            print(f"学习率影响:")
            for lr, stats in lr_groups.iterrows():
                print(f"{lr}: {stats['mean']:.4f} (n={stats['count']})")

        if 'batch_size' in df.columns:
            bs_groups = df.groupby('batch_size')['mean_accuracy'].agg(['mean', 'std', 'count'])
            print(f"批大小影响:")
            for bs, stats in bs_groups.iterrows():
                print(f"{bs}: {stats['mean']:.4f} (n={stats['count']})")

        if 'optimizer_type' in df.columns:
            opt_groups = df.groupby('optimizer_type')['mean_accuracy'].agg(['mean', 'std', 'count'])
            print(f"优化器影响:")
            for opt, stats in opt_groups.iterrows():
                print(f"{opt}: {stats['mean']:.4f} (n={stats['count']})")

        self._plot_search_analysis(df)

    def _plot_kfold_results(self, fold_scores, fold_histories):
        """绘制K折结果"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].bar(range(1, len(fold_scores) + 1), fold_scores, color='skyblue', alpha=0.8)
        axes[0].axhline(y=np.mean(fold_scores), color='red', linestyle='--',
                        label=f'平均: {np.mean(fold_scores):.4f}')
        axes[0].set_xlabel('Fold')
        axes[0].set_ylabel('验证准确率')
        axes[0].set_title(f'{len(fold_scores)}-折交叉验证准确率')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].set_title('各折验证准确率曲线')
        colors = plt.cm.Set2(np.linspace(0, 1, len(fold_histories)))

        for i, history in enumerate(fold_histories):
            axes[1].plot(history['val_acc'], label=f'Fold {i + 1}',
                         color=colors[i], linewidth=1.5, alpha=0.7)

        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('验证准确率')
        axes[1].legend(loc='lower right', fontsize=9)
        axes[1].grid(True, alpha=0.3)

        plt.suptitle(f'K折交叉验证分析 - {self.model_type}模型', fontsize=14, fontweight='bold')
        plt.tight_layout()

        plot_path = '/home/aistudio/work/hyperparameter_tuning_result/kfold_analysis.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"K折分析图已保存: {plot_path}")
        plt.show()

    def _plot_search_analysis(self, df):
        """绘制搜索分析图"""
        if len(df) < 3:
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        if 'learning_rate' in df.columns:
            axes[0].scatter(df['learning_rate'], df['mean_accuracy'], alpha=0.6)
            axes[0].set_xlabel('学习率')
            axes[0].set_ylabel('平均准确率')
            axes[0].set_title('学习率影响')
            axes[0].set_xscale('log')
            axes[0].grid(True, alpha=0.3)

        if 'batch_size' in df.columns:
            axes[1].scatter(df['batch_size'], df['mean_accuracy'], alpha=0.6, color='green')
            axes[1].set_xlabel('批大小')
            axes[1].set_ylabel('平均准确率')
            axes[1].set_title('批大小影响')
            axes[1].grid(True, alpha=0.3)

        if 'optimizer_type' in df.columns:
            optimizer_means = df.groupby('optimizer_type')['mean_accuracy'].mean()
            axes[2].bar(range(len(optimizer_means)), optimizer_means.values,
                        tick_label=optimizer_means.index)
            axes[2].set_xlabel('优化器类型')
            axes[2].set_ylabel('平均准确率')
            axes[2].set_title('优化器性能对比')
            axes[2].grid(True, alpha=0.3)

        axes[3].plot(df.index, df['mean_accuracy'].sort_values(ascending=False).values,
                     marker='o', linewidth=1.5)
        axes[3].set_xlabel('配置排名')
        axes[3].set_ylabel('准确率')
        axes[3].set_title('超参数配置排序')
        axes[3].grid(True, alpha=0.3)

        plt.suptitle('超参数搜索结果分析', fontsize=14, fontweight='bold')
        plt.tight_layout()

        plot_path = '/home/aistudio/work/hyperparameter_tuning_result/search_analysis.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"搜索分析图已保存: {plot_path}")
        plt.show()


def main():
    """主函数"""
    setup_chinese_font()
    print("\n德国交通标志识别 - 超参数调优系统 (PaddlePaddle)")

    print("\n选择要优化的模型架构:")
    print("1. standard - 标准CNN模型")
    print("2. simple - 简单CNN模型")
    print("3. reference - 参考项目模型")

    choice = input("\n请选择模型类型 (1/2/3, 默认2): ").strip()

    model_types = {'1': 'standard', '2': 'simple', '3': 'reference'}
    model_type = model_types.get(choice, 'simple')

    optimizer = HyperparameterOptimizer(model_type=model_type)

    while True:
        print("\n超参数调优菜单")
        print("1. K折交叉验证 (评估模型稳定性，包含过拟合分析)")
        print("2. 随机搜索 (寻找最佳超参数)")
        print("3. 分层搜索 (推荐，更高效)")
        print("4. 查看当前最佳参数")
        print("5. 生成给成员B的参数建议")
        print("6. 退出")

        choice = input("请选择 (1-6): ").strip()

        if choice == '1':
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

            print(f"\n最佳参数找到! 准确率: {best_score:.4f}")

        elif choice == '3':
            print("\n分层搜索策略 (推荐)")
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

            print(f"\n分层搜索完成! 最佳准确率: {best_score:.4f}")

        elif choice == '4':
            if optimizer.best_params:
                print("\n当前最佳参数:")
                for key, value in optimizer.best_params.items():
                    if key not in ['mean_accuracy', 'std_accuracy', 'fold_accuracies', 'iteration']:
                        print(f"{key}: {value}")
                print(f"验证准确率: {optimizer.best_score:.4f}")
            else:
                print("还没有找到最佳参数，请先运行随机搜索或分层搜索")

        elif choice == '5':
            if optimizer.best_params:
                print("\n给成员B的参数建议:")
                print(f"learning_rate = {optimizer.best_params['learning_rate']}")
                print(f"batch_size = {optimizer.best_params['batch_size']}")
                print(f"optimizer = '{optimizer.best_params['optimizer_type']}'")
                print(f"model_type = '{optimizer.model_type}'")
                print(f"交叉验证结果: 平均准确率 {optimizer.best_score:.4f}")

                advice_path = '/home/aistudio/work/recommended_parameters_for_memberB.txt'
                with open(advice_path, 'w', encoding='utf-8') as f:
                    f.write("# 成员C推荐的超参数（基于K折交叉验证）\n")
                    f.write("# ============================================\n\n")
                    f.write(f"learning_rate = {optimizer.best_params['learning_rate']}\n")
                    f.write(f"batch_size = {optimizer.best_params['batch_size']}\n")
                    f.write(f"optimizer = '{optimizer.best_params['optimizer_type']}'\n")
                    f.write(f"model_type = '{optimizer.model_type}'\n\n")
                    f.write(f"# 交叉验证结果:\n")
                    f.write(f"# 平均准确率: {optimizer.best_score:.4f} ({optimizer.best_score:.2%})\n")
                    f.write(f"# 标准差: {optimizer.best_params.get('std_accuracy', 0):.4f}\n")
                    if 'fold_accuracies' in optimizer.best_params:
                        f.write(f"# 各折准确率: {optimizer.best_params['fold_accuracies']}\n")

                print(f"建议已保存到 {advice_path}")
            else:
                print("请先运行随机搜索或分层搜索找到最佳参数")

        elif choice == '6':
            print("\n退出程序")
            print("\n生成的文件:")
            print("/hyperparameter_tuning_result/ - 所有结果文件")
            print("/recommended_parameters_for_memberB.txt - 给成员B的建议")
            break

        else:
            print("无效选择")


if __name__ == "__main__":
    try:
        print(f"PaddlePaddle版本: {paddle.__version__}")
    except:
        print("请安装PaddlePaddle: pip install paddlepaddle")
        exit(1)

    data_path = '/home/aistudio/work/processed_data/'
    if not os.path.exists(data_path):
        print(f"找不到 {data_path} 目录")
        print("请先运行 data_preprocessing.py 生成预处理数据")
        exit(1)

    main()