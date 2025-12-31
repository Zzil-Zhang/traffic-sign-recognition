# hyperparameter_tuning_final_paddle.py
"""
 用于优化CNN模型超参数和五折交叉验证
"""
import os
import time
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import paddle
import paddle.nn as nn
import paddle.optimizer as opt
from paddle.io import DataLoader, TensorDataset
from sklearn.model_selection import KFold

warnings.filterwarnings('ignore')


def setup_chinese_font():
    """Try to set a Chinese-capable font for matplotlib (best-effort)."""
    try:
        import matplotlib.font_manager as fm
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        if any('WenQuanYi' in font for font in available_fonts):
            matplotlib.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
            print("使用文泉驿中文字体")
        elif any('SimHei' in font for font in available_fonts):
            matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
            print("使用SimHei中文字体")
        else:
            matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
            print("使用英文字体")
        matplotlib.rcParams['axes.unicode_minus'] = False
        return True
    except Exception as e:
        print(f"字体设置失败: {e}")
        matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
        matplotlib.rcParams['axes.unicode_minus'] = False
        return False


def setup_device():
    """使用设备 iluvatar_gpu:0 -> gpu -> cpu."""
    try:
        paddle.set_device('iluvatar_gpu:0')
    except Exception:
        try:
            paddle.set_device('gpu')
        except Exception:
            paddle.set_device('cpu')
    return paddle.device.get_device()


from cnn_model_paddle import create_model_by_type
from data_utils import load_data_from_npy


class HyperparameterOptimizer:

    def __init__(self, model_type='simple',
                 weights_path='trained_models/traffic_sign_cnn_paddle_final_20251228_003218.pdparams',
                 use_pretrained=True):
        self.model_type = model_type
        self.weights_path = weights_path if (use_pretrained and weights_path) else None
        self.best_params = None
        self.best_score = 0.0
        self.results = []
        os.makedirs('/home/aistudio/work/hyperparameter_tuning_result', exist_ok=True)
        print(f"模型类型: {model_type} | 设备: {paddle.device.get_device()}")
        if self.weights_path and os.path.exists(self.weights_path):
            print(f"初始化权重: {self.weights_path}")
        elif use_pretrained and self.weights_path:
            print(f"未找到初始化权重文件: {self.weights_path}，将从随机初始化开始")

    def _load_data(self):
        """Load preprocess npy using data_utils.load_data_from_npy and combine train+val for CV."""
        train, val, test = load_data_from_npy()
        if train is None:
            print("数据加载失败，退出")
            exit(1)
        X_train, y_train = train
        X_val, y_val = val
        X_test, y_test = test
        X_full = np.concatenate([X_train, X_val], axis=0)
        y_full = np.concatenate([y_train, y_val], axis=0)
        return X_full, y_full, X_test, y_test

    def _create_model(self, num_classes=43):
        model = create_model_by_type(model_type=self.model_type, num_classes=num_classes)
        if self.weights_path and os.path.exists(self.weights_path):
            try:
                state_dict = paddle.load(self.weights_path)
                model.set_state_dict(state_dict)
            except Exception as e:
                print(f"预训练权重加载失败: {e}")
        return model

    def _create_optimizer(self, model, params):
        lr = params.get('learning_rate', 0.001)
        opt_type = params.get('optimizer_type', 'adam').lower()
        if opt_type == 'sgd':
            return opt.Momentum(learning_rate=lr, momentum=0.9, parameters=model.parameters())
        elif opt_type == 'rmsprop':
            return opt.RMSProp(learning_rate=lr, parameters=model.parameters())
        elif opt_type == 'adamw':
            return opt.AdamW(learning_rate=lr, parameters=model.parameters(), weight_decay=0.0005)
        else:
            return opt.Adam(learning_rate=lr, parameters=model.parameters())

    def _train_epoch(self, model, train_loader, optimizer, criterion):
        model.train()
        total_loss, total_correct, total_samples = 0.0, 0, 0
        for data, target in train_loader:
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            total_loss += float(loss.numpy()) * data.shape[0]
            pred = output.argmax(axis=1)
            total_correct += int((pred == target).sum().numpy())
            total_samples += data.shape[0]
        return total_loss / total_samples, total_correct / total_samples

    def _evaluate(self, model, data_loader, criterion):
        model.eval()
        total_loss, total_correct, total_samples = 0.0, 0, 0
        with paddle.no_grad():
            for data, target in data_loader:
                output = model(data)
                loss = criterion(output, target)
                total_loss += float(loss.numpy()) * data.shape[0]
                pred = output.argmax(axis=1)
                total_correct += int((pred == target).sum().numpy())
                total_samples += data.shape[0]
        return total_loss / total_samples, total_correct / total_samples

    def kfold_cross_validation(self, n_splits=5, batch_size=32, epochs=10):
        print(f"K折交叉验证: {n_splits} 折 | epochs={epochs} | batch_size={batch_size}")
        X, y, _, _ = self._load_data()
        X = paddle.to_tensor(X, dtype='float32').transpose([0, 3, 1, 2])
        y_tensor = paddle.to_tensor(y, dtype='int64')

        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_scores, fold_histories = [], []

        for fold, (train_idx, val_idx) in enumerate(kfold.split(range(len(X))), 1):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y_tensor[train_idx], y_tensor[val_idx]

            train_loader = DataLoader(TensorDataset([X_train, y_train]), batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(TensorDataset([X_val, y_val]), batch_size=batch_size)

            model = self._create_model(num_classes=43)
            optimizer = self._create_optimizer(model, {'learning_rate': 0.001, 'optimizer_type': 'adam'})
            criterion = nn.CrossEntropyLoss()

            history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
            start = time.time()

            for ep in range(epochs):
                tl_trainmode, ta_trainmode = self._train_epoch(model, train_loader, optimizer, criterion)

                vl, va = self._evaluate(model, val_loader, criterion)

                tl_eval, ta_eval = self._evaluate(model, train_loader, criterion)

                history['train_loss'].append(tl_eval)
                history['train_acc'].append(ta_eval)
                history['val_loss'].append(vl)
                history['val_acc'].append(va)

                print(f"Fold {fold} - Epoch {ep+1:02d}/{epochs} | "
                      f"train_loss: {tl_eval:.4f}, train_acc: {ta_eval:.4f} | "
                      f"val_loss: {vl:.4f}, val_acc: {va:.4f}")

            elapsed = time.time() - start
            final_train_acc = history['train_acc'][-1]
            final_val_acc = history['val_acc'][-1]
            gap = final_train_acc - final_val_acc

            print(f"Fold {fold} 完成 - train_acc: {final_train_acc:.4f}, val_acc: {final_val_acc:.4f}, gap: {gap:.4f} | time={elapsed:.1f}s")

            fold_scores.append(final_val_acc)
            fold_histories.append(history)

            del model, optimizer, criterion

        self._analyze_kfold_results(fold_scores, fold_histories, n_splits)
        return float(np.mean(fold_scores)), float(np.std(fold_scores))

    def random_search(self, n_iter=20, n_folds=5, epochs=5):
        print(f"随机搜索: {n_iter} 次 | {n_folds} 折 | epochs={epochs}")
        X, y, _, _ = self._load_data()
        X = paddle.to_tensor(X, dtype='float32').transpose([0, 3, 1, 2])
        y_tensor = paddle.to_tensor(y, dtype='int64')

        param_space = {
            'learning_rate': [0.0001, 0.0005, 0.001, 0.005, 0.01],
            'batch_size': [16, 32, 64, 128],
            'optimizer_type': ['adam', 'sgd', 'rmsprop', 'adamw']
        }

        results = []
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)

        for i in range(n_iter):
            params = {
                'learning_rate': float(np.random.choice(param_space['learning_rate'])),
                'batch_size': int(np.random.choice(param_space['batch_size'])),
                'optimizer_type': str(np.random.choice(param_space['optimizer_type'])),
                'iteration': int(i + 1)
            }
            fold_scores = []
            for fold, (train_idx, val_idx) in enumerate(kfold.split(range(len(X))), 1):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y_tensor[train_idx], y_tensor[val_idx]
                train_loader = DataLoader(TensorDataset([X_train, y_train]), batch_size=params['batch_size'], shuffle=True)
                val_loader = DataLoader(TensorDataset([X_val, y_val]), batch_size=params['batch_size'])

                model = self._create_model(num_classes=43)
                optimizer = self._create_optimizer(model, params)
                criterion = nn.CrossEntropyLoss()

                for _ in range(epochs):
                    self._train_epoch(model, train_loader, optimizer, criterion)
                _, val_acc = self._evaluate(model, val_loader, criterion)
                fold_scores.append(float(val_acc))

                del model, optimizer, criterion

            mean_score = float(np.mean(fold_scores))
            std_score = float(np.std(fold_scores))
            params['mean_accuracy'] = mean_score
            params['std_accuracy'] = std_score
            params['fold_accuracies'] = [float(s) for s in fold_scores]
            print(f"Iter {i+1}: {params} | mean_acc={mean_score:.4f}")
            results.append(params)

            if mean_score > self.best_score:
                self.best_score = mean_score
                self.best_params = params.copy()

        self.results = results
        self._save_search_results()
        self._analyze_search_results()
        return self.best_params, self.best_score

    # Stage search: coarse -> fine
    def optimized_search(self, n_coarse=10, n_fine=3, n_folds=5, coarse_epochs=3, fine_epochs=10):
        print(f"分层搜索: 粗筛{n_coarse} -> 精调{n_fine} | 折数={n_folds}")
        coarse = self._coarse_search(n_iter=n_coarse, epochs=coarse_epochs)
        coarse_sorted = sorted(coarse, key=lambda x: x['mean_accuracy'], reverse=True)[:n_fine]
        fine = self._fine_search(coarse_sorted, n_folds=n_folds, epochs=fine_epochs)
        best = max(fine, key=lambda x: x['mean_accuracy'])
        self.best_score = float(best['mean_accuracy'])
        self.best_params = {k: v for k, v in best.items() if k not in ['mean_accuracy', 'std_accuracy', 'fold_accuracies', 'original_rank']}
        self.results = fine
        self._save_search_results()
        self._analyze_search_results()
        print(f"完成 | best_acc={self.best_score:.4f}")
        return self.best_params, self.best_score

    def _coarse_search(self, n_iter=10, epochs=3):
        X, y, _, _ = self._load_data()
        X = paddle.to_tensor(X, dtype='float32').transpose([0, 3, 1, 2])
        y_tensor = paddle.to_tensor(y, dtype='int64')
        param_space = {
            'learning_rate': [0.0001, 0.0005, 0.001, 0.005, 0.01],
            'batch_size': [16, 32, 64, 128],
            'optimizer_type': ['adam', 'sgd', 'rmsprop', 'adamw']
        }
        results = []
        from sklearn.model_selection import train_test_split
        for i in range(n_iter):
            params = {
                'learning_rate': float(np.random.choice(param_space['learning_rate'])),
                'batch_size': int(np.random.choice(param_space['batch_size'])),
                'optimizer_type': str(np.random.choice(param_space['optimizer_type'])),
                'iteration': int(i + 1)
            }
            indices = list(range(len(X)))
            train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y_tensor[train_idx], y_tensor[val_idx]
            train_loader = DataLoader(TensorDataset([X_train, y_train]), batch_size=params['batch_size'], shuffle=True)
            val_loader = DataLoader(TensorDataset([X_val, y_val]), batch_size=params['batch_size'])

            model = self._create_model(num_classes=43)
            optimizer = self._create_optimizer(model, params)
            criterion = nn.CrossEntropyLoss()

            for _ in range(epochs):
                self._train_epoch(model, train_loader, optimizer, criterion)
            _, val_acc = self._evaluate(model, val_loader, criterion)

            params['mean_accuracy'] = float(val_acc)
            params['std_accuracy'] = 0.0
            params['fold_accuracies'] = [float(val_acc)]
            results.append(params)

            del model, optimizer, criterion
            print(f"快速 {i+1}/{n_iter}: val_acc={val_acc:.4f}")

        return results

    def _fine_search(self, coarse_results, n_folds=5, epochs=10):
        X, y, _, _ = self._load_data()
        X = paddle.to_tensor(X, dtype='float32').transpose([0, 3, 1, 2])
        y_tensor = paddle.to_tensor(y, dtype='int64')

        results = []
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)

        for i, base in enumerate(coarse_results, 1):
            params = {k: v for k, v in base.items() if k not in ['mean_accuracy', 'std_accuracy', 'fold_accuracies', 'iteration']}
            fold_scores = []
            for fold, (train_idx, val_idx) in enumerate(kfold.split(range(len(X))), 1):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y_tensor[train_idx], y_tensor[val_idx]
                train_loader = DataLoader(TensorDataset([X_train, y_train]), batch_size=params['batch_size'], shuffle=True)
                val_loader = DataLoader(TensorDataset([X_val, y_val]), batch_size=params['batch_size'])
                model = self._create_model(num_classes=43)
                optimizer = self._create_optimizer(model, params)
                criterion = nn.CrossEntropyLoss()
                for _ in range(epochs):
                    self._train_epoch(model, train_loader, optimizer, criterion)
                _, val_acc = self._evaluate(model, val_loader, criterion)
                fold_scores.append(float(val_acc))
                del model, optimizer, criterion
            mean_score = float(np.mean(fold_scores))
            std_score = float(np.std(fold_scores))
            params['mean_accuracy'] = mean_score
            params['std_accuracy'] = std_score
            params['fold_accuracies'] = [float(s) for s in fold_scores]
            params['original_rank'] = int(i)
            results.append(params)
            print(f"精调 {i}/{len(coarse_results)}: mean_acc={mean_score:.4f} (±{std_score:.4f})")
        return results

    def _analyze_kfold_results(self, fold_scores, fold_histories, n_splits):
        mean_score = float(np.mean(fold_scores))
        std_score = float(np.std(fold_scores))
        print(f"CV结果: mean_acc={mean_score:.4f} (±{std_score:.4f}) | per-fold={['%.4f'%s for s in fold_scores]}")
        self._analyze_overfitting(fold_histories)
        self._plot_kfold_results(fold_scores, fold_histories)
        self._plot_detailed_training_curves(fold_histories)
        out = {
            'n_splits': n_splits,
            'model_type': self.model_type,
            'mean_accuracy': mean_score,
            'std_accuracy': std_score,
            'fold_accuracies': [float(s) for s in fold_scores],
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        with open('/home/aistudio/work/hyperparameter_tuning_result/kfold_results.json', 'w', encoding='utf-8') as f:
            json.dump(out, f, indent=4, ensure_ascii=False)

    def _analyze_overfitting(self, fold_histories):
        print("过拟合/欠拟合分析:")
        gaps = []
        for i, h in enumerate(fold_histories, 1):
            train_acc = h['train_acc'][-1]
            val_acc = h['val_acc'][-1]
            gap = train_acc - val_acc
            gaps.append(gap)
            if gap > 0.15:
                status, suggestion = "严重过拟合", "增加Dropout/正则化，降低模型复杂度"
            elif gap > 0.10:
                status, suggestion = "过拟合", "适度增加Dropout或数据增强"
            elif gap > 0.05:
                status, suggestion = "轻微过拟合", "可接受或微调Dropout"
            elif gap > 0.02:
                status, suggestion = "正常", "拟合良好"
            elif gap > 0:
                status, suggestion = "轻微欠拟合", "可接受"
            else:
                status, suggestion = "欠拟合", "增加模型复杂度/训练轮数"
            print(f"Fold {i}: gap={gap:.4f} ({status}) | 建议: {suggestion}")

        avg_gap = float(np.mean(gaps))
        overall = ("严重过拟合" if avg_gap > 0.15 else
                   "过拟合" if avg_gap > 0.10 else
                   "轻微过拟合" if avg_gap > 0.05 else
                   "良好" if avg_gap > 0.02 else
                   "欠拟合")
        print(f"整体: {overall} | 平均差距={avg_gap:.4f}")

    def _plot_kfold_results(self, fold_scores, fold_histories):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].bar(range(1, len(fold_scores)+1), fold_scores, color='skyblue', alpha=0.8)
        axes[0].axhline(y=np.mean(fold_scores), color='red', linestyle='--',
                        label=f'平均: {np.mean(fold_scores):.4f}')
        axes[0].set_title('各折验证准确率')
        axes[0].legend(); axes[0].grid(True, alpha=0.3)
        colors = plt.cm.Set2(np.linspace(0, 1, len(fold_histories)))
        for i, h in enumerate(fold_histories):
            axes[1].plot(h['val_acc'], label=f'Fold {i+1}', color=colors[i], linewidth=1.5, alpha=0.7)
        axes[1].set_title('验证准确率曲线')
        axes[1].legend(loc='lower right', fontsize=9); axes[1].grid(True, alpha=0.3)
        plt.suptitle(f'K折交叉验证 - {self.model_type}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        path = '/home/aistudio/work/hyperparameter_tuning_result/kfold_analysis.png'
        plt.savefig(path, dpi=150, bbox_inches='tight'); print(f"保存: {path}")
        plt.show()

    def _plot_detailed_training_curves(self, fold_histories):
        n = len(fold_histories)
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        colors = plt.cm.Set2(np.linspace(0, 1, n))

        ax1 = axes[0, 0]
        for i, h in enumerate(fold_histories):
            ep = range(1, len(h['train_acc']) + 1)
            ax1.plot(ep, h['train_acc'], label=f'Train {i+1}', color=colors[i], alpha=0.7)
            ax1.plot(ep, h['val_acc'], label=f'Val {i+1}', color=colors[i], linestyle='--', alpha=0.7)
        ax1.set_title('训练 vs 验证准确率'); ax1.legend(); ax1.grid(True, alpha=0.3)

        ax2 = axes[0, 1]
        for i, h in enumerate(fold_histories):
            ep = range(1, len(h['train_loss']) + 1)
            ax2.plot(ep, h['train_loss'], label=f'Train {i+1}', color=colors[i], alpha=0.7)
            ax2.plot(ep, h['val_loss'], label=f'Val {i+1}', color=colors[i], linestyle='--', alpha=0.7)
        ax2.set_title('训练 vs 验证损失'); ax2.legend(); ax2.grid(True, alpha=0.3)

        ax3 = axes[0, 2]
        for i, h in enumerate(fold_histories):
            ep = range(1, len(h['train_acc']) + 1)
            gap = [t - v for t, v in zip(h['train_acc'], h['val_acc'])]
            ax3.plot(ep, gap, label=f'Fold {i+1}', color=colors[i], linewidth=1.5, alpha=0.7)
        ax3.axhline(y=0.15, color='r', linestyle='--', alpha=0.7, label='15%过拟合阈值')
        ax3.axhline(y=0.05, color='g', linestyle='--', alpha=0.7, label='5%正常范围')
        ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax3.set_title('训练-验证准确率差距'); ax3.legend(); ax3.grid(True, alpha=0.3)

        ax4 = axes[1, 0]
        final_val = [h['val_acc'][-1] for h in fold_histories]
        final_train = [h['train_acc'][-1] for h in fold_histories]
        x = range(1, n + 1)
        ax4.bar(x, final_train, width=0.4, label='训练', alpha=0.7, color='skyblue')
        ax4.bar([i + 0.4 for i in x], final_val, width=0.4, label='验证', alpha=0.7, color='lightcoral')
        ax4.set_title('各折最终准确率'); ax4.legend(); ax4.grid(True, alpha=0.3)

        ax5 = axes[1, 1]
        for i, h in enumerate(fold_histories):
            ax5.plot(h['val_acc'], label=f'Fold {i + 1}', color=colors[i])
        ax5.set_title('收敛速度'); ax5.legend(); ax5.grid(True, alpha=0.3)

        ax6 = axes[1, 2]
        for i, h in enumerate(fold_histories):
            ax6.plot(h['val_loss'], label=f'Fold {i + 1}', color=colors[i])
        ax6.set_title('损失收敛'); ax6.legend(); ax6.grid(True, alpha=0.3)

        plt.suptitle(f'详细训练曲线 - {self.model_type}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        path = '/home/aistudio/work/hyperparameter_tuning_result/detailed_training_curves.png'
        plt.savefig(path, dpi=150, bbox_inches='tight'); print(f"保存: {path}")
        plt.show()

    def _save_search_results(self):
        if not self.results:
            return
        sorted_results = sorted(self.results, key=lambda x: x['mean_accuracy'], reverse=True)
        out = {
            'model_type': str(self.model_type),
            'best_params': self.best_params,
            'best_score': float(self.best_score) if self.best_score else 0.0,
            'all_results': sorted_results,
            'summary': {
                'total_iterations': int(len(self.results)),
                'mean_best_score': float(np.mean([r['mean_accuracy'] for r in sorted_results[:5]]) if sorted_results else 0.0),
                'timestamp': str(time.strftime("%Y-%m-%d %H:%M:%S"))
            }
        }
        path_json = '/home/aistudio/work/hyperparameter_tuning_result/random_search_results.json'
        path_csv = '/home/aistudio/work/hyperparameter_tuning_result/search_results.csv'
        with open(path_json, 'w', encoding='utf-8') as f:
            json.dump(out, f, indent=4, ensure_ascii=False)
        pd.DataFrame(sorted_results).to_csv(path_csv, index=False)
        print(f"保存: {path_json}\n保存: {path_csv}")

    def _analyze_search_results(self):
        if not self.results:
            return
        df = pd.DataFrame(self.results)
        print("\n超参数影响分析/最佳组合")
        if self.best_params:
            print(f"best: lr={self.best_params.get('learning_rate')} "
                  f"bs={self.best_params.get('batch_size')} "
                  f"opt={self.best_params.get('optimizer_type')} "
                  f"acc={self.best_score:.4f}")
        for col in ['learning_rate', 'batch_size', 'optimizer_type']:
            if col in df.columns:
                grp = df.groupby(col)['mean_accuracy'].agg(['mean', 'std', 'count'])
                print(f"{col}:\n{grp}")
        self._plot_search_analysis(df)

    def _plot_search_analysis(self, df):
        if len(df) < 3:
            return
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        if 'learning_rate' in df.columns:
            axes[0].scatter(df['learning_rate'], df['mean_accuracy'], alpha=0.6)
            axes[0].set_xscale('log'); axes[0].set_title('学习率影响'); axes[0].grid(True, alpha=0.3)
        if 'batch_size' in df.columns:
            axes[1].scatter(df['batch_size'], df['mean_accuracy'], alpha=0.6, color='green')
            axes[1].set_title('批大小影响'); axes[1].grid(True, alpha=0.3)
        if 'optimizer_type' in df.columns:
            means = df.groupby('optimizer_type')['mean_accuracy'].mean()
            axes[2].bar(range(len(means)), means.values, tick_label=means.index)
            axes[2].set_title('优化器对比'); axes[2].grid(True, alpha=0.3)
        order = df['mean_accuracy'].sort_values(ascending=False).values
        axes[3].plot(range(len(order)), order, marker='o', linewidth=1.5)
        axes[3].set_title('配置排序'); axes[3].grid(True, alpha=0.3)
        plt.suptitle('超参数搜索结果分析', fontsize=14, fontweight='bold')
        plt.tight_layout()
        path = '/home/aistudio/work/hyperparameter_tuning_result/search_analysis.png'
        plt.savefig(path, dpi=150, bbox_inches='tight'); print(f"保存: {path}")
        plt.show()


def main():
    setup_chinese_font()
    device = setup_device()
    print(f"设备: {device}")
    print("\n选择模型类型: 1=standard 2=simple 3=reference")
    choice = input("输入(1/2/3, 默认2): ").strip()
    model_types = {'1': 'standard', '2': 'simple', '3': 'reference'}
    model_type = model_types.get(choice, 'simple')

    weights_default = 'trained_models/traffic_sign_cnn_paddle_final_20251228_003218.pdparams'
    use_pretrained = os.path.exists(weights_default)
    optimizer = HyperparameterOptimizer(model_type=model_type,
                                        weights_path=weights_default,
                                        use_pretrained=use_pretrained)

    while True:
        print("\n菜单: 1=K折CV 2=随机搜索 3=分层搜索 4=查看最佳 5=退出")
        choice = input("选择(1-5): ").strip()
        if choice == '1':
            n_folds = int(input("折数(默认5): ").strip() or 5)
            epochs = int(input("每折epochs(默认10): ").strip() or 10)
            batch_size = int(input("批大小(默认32): ").strip() or 32)
            optimizer.kfold_cross_validation(n_splits=n_folds, epochs=epochs, batch_size=batch_size)
        elif choice == '2':
            n_iter = int(input("迭代(默认20): ").strip() or 20)
            n_folds = int(input("折数(默认5): ").strip() or 5)
            epochs = int(input("每轮epochs(默认5): ").strip() or 5)
            optimizer.random_search(n_iter=n_iter, n_folds=n_folds, epochs=epochs)
        elif choice == '3':
            n_coarse = int(input("快速筛选数(默认10): ").strip() or 10)
            n_fine = int(input("精调个数(默认3): ").strip() or 3)
            n_folds = int(input("精调折数(默认5): ").strip() or 5)
            ce = int(input("快速epochs(默认3): ").strip() or 3)
            fe = int(input("精调epochs(默认10): ").strip() or 10)
            optimizer.optimized_search(n_coarse=n_coarse, n_fine=n_fine, n_folds=n_folds,
                                       coarse_epochs=ce, fine_epochs=fe)
        elif choice == '4':
            if optimizer.best_params:
                print("当前最佳参数:")
                for k, v in optimizer.best_params.items():
                    if k not in ['mean_accuracy', 'std_accuracy', 'fold_accuracies', 'iteration', 'original_rank']:
                        print(f" - {k}: {v}")
                print(f"验证准确率: {optimizer.best_score:.4f}")
            else:
                print("尚未找到最佳参数")
        elif choice == '5':
            print("退出")
            break
        else:
            print("无效选择")


if __name__ == "__main__":
    try:
        print(f"PaddlePaddle版本: {paddle.__version__}")
    except Exception:
        print("请安装PaddlePaddle: pip install paddlepaddle")
        exit(1)
    if not os.path.exists('processed_data') and not os.path.exists('/home/aistudio/work/processed_data'):
        print("找不到 processed_data 目录，请先生成预处理数据")
        exit(1)
    main()