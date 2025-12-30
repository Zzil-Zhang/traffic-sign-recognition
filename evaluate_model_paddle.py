"""
evaluate_model_paddle.py - 模型评估脚本
"""

import paddle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import warnings
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize
import random
import os
import sys
import traceback
import pickle
import pandas as pd
from pathlib import Path

# 忽略警告
warnings.filterwarnings('ignore')

# ==================== 配置常量 ====================
BASELINE_MODEL_PATH = '/home/aistudio/work/fixed_baseline_model/model.pkl'
EVALUATION_RESULTS_DIR = 'evaluation_results'

# ==================== 数据加载函数 ====================
def load_test_data_using_data_utils():
    """
    使用data_utils的create_data_loaders加载测试数据
    确保与训练时使用相同的预处理
    """
    print("使用data_utils加载测试数据...")
    try:
        from data_utils import create_data_loaders, GTSRBDatasetPaddle
        
        batch_size = 32
        train_loader, val_loader, test_loader, data_info = create_data_loaders(
            batch_size=batch_size,
            augment_train=False
        )
        X_train, y_train, X_val, y_val, X_test, y_test = data_info
        print(f"数据加载成功:")
        print(f"  测试集原始数据: {X_test.shape} - {len(y_test)} 样本")

        # 从npy文件重新加载，确保是原始数据
        X_test_raw = np.load('processed_data/X_test.npy')
        y_test_raw = np.load('processed_data/y_test.npy')

        # 与训练一致的预处理
        test_dataset = GTSRBDatasetPaddle(
            X_test_raw,
            y_test_raw,
            is_training=False,
            augment=False
        )

        X_test_processed, y_test_processed = [], []
        print(f"应用数据集预处理...")
        for i in range(len(test_dataset)):
            img, label = test_dataset[i]
            X_test_processed.append(img.numpy())
            y_test_processed.append(label)

        X_test = np.array(X_test_processed)
        y_test = np.array(y_test_processed)

        print(f"数据处理完成:")
        print(f"  处理后形状: {X_test.shape}")
        print(f"  处理后范围: [{X_test.min():.4f}, {X_test.max():.4f}]")
        return X_test, y_test
    except Exception as e:
        print(f"数据加载失败: {e}")
        traceback.print_exc()
        return None, None

def to_hwc_batch(X):
    """将批量图像从 NCHW 转换为 NHWC"""
    if X is None:
        return None
    if X.ndim == 4 and X.shape[1] in (1, 3):  # NCHW
        return np.transpose(X, (0, 2, 3, 1))
    return X

def evaluate_baseline_model_raw(model_path, X_test_hwc, y_test):
    """
    评估基准模型的原始函数
    """
    if not os.path.exists(model_path):
        print(f"基准模型文件不存在: {model_path}")
        return None
    
    try:
        # 导入基准模型相关类
        from baseline_model_fixed import OptimizedSVMClassifier, EnhancedHOGFeatureExtractor
        
        # 自定义Unpickler解决模块导入问题
        class RenameUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module == '__main__' and name == 'EnhancedHOGFeatureExtractor':
                    return EnhancedHOGFeatureExtractor
                if module == '__main__' and name == 'OptimizedSVMClassifier':
                    return OptimizedSVMClassifier
                return super().find_class(module, name)
        
        print("加载基准模型...")
        with open(model_path, 'rb') as f:
            try:
                model_data = pickle.load(f)
            except AttributeError as e:
                f.seek(0)
                print(f"标准pickle反序列化失败，使用重映射Unpickler: {e}")
                model_data = RenameUnpickler(f).load()
        
        # 创建分类器实例
        clf = OptimizedSVMClassifier(hog_extractor=model_data['hog_extractor'])
        clf.svm_model = model_data['svm_model']
        clf.scaler = model_data['scaler']
        clf.training_log = model_data.get('training_log', [])
        
        # 数据预处理
        X_test_hwc = X_test_hwc.astype('float32', copy=False)
        
        # 提取特征
        X_hog = clf.hog_extractor.extract_features(X_test_hwc)
        X_scaled = clf.scaler.transform(X_hog)
        
        # 预测
        y_pred = clf.svm_model.predict(X_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        # 获取决策分数
        scores = None
        if hasattr(clf.svm_model, 'decision_function'):
            scores = clf.svm_model.decision_function(X_scaled)
            if scores.ndim == 1:
                scores = np.stack([-scores, scores], axis=1)
        
        return {
            'accuracy': accuracy,
            'y_pred': y_pred,
            'scores': scores,
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
    except Exception as e:
        print(f"基准模型评估失败: {e}")
        traceback.print_exc()
        return None

class ModelEvaluator:
    """模型评估器：负责计算所有指标"""
    
    def __init__(self):
        self.cnn_metrics = {}
        self.baseline_metrics = {}
        self.cnn_predictions = {}
        self.baseline_predictions = {}
        self.X_test = None
        self.y_test = None
        
    def load_test_data(self):
        """加载测试数据"""
        self.X_test, self.y_test = load_test_data_using_data_utils()
        if self.X_test is None or self.y_test is None:
            raise ValueError("数据加载失败")
        return self.X_test, self.y_test
        
    def load_cnn_model(self, model_path=None):
        """加载CNN模型"""
        if model_path is None or not os.path.exists(model_path):
            model_path = self._find_model_file()
            if model_path is None:
                raise FileNotFoundError("找不到模型文件")
                
        print(f"加载飞桨模型: {model_path}")
        
        # 动态导入模型类
        try:
            from cnn_model_paddle import SimpleCNNPaddle
        except ImportError:
            # 如果导入失败，使用简化版本
            class SimpleCNNPaddle(paddle.nn.Layer):
                def __init__(self, num_classes=43):
                    super().__init__()
                    self.conv1 = paddle.nn.Conv2D(3, 32, 3, padding=1)
                    self.conv2 = paddle.nn.Conv2D(32, 64, 3, padding=1)
                    self.pool = paddle.nn.MaxPool2D(2, 2)
                    self.fc1 = paddle.nn.Linear(64 * 16 * 16, 128)
                    self.fc2 = paddle.nn.Linear(128, num_classes)
                    
                def forward(self, x):
                    x = self.pool(paddle.nn.functional.relu(self.conv1(x)))
                    x = self.pool(paddle.nn.functional.relu(self.conv2(x)))
                    x = paddle.flatten(x, 1)
                    x = paddle.nn.functional.relu(self.fc1(x))
                    x = self.fc2(x)
                    return x
        
        model = SimpleCNNPaddle(num_classes=43)
        model_state_dict = paddle.load(model_path)
        model.set_state_dict(model_state_dict)
        model.eval()
        
        # 记录模型信息
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if not p.stop_gradient)
        
        return {
            'model': model,
            'path': model_path,
            'total_params': total_params,
            'trainable_params': trainable_params
        }
    
    def evaluate_cnn_model(self, model, X_test=None, y_test=None, batch_size=32):
        """评估CNN模型，返回所有指标"""
        if X_test is None:
            X_test = self.X_test
        if y_test is None:
            y_test = self.y_test
            
        if X_test is None or y_test is None:
            raise ValueError("请先加载测试数据")
        
        X_test_tensor = paddle.to_tensor(X_test.astype('float32'))
        y_test_tensor = paddle.to_tensor(y_test.astype('int64'))
        
        total_loss = 0.0
        total_correct = 0
        all_predictions = []
        all_confidences = []
        
        criterion = paddle.nn.CrossEntropyLoss()
        
        with paddle.no_grad():
            for i in range(0, len(X_test_tensor), batch_size):
                batch_x = X_test_tensor[i:i+batch_size]
                batch_y = y_test_tensor[i:i+batch_size]
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                total_loss += loss.numpy().item() * len(batch_x)
                
                preds = paddle.argmax(outputs, axis=1)
                probs = paddle.nn.functional.softmax(outputs, axis=1)
                
                correct = (preds == batch_y).sum().item()
                total_correct += correct
                
                all_predictions.extend(preds.numpy())
                all_confidences.extend(probs.numpy())
        
        # 计算指标
        y_pred = np.array(all_predictions)
        y_probs = np.array(all_confidences)
        
        metrics = self._compute_all_metrics(y_test, y_pred, y_probs)
        metrics['loss'] = total_loss / len(X_test_tensor)
        metrics['accuracy'] = total_correct / len(X_test_tensor)
        
        # 保存预测结果
        self.cnn_predictions = {
            'y_true': y_test,
            'y_pred': y_pred,
            'y_probs': y_probs,
            'all_confidences': all_confidences
        }
        self.cnn_metrics = metrics
        
        return metrics
    
    def evaluate_baseline_model(self, model_path, X_test_hwc=None, y_test=None):
        """评估基准模型，返回所有指标"""
        if X_test_hwc is None:
            X_test_hwc = self.X_test
        if y_test is None:
            y_test = self.y_test
            
        baseline_results = evaluate_baseline_model_raw(model_path, X_test_hwc, y_test)
        
        if baseline_results is None:
            raise ValueError("基准模型评估失败")
            
        # 提取指标
        y_pred = baseline_results.get('y_pred')
        scores = baseline_results.get('scores', None)
        
        metrics = self._compute_all_metrics(y_test, y_pred, scores)
        metrics['accuracy'] = baseline_results['accuracy']
        
        # 保存预测结果
        self.baseline_predictions = {
            'y_true': y_test,
            'y_pred': y_pred,
            'scores': scores
        }
        self.baseline_metrics = metrics
        
        return metrics
    
    def _compute_all_metrics(self, y_true, y_pred, scores=None):
        """计算所有指标"""
        metrics = {}
        
        # 基础分类指标
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        # ROC AUC（如果提供了分数/概率）
        if scores is not None and scores.shape[1] > 1:
            try:
                y_true_onehot = label_binarize(y_true, classes=np.arange(scores.shape[1]))
                metrics['roc_auc_macro'] = roc_auc_score(
                    y_true_onehot, scores, average='macro', multi_class='ovr'
                )
                metrics['roc_auc_weighted'] = roc_auc_score(
                    y_true_onehot, scores, average='weighted', multi_class='ovr'
                )
            except Exception as e:
                print(f"ROC AUC计算失败: {e}")
                metrics['roc_auc_macro'] = None
                metrics['roc_auc_weighted'] = None
        
        # 分类报告
        metrics['classification_report'] = classification_report(
            y_true, y_pred, target_names=[f'Class_{i}' for i in range(43)], 
            output_dict=True, zero_division=0
        )
        
        # 混淆矩阵
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
        
        return metrics
    
    def _find_model_file(self):
        """自动查找模型文件"""
        possible_models = [
            'trained_models/traffic_sign_cnn_paddle_final_20251225_151312.pdparams',
            'trained_models/traffic_sign_cnn_paddle_best_20251225_151312.pdparams',
            'trained_models/my_traffic_classifier_paddle.pdparams',
            'my_traffic_classifier_paddle.pdparams',
            'model_final.pdparams',
            'my_traffic_classifier.pdparams'
        ]
        for model_file in possible_models:
            if os.path.exists(model_file):
                return model_file
        return None

class VisualizationManager:
    """可视化管理器：负责所有可视化任务"""
    
    def __init__(self, output_dir=EVALUATION_RESULTS_DIR):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def plot_confusion_matrix(self, cm, title="混淆矩阵", model_name=""):
        """绘制混淆矩阵"""
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=range(min(20, cm.shape[1])),
            yticklabels=range(min(20, cm.shape[0]))
        )
        plt.title(f'{model_name} - {title} (前20个类别)', fontsize=16)
        plt.xlabel('预测标签', fontsize=14)
        plt.ylabel('真实标签', fontsize=14)
        plt.tight_layout()
        
        filename = f"{model_name.lower()}_confusion_matrix.png" if model_name else "confusion_matrix.png"
        plt.savefig(self.output_dir / filename, dpi=150, bbox_inches='tight')
        plt.show()
        
        # 打印类别准确率
        self._print_class_accuracies(cm)
    
    def plot_prediction_samples(self, model, X_test, y_true, y_pred, confidences=None, 
                               model_name="", num_samples=10):
        """显示预测样本"""
        if len(X_test) == 0:
            print("没有测试数据，无法显示预测样本")
            return
            
        indices = random.sample(range(len(X_test)), num_samples) if len(X_test) > num_samples else range(min(num_samples, len(X_test)))
        
        plt.figure(figsize=(15, 6))
        for i, idx in enumerate(indices):
            plt.subplot(2, 5, i+1)
            img = X_test[idx]
            
            # 处理图像格式
            if len(img.shape) == 3 and img.shape[0] == 3:  # CHW
                img_display = img.transpose(1, 2, 0)
            else:
                img_display = img
            
            # 归一化显示
            img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min() + 1e-8)
            plt.imshow(img_display)
            
            true_label = y_true[idx]
            pred_label = y_pred[idx]
            
            # 获取置信度
            if confidences is not None and idx < len(confidences):
                confidence = confidences[idx][pred_label]
            else:
                confidence = self._get_confidence_from_model(model, img)
            
            color = 'green' if true_label == pred_label else 'red'
            title = f"True: {true_label}\nPred: {pred_label}\nConf: {confidence:.2f}"
            plt.title(title, color=color, fontsize=10)
            plt.axis('off')
        
        plt.suptitle(f'{model_name} - 模型预测样本示例 (绿色=正确, 红色=错误)', fontsize=14)
        plt.tight_layout()
        
        filename = f"{model_name.lower()}_prediction_samples.png" if model_name else "prediction_samples.png"
        plt.savefig(self.output_dir / filename, dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_roc_comparison(self, cnn_scores, baseline_scores, y_true, model_names=("CNN", "基准")):
        """绘制ROC曲线对比"""
        try:
            num_classes = cnn_scores.shape[1]
            y_true_onehot = label_binarize(y_true, classes=np.arange(num_classes))
            
            # CNN微平均ROC
            fpr_cnn, tpr_cnn, _ = roc_curve(y_true_onehot.ravel(), cnn_scores.ravel())
            auc_cnn = auc(fpr_cnn, tpr_cnn)
            
            # 基准微平均ROC
            fpr_base, tpr_base, _ = roc_curve(y_true_onehot.ravel(), baseline_scores.ravel())
            auc_base = auc(fpr_base, tpr_base)
            
            plt.figure(figsize=(10, 8))
            plt.plot(fpr_cnn, tpr_cnn, color='tomato', lw=2, 
                    label=f'{model_names[0]} 微平均ROC (AUC={auc_cnn:.3f})')
            plt.plot(fpr_base, tpr_base, color='steelblue', lw=2, 
                    label=f'{model_names[1]} 微平均ROC (AUC={auc_base:.3f})')
            plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', alpha=0.5)
            
            plt.xlabel('假阳性率 (FPR)', fontsize=12)
            plt.ylabel('真阳性率 (TPR)', fontsize=12)
            plt.title('ROC曲线对比', fontsize=14)
            plt.legend(loc='lower right')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            plt.savefig(self.output_dir / 'roc_comparison.png', dpi=150, bbox_inches='tight')
            plt.show()
            
            return auc_cnn, auc_base
            
        except Exception as e:
            print(f"ROC对比图绘制失败: {e}")
            return None, None
    
    def plot_metrics_comparison(self, cnn_metrics, baseline_metrics):
        """绘制指标对比图"""
        labels = ['准确率', '精确率', '召回率', 'F1 (weighted)', 'F1 (macro)']
        baseline_vals = [
            baseline_metrics.get('accuracy', 0),
            baseline_metrics.get('precision', 0),
            baseline_metrics.get('recall', 0),
            baseline_metrics.get('f1_weighted', 0),
            baseline_metrics.get('f1_macro', 0)
        ]
        cnn_vals = [
            cnn_metrics.get('accuracy', 0),
            cnn_metrics.get('precision', 0),
            cnn_metrics.get('recall', 0),
            cnn_metrics.get('f1_weighted', 0),
            cnn_metrics.get('f1_macro', 0)
        ]
        
        x = np.arange(len(labels))
        width = 0.35
        
        plt.figure(figsize=(12, 6))
        bars1 = plt.bar(x - width/2, baseline_vals, width, label='基准 (SVM+HOG)', color='steelblue')
        bars2 = plt.bar(x + width/2, cnn_vals, width, label='飞桨CNN', color='tomato')
        
        plt.xticks(x, labels, rotation=15)
        plt.ylim(0, 1.05)
        plt.ylabel('指标值', fontsize=12)
        plt.title('模型性能指标对比', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(axis='y', alpha=0.3)
        
        # 添加数值标签
        for bars in (bars1, bars2):
            for bar in bars:
                h = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2, h + 0.02, 
                        f"{h:.3f}", ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'metrics_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def _print_class_accuracies(self, cm, max_classes=20):
        """打印类别准确率"""
        print(f"\n前{max_classes}个类别的准确率:")
        for i in range(min(max_classes, cm.shape[0])):
            if i < cm.shape[0] and i < cm.shape[1]:
                correct = cm[i, i]
                total_class = cm[i, :].sum()
                if total_class > 0:
                    acc = correct / total_class
                    print(f"  类别 {i:2d}: {acc:.2%} ({correct}/{total_class})")
                else:
                    print(f"  类别 {i:2d}: 无测试样本")
    
    def _get_confidence_from_model(self, model, img):
        """从模型获取置信度"""
        try:
            with paddle.no_grad():
                img_tensor = paddle.to_tensor(img[np.newaxis, ...].astype('float32'))
                pred_prob = model(img_tensor).numpy()[0]
                pred_label = np.argmax(pred_prob)
                return pred_prob[pred_label]
        except:
            return 0.0

def generate_comparison_report(cnn_metrics, baseline_metrics, output_dir):
    """生成详细的对比报告"""
    report = {
        '对比时间': pd.Timestamp.now(),
        '对比指标': {}
    }
    
    for metric_name in ['accuracy', 'precision', 'recall', 'f1_weighted', 'f1_macro']:
        if metric_name in cnn_metrics and metric_name in baseline_metrics:
            cnn_val = cnn_metrics[metric_name]
            base_val = baseline_metrics[metric_name]
            improvement = cnn_val - base_val
            improvement_pct = (improvement / base_val * 100) if base_val > 0 else 0
            
            report['对比指标'][metric_name] = {
                '基准模型': base_val,
                '飞桨CNN': cnn_val,
                '绝对提升': improvement,
                '相对提升(%)': improvement_pct
            }
    
    # 保存报告
    report_df = pd.DataFrame(report['对比指标']).T
    report_df.to_csv(output_dir / 'comparison_report.csv', encoding='utf-8-sig')
    
    print(f"\n对比分析:")
    for metric, data in report['对比指标'].items():
        print(f"  {metric}: 基准={data['基准模型']:.4f}, CNN={data['飞桨CNN']:.4f}, "
              f"提升={data['相对提升(%)']:+.1f}%")

def save_detailed_results(evaluator, output_dir):
    """保存详细结果"""
    output_dir = Path(output_dir)
    
    # 保存CNN结果
    if evaluator.cnn_metrics:
        # 创建简化版指标（去除不能序列化的部分）
        cnn_metrics_simple = {k: v for k, v in evaluator.cnn_metrics.items() 
                             if not isinstance(v, (np.ndarray, dict))}
        cnn_df = pd.DataFrame([cnn_metrics_simple])
        cnn_df.to_csv(output_dir / 'cnn_metrics.csv', index=False, encoding='utf-8-sig')
        
        # 保存混淆矩阵
        np.save(output_dir / 'cnn_confusion_matrix.npy', evaluator.cnn_metrics['confusion_matrix'])
    
    # 保存基准结果
    if evaluator.baseline_metrics:
        baseline_metrics_simple = {k: v for k, v in evaluator.baseline_metrics.items() 
                                  if not isinstance(v, (np.ndarray, dict))}
        baseline_df = pd.DataFrame([baseline_metrics_simple])
        baseline_df.to_csv(output_dir / 'baseline_metrics.csv', index=False, encoding='utf-8-sig')
        
        # 保存混淆矩阵
        np.save(output_dir / 'baseline_confusion_matrix.npy', evaluator.baseline_metrics['confusion_matrix'])

def run_complete_evaluation():
    """运行完整的评估流程"""
    print("=" * 60)
    print("模型评估与对比分析")
    print("=" * 60)
    
    # 初始化组件
    evaluator = ModelEvaluator()
    visualizer = VisualizationManager()
    
    try:
        # 1. 加载数据
        print("\n1. 加载测试数据...")
        X_test, y_test = evaluator.load_test_data()
        print(f"   数据形状: {X_test.shape}")
        print(f"   样本数量: {len(y_test)}")
        
        # 2. 评估CNN模型
        print("\n2. 评估飞桨CNN模型...")
        model_info = evaluator.load_cnn_model()
        print(f"   模型信息: {model_info['total_params']:,} 参数")
        
        cnn_metrics = evaluator.evaluate_cnn_model(model_info['model'])
        print(f"   CNN准确率: {cnn_metrics['accuracy']:.4f} ({cnn_metrics['accuracy']*100:.2f}%)")
        print(f"   CNN损失: {cnn_metrics.get('loss', 0):.4f}")
        
        # 3. 评估基准模型
        print("\n3. 评估基准模型 (SVM+HOG)...")
        try:
            X_test_hwc = to_hwc_batch(X_test)
            baseline_metrics = evaluator.evaluate_baseline_model(
                BASELINE_MODEL_PATH, X_test_hwc, y_test
            )
            print(f"   基准准确率: {baseline_metrics['accuracy']:.4f} ({baseline_metrics['accuracy']*100:.2f}%)")
        except Exception as e:
            print(f"   基准模型评估失败: {e}")
            baseline_metrics = None
        
        # 4. 可视化结果
        print("\n4. 生成可视化图表...")
        
        # CNN可视化
        visualizer.plot_confusion_matrix(
            cnn_metrics['confusion_matrix'], 
            model_name="飞桨CNN"
        )
        visualizer.plot_prediction_samples(
            model_info['model'], 
            X_test, 
            evaluator.cnn_predictions['y_true'],
            evaluator.cnn_predictions['y_pred'],
            evaluator.cnn_predictions['all_confidences'],
            model_name="CNN"
        )
        
        # 如果有基准模型，进行对比
        if baseline_metrics is not None:
            # 基准模型可视化
            visualizer.plot_confusion_matrix(
                baseline_metrics['confusion_matrix'], 
                model_name="基准SVM+HOG"
            )
            
            # 指标对比
            visualizer.plot_metrics_comparison(cnn_metrics, baseline_metrics)
            
            # ROC对比
            if (evaluator.cnn_predictions.get('y_probs') is not None and 
                evaluator.baseline_predictions.get('scores') is not None):
                auc_cnn, auc_base = visualizer.plot_roc_comparison(
                    evaluator.cnn_predictions['y_probs'],
                    evaluator.baseline_predictions['scores'],
                    y_test
                )
                if auc_cnn and auc_base:
                    print(f"   CNN ROC AUC: {auc_cnn:.4f}")
                    print(f"   基准 ROC AUC: {auc_base:.4f}")
            
            # 生成对比报告
            generate_comparison_report(cnn_metrics, baseline_metrics, visualizer.output_dir)
        
        # 5. 保存详细结果
        save_detailed_results(evaluator, visualizer.output_dir)
        
        print(f"\n✓ 评估完成！结果保存在: {visualizer.output_dir}")
        print(f"  主要文件:")
        print(f"    - cnn_metrics.csv: CNN模型指标")
        print(f"    - baseline_metrics.csv: 基准模型指标")
        print(f"    - comparison_report.csv: 对比分析报告")
        print(f"    - *.png: 各种可视化图表")
        
    except Exception as e:
        print(f"\n✗ 评估失败: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    # 设置设备
    try:
        paddle.set_device('iluvatar_gpu:0')
        print("使用依图加速卡 (iluvatar_gpu:0)")
    except Exception as e:
        print(f"无法设置依图加速卡设备，使用CPU: {e}")
        paddle.set_device('cpu')
    
    # 设置中文字体
    def setup_chinese_font():
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
        except Exception as e:
            print(f"字体设置失败: {e}")
            matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
            matplotlib.rcParams['axes.unicode_minus'] = False
    
    setup_chinese_font()
    
    # 运行完整评估
    run_complete_evaluation()