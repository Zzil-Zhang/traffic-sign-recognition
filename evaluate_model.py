"""
evaluate_model.py - 模型评估脚本
用于评估训练好的模型性能
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
from keras.models import load_model
from keras.utils import to_categorical

from data_preprocessing import GTSRBDataLoader

def evaluate_model(model_path='my_traffic_classifier.keras'):
    """
    评估模型在测试集上的性能
    """
    print("=" * 60)
    print("模型评估")
    print("=" * 60)
    
    # 1. 加载模型
    print(f"加载模型: {model_path}")
    try:
        model = load_model(model_path)
        print("✓ 模型加载成功")
        model.summary()
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        return
    
    # 2. 加载测试数据
    print("\n加载测试数据...")
    loader = GTSRBDataLoader(data_root='data', image_size=(64, 64), normalize='minmax')
    X_train, X_val, X_test, y_train, y_val, y_test = loader.load_processed_data('processed_data')
    y_test_onehot = to_categorical(y_test, 43)
    
    print(f"测试集: {X_test.shape} - {len(y_test)} 个样本")
    
    # 3. 评估模型
    print("\n评估模型...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test_onehot, verbose=0)
    print(f"测试集损失: {test_loss:.4f}")
    print(f"测试集准确率: {test_accuracy:.4f} ({test_accuracy:.2%})")
    
    # 4. 预测
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    # 5. 分类报告
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=[f'Class_{i}' for i in range(43)]))
    
    # 6. 混淆矩阵（只显示前20类，避免太大）
    plot_confusion_matrix(y_test, y_pred, max_classes=20)
    
    # 7. 预测样本示例
    plot_prediction_samples(model, X_test, y_test, y_pred, num_samples=10)
    
    return test_accuracy

def plot_confusion_matrix(y_true, y_pred, max_classes=20):
    """
    绘制混淆矩阵（只显示前max_classes个类别）
    """
    # 只显示前max_classes个类别
    mask = y_true < max_classes
    y_true_filtered = y_true[mask]
    y_pred_filtered = y_pred[mask]
    
    if len(y_true_filtered) == 0:
        print("没有足够的数据绘制混淆矩阵")
        return
    
    cm = confusion_matrix(y_true_filtered, y_pred_filtered)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(max_classes),
                yticklabels=range(max_classes))
    plt.title(f'混淆矩阵 (前{max_classes}个类别)', fontsize=16, fontweight='bold')
    plt.xlabel('预测标签', fontsize=14)
    plt.ylabel('真实标签', fontsize=14)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 计算每个类别的准确率
    print(f"\n前{max_classes}个类别的准确率:")
    for i in range(max_classes):
        # 检查该类别是否有样本
        total_samples = np.sum(y_true_filtered == i)
        if total_samples > 0:
            correct_predictions = cm[i, i] if i < cm.shape[0] and i < cm.shape[1] else 0
            accuracy = correct_predictions / total_samples
            print(f"  类别 {i:2d}: {accuracy:.2%} ({correct_predictions}/{total_samples})")
        else:
            print(f"  类别 {i:2d}: 无测试样本")

def plot_prediction_samples(model, X_test, y_true, y_pred, num_samples=10):
    """
    显示一些预测样本
    """
    import random
    
    # 随机选择一些样本
    indices = random.sample(range(len(X_test)), min(num_samples, len(X_test)))
    
    plt.figure(figsize=(15, 6))
    for i, idx in enumerate(indices):
        plt.subplot(2, 5, i+1)
        
        # 显示图像
        img = X_test[idx]
        plt.imshow(img)
        
        # 获取真实标签和预测标签
        true_label = y_true[idx]
        pred_label = y_pred[idx]
        
        # 获取预测概率
        pred_prob = model.predict(img[np.newaxis, ...], verbose=0)[0]
        confidence = pred_prob[pred_label]
        
        # 设置标题颜色（绿色正确，红色错误）
        color = 'green' if true_label == pred_label else 'red'
        title = f"True: {true_label}\nPred: {pred_label}\nConf: {confidence:.2f}"
        plt.title(title, color=color, fontsize=10)
        plt.axis('off')
    
    plt.suptitle('预测样本示例 (绿色=正确, 红色=错误)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('prediction_samples.png', dpi=150, bbox_inches='tight')
    plt.show(block=False)  # 非阻塞显示
    plt.pause(2)  # 显示2秒

def compare_with_baseline(cnn_accuracy):
    """
    与基准模型（SVM+HOG）对比
    """
    print("\n" + "=" * 60)
    print("与基准模型对比")
    print("=" * 60)
    
    # 基准模型准确率（假设成员A的结果）
    # 你可以修改这个值，根据成员A的实际结果
    baseline_accuracy = 0.70  # 假设基准模型70%准确率
    
    print(f"基准模型 (SVM+HOG) 准确率: {baseline_accuracy:.2%}")
    print(f"深度CNN模型准确率: {cnn_accuracy:.2%}")
    
    # 计算提升
    improvement = cnn_accuracy - baseline_accuracy
    improvement_percent = (improvement / baseline_accuracy) * 100
    
    print(f"\n性能提升:")
    print(f"  绝对提升: {improvement:.4f} ({improvement:.2%})")
    print(f"  相对提升: {improvement_percent:.1f}%")
    
    # 可视化对比
    models = ['基准模型 (SVM+HOG)', '深度CNN模型']
    accuracies = [baseline_accuracy, cnn_accuracy]
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(models, accuracies, color=['skyblue', 'lightcoral'])
    plt.title('模型性能对比', fontsize=16, fontweight='bold')
    plt.ylabel('准确率', fontsize=14)
    plt.ylim([0, 1])
    
    # 在柱子上显示数值
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.2%}', ha='center', va='bottom', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
    plt.show(block=False)  # 非阻塞显示
    plt.pause(2)  # 显示2秒
    
    # 判断是否达到要求
    if cnn_accuracy > baseline_accuracy:
        print(f"✓ CNN模型超越了基准模型，提升 {improvement_percent:.1f}%")
    else:
        print(f"⚠️ CNN模型未超越基准模型，需要进一步优化")

if __name__ == "__main__":
    # 你可以指定模型路径，默认为 my_traffic_classifier.keras
    model_to_evaluate = 'my_traffic_classifier.keras'
    
    # 检查模型文件是否存在
    import os
    if not os.path.exists(model_to_evaluate):
        print(f"模型文件 {model_to_evaluate} 不存在")
        print("请先运行 train_cnn.py 训练模型")
        exit(1)
    
    # 评估模型
    cnn_accuracy = evaluate_model(model_to_evaluate)
    
    # 如果评估失败（返回None），就不要再做基线对比，避免格式化错误
    if cnn_accuracy is None:
        print("评估失败，无法与基线模型进行对比。")
        exit(1)
    
    # 与基准模型对比
    compare_with_baseline(cnn_accuracy)
    
    print("\n" + "=" * 60)
    print("评估完成！")
    print("生成的文件:")
    print("  1. confusion_matrix.png - 混淆矩阵")
    print("  2. prediction_samples.png - 预测样本示例")
    print("  3. model_comparison.png - 模型对比图")
    print("=" * 60)