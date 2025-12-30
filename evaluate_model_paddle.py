"""
evaluate_model_paddle.py - 飞桨模型评估脚本（使用data_utils统一加载和cnn_model_paddle的模型）
用于评估训练好的飞桨模型性能，确保数据预处理与训练时一致
"""

import paddle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import warnings
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import random
import os
import sys
import traceback

# 忽略警告
warnings.filterwarnings('ignore')

# ==================== 导入必要的模块 ====================
try:
    from data_utils import create_data_loaders, GTSRBDatasetPaddle
    print("成功导入data_utils模块")
except ImportError as e:
    print(f"无法导入data_utils模块: {e}")
    print("请确保data_utils.py在同一目录下")
    sys.exit(1)

# ==================== 导入模型类 ====================
try:
    # 假设模型定义在 cnn_model_paddle.py 中
    from cnn_model_paddle import TrafficCNNPaddle,SimpleCNNPaddle
    print("成功导入TrafficCNNPaddle模型类")
except ImportError as e:
    print(f"无法导入TrafficCNNPaddle: {e}")
    print("请确保cnn_model_paddle.py在同一目录下")
    sys.exit(1)

# ==================== 中文字体设置 ====================
def setup_chinese_font():
    """设置中文字体"""
    try:
        import matplotlib.font_manager as fm

        # 获取系统中已安装的字体
        available_fonts = [f.name for f in fm.fontManager.ttflist]

        # 检查文泉驿中文字体
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

setup_chinese_font()

# ==================== 设备设置 ====================
try:
    paddle.set_device('iluvatar_gpu:0')  # 依图加速卡设备
    print("使用依图加速卡 (iluvatar_gpu:0)")
except Exception as e:
    print(f"无法设置依图加速卡设备，使用CPU: {e}")
    paddle.set_device('cpu')

# ==================== 数据加载函数 ====================
def load_test_data_using_data_utils():
    """
    使用data_utils的create_data_loaders加载测试数据
    确保与训练时使用相同的预处理
    """
    print("使用data_utils加载测试数据...")

    try:
        # 使用与训练时相同的参数加载数据
        batch_size = 32
        train_loader, val_loader, test_loader, data_info = create_data_loaders(
            batch_size=batch_size,
            augment_train=False  # 测试时不使用数据增强
        )

        X_train, y_train, X_val, y_val, X_test, y_test = data_info

        print(f"数据加载成功:")
        print(f"  测试集原始数据: {X_test.shape} - {len(y_test)} 样本")

        # 从npy文件重新加载，确保是原始数据
        X_test_raw = np.load('processed_data/X_test.npy')
        y_test_raw = np.load('processed_data/y_test.npy')

        # 创建测试数据集（使用与训练时相同的预处理）
        test_dataset = GTSRBDatasetPaddle(
            X_test_raw,
            y_test_raw,
            is_training=False,
            augment=False
        )

        # 收集处理后的数据
        X_test_processed = []
        y_test_processed = []

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

# ==================== 模型评估函数 ====================
def evaluate_model(model_path=None):
    """
    评估飞桨模型在测试集上的性能
    使用与训练时完全相同的预处理流程和模型结构
    """
    print("=" * 60)
    print("飞桨模型评估（使用统一数据加载和模型）")
    print("=" * 60)

    # 1. 自动查找模型文件
    if model_path is None or not os.path.exists(model_path):
        print(f"自动查找模型文件...")
        possible_models = [
            'trained_models/traffic_sign_cnn_paddle_final_20251225_151312.pdparams',  # 根据你的时间戳
            'trained_models/traffic_sign_cnn_paddle_best_20251225_151312.pdparams',
            'trained_models/my_traffic_classifier_paddle.pdparams',
            'my_traffic_classifier_paddle.pdparams',
            'model_final.pdparams',
            'my_traffic_classifier.pdparams'
        ]

        for model_file in possible_models:
            if os.path.exists(model_file):
                model_path = model_file
                print(f"找到模型文件: {model_path}")
                break
        else:
            print("找不到任何模型文件")
            print("请先运行 train_cnn_paddle.py 训练模型")
            return None

    print(f"加载飞桨模型: {model_path}")

    # 2. 加载模型（使用cnn_model_paddle中的TrafficCNNPaddle）
    try:
        # 使用导入的TrafficCNNPaddle类
        model = SimpleCNNPaddle(num_classes=43)
        model_state_dict = paddle.load(model_path)
        model.set_state_dict(model_state_dict)
        model.eval()
        print("飞桨模型加载成功")

        # 打印模型信息
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if not p.stop_gradient)
        print(f"  总参数: {total_params:,}")
        print(f"  可训练参数: {trainable_params:,}")

    except Exception as e:
        print(f"模型加载失败: {e}")
        traceback.print_exc()
        return None

    # 3. 加载测试数据（使用data_utils）
    print("\n加载测试数据...")
    X_test, y_test = load_test_data_using_data_utils()

    if X_test is None or y_test is None:
        print("数据加载失败，无法评估")
        return None

    # 4. 转换为paddle tensor
    X_test_tensor = paddle.to_tensor(X_test.astype('float32'))
    y_test_tensor = paddle.to_tensor(y_test.astype('int64'))

    print(f"测试集: {X_test_tensor.shape} - {len(y_test)} 个样本")

    # 5. 评估模型
    print("\n评估模型...")
    criterion = paddle.nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    batch_size = 32
    all_predictions = []
    all_confidences = []

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
    test_loss = total_loss / len(X_test_tensor)
    test_accuracy = total_correct / len(X_test_tensor)
    y_pred = np.array(all_predictions)

    print(f"评估结果:")
    print(f"  测试集损失: {test_loss:.4f}")
    print(f"  测试集准确率: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"  正确样本数: {total_correct}/{len(X_test_tensor)}")

    # 6. 分类报告
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=[f'Class_{i}' for i in range(43)]))

    # 7. 混淆矩阵
    plot_confusion_matrix(y_test, y_pred)

    # 8. 预测样本示例
    plot_prediction_samples(model, X_test, y_test, y_pred, all_confidences)

    return test_accuracy

# ==================== 可视化函数 ====================
def plot_confusion_matrix(y_true, y_pred, max_classes=20):
    """
    绘制混淆矩阵
    """
    # 只显示前max_classes个类别
    mask = y_true < max_classes
    y_true_filtered = y_true[mask]
    y_pred_filtered = y_pred[mask]

    if len(y_true_filtered) == 0:
        print(f"没有前{max_classes}个类别的数据")
        return

    cm = confusion_matrix(y_true_filtered, y_pred_filtered)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(min(max_classes, cm.shape[1])),
                yticklabels=range(min(max_classes, cm.shape[0])))
    plt.title(f'混淆矩阵 (前{max_classes}个类别)', fontsize=16)
    plt.xlabel('预测标签', fontsize=14)
    plt.ylabel('真实标签', fontsize=14)
    plt.tight_layout()

    # 保存图像
    output_dir = 'evaluation_results'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()

    # 显示各类别准确率
    print(f"\n前{max_classes}个类别的准确率:")
    for i in range(min(max_classes, cm.shape[0])):
        total = np.sum(y_true_filtered == i)
        if total > 0:
            correct = cm[i, i] if i < cm.shape[0] and i < cm.shape[1] else 0
            acc = correct / total
            print(f"  类别 {i:2d}: {acc:.2%} ({correct}/{total})")
        else:
            print(f"  类别 {i:2d}: 无测试样本")

def plot_prediction_samples(model, X_test, y_true, y_pred, confidences=None, num_samples=10):
    """
    显示预测样本
    """
    if len(X_test) == 0:
        print("⚠️  没有测试数据，无法显示预测样本")
        return

    # 随机选择样本
    if len(X_test) > num_samples:
        indices = random.sample(range(len(X_test)), num_samples)
    else:
        indices = range(min(num_samples, len(X_test)))

    plt.figure(figsize=(15, 6))

    for i, idx in enumerate(indices):
        plt.subplot(2, 5, i+1)

        # 获取图像并转换为HWC格式显示
        img = X_test[idx]
        if len(img.shape) == 3 and img.shape[0] == 3:  # CHW格式
            img_display = img.transpose(1, 2, 0)
        else:
            img_display = img

        # 归一化显示
        img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min() + 1e-8)
        plt.imshow(img_display)

        # 标签信息
        true_label = y_true[idx]
        pred_label = y_pred[idx]

        # 获取置信度
        if confidences is not None and idx < len(confidences):
            confidence = confidences[idx][pred_label]
        else:
            # 如果没有置信度，重新计算
            with paddle.no_grad():
                img_tensor = paddle.to_tensor(img[np.newaxis, ...].astype('float32'))
                pred_prob = model(img_tensor).numpy()[0]
                confidence = pred_prob[pred_label] if pred_label < len(pred_prob) else 0

        # 设置标题颜色
        color = 'green' if true_label == pred_label else 'red'
        title = f"True: {true_label}\nPred: {pred_label}\nConf: {confidence:.2f}"
        plt.title(title, color=color, fontsize=10)
        plt.axis('off')

    plt.suptitle('模型预测样本示例 (绿色=正确, 红色=错误)', fontsize=14)
    plt.tight_layout()

    # 保存图像
    output_dir = 'evaluation_results'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/prediction_samples.png', dpi=150, bbox_inches='tight')
    plt.show()

# ==================== 与基准对比函数 ====================
def compare_with_baseline(cnn_accuracy):
    """
    与基准模型（SVM+HOG）对比
    """
    print("\n" + "=" * 60)
    print("与基准模型对比")
    print("=" * 60)

    if cnn_accuracy is None:
        print("无法进行对比，飞桨模型评估失败")
        return

    # 基准模型准确率（根据你的训练报告）
    baseline_accuracy = 0.95  # 95%准确率

    print(f"模型性能对比:")
    print(f"  基准模型 (SVM+HOG): {baseline_accuracy:.2%}")
    print(f"  飞桨CNN模型: {cnn_accuracy:.2%}")

    # 计算提升
    improvement = cnn_accuracy - baseline_accuracy
    improvement_percent = (improvement / baseline_accuracy) * 100 if baseline_accuracy > 0 else 0

    print(f"\n性能提升:")
    print(f"  绝对提升: {improvement:.4f} ({improvement:.2%})")
    print(f"  相对提升: {improvement_percent:.1f}%")

    # 可视化对比
    models = ['基准模型 (SVM+HOG)', '飞桨CNN模型']
    accuracies = [baseline_accuracy, cnn_accuracy]

    plt.figure(figsize=(8, 6))
    bars = plt.bar(models, accuracies, color=['skyblue', 'lightcoral'])
    plt.title('模型性能对比', fontsize=16)
    plt.ylabel('准确率', fontsize=14)
    plt.ylim([0, 1.05])

    # 在柱子上显示数值
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.2%}', ha='center', va='bottom', fontsize=12)

    plt.tight_layout()

    # 保存图像
    output_dir = 'evaluation_results'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/model_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

    # 判断是否达到要求
    if cnn_accuracy > baseline_accuracy:
        print(f"飞桨CNN模型超越了基准模型，提升 {improvement_percent:.1f}%")
    else:
        print(f"飞桨CNN模型未超越基准模型，需要进一步优化")

# ==================== 验证一致性函数 ====================
def validate_consistency(cnn_accuracy):
    """
    验证评估结果与训练报告的一致性
    """
    print("\n" + "=" * 60)
    print("一致性验证")
    print("=" * 60)

    if cnn_accuracy is None:
        print(" 无法验证一致性")
        return

    # 训练时报告的准确率（根据你的训练输出）
    training_reported_accuracy = 0.9959  # 99.59%

    print(f"训练时报告的测试准确率: {training_reported_accuracy:.4f} ({training_reported_accuracy*100:.2f}%)")
    print(f"当前评估的测试准确率: {cnn_accuracy:.4f} ({cnn_accuracy*100:.2f}%)")

    diff = abs(cnn_accuracy - training_reported_accuracy)
    threshold = 0.01  # 1%的差异阈值

    if diff < threshold:
        print(f"评估结果与训练时一致！差异: {diff:.4f} (<{threshold})")
    else:
        print(f" 评估结果与训练时不一致，差异: {diff:.4f} (≥{threshold})")
# ==================== 主函数 ====================
def main():
    """主函数"""
    print("开始飞桨模型评估")
    print("=" * 60)
    print("配置:")
    print(f"  数据加载: data_utils.create_data_loaders")
    print(f"  模型类: cnn_model_paddle.TrafficCNNPaddle")
    print("=" * 60)

    # 评估模型
    cnn_accuracy = evaluate_model()

    if cnn_accuracy is None:
        print("\n模型评估失败")
        return

    # 验证一致性
    validate_consistency(cnn_accuracy)

    # 与基准模型对比
    compare_with_baseline(cnn_accuracy)

    print("\n" + "=" * 60)
    print("飞桨模型评估完成！")
    print("=" * 60)
    print("生成的文件保存在 evaluation_results/ 目录:")
    print("  1. confusion_matrix.png - 混淆矩阵")
    print("  2. prediction_samples.png - 预测样本示例")
    print("  3. model_comparison.png - 模型对比图")
    print("=" * 60)

if __name__ == "__main__":
    main()