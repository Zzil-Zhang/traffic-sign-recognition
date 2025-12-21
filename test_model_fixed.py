# test_model_fixed.py - 完整修复版
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import os

print("=" * 60)
print("🚦 德国交通标志识别系统 - 修复版")
print("=" * 60)

# 设置中文字体（避免方框）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 加载模型
model = tf.keras.models.load_model('my_traffic_classifier.keras')
print("✅ 模型加载成功")

# 类别名称（中文）
class_names_cn = [
    '限速20', '限速30', '限速50', '限速60', '限速70', '限速80',
    '限速解除', '限速100', '限速120', '超车禁止', '卡车限速',
    '优先道路', '让行', '停止', '禁止通行', '卡车禁止',
    '禁止驶入', '注意危险', '左急弯', '右急弯', '连续弯路',
    '不平路面', '打滑', '变窄', '施工', '信号灯', '注意行人',
    '注意儿童', '注意自行车', '注意雪/冰', '注意动物',
    '解除限速', '右转', '左转', '直行', '直行或右转',
    '直行或左转', '靠右行驶', '靠左行驶', '环岛',
    '超车解除', '卡车超车解除'
]


def preprocess_image(image_path, target_size=(64, 64)):
    """正确的图像预处理"""
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图像: {image_path}")

    # 转换为RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 保存原始图像用于显示
    original_img = img_rgb.copy()

    # 预处理（与训练时一致）
    # 1. 调整大小
    img_resized = cv2.resize(img_rgb, target_size)

    # 2. 归一化到[0,1]
    img_normalized = img_resized.astype(np.float32) / 255.0

    # 3. 确保形状正确
    if len(img_normalized.shape) == 3:
        img_normalized = np.expand_dims(img_normalized, axis=0)

    return original_img, img_resized, img_normalized


def predict_with_explanation(image_path):
    """带详细解释的预测"""
    print(f"\n📸 正在识别: {os.path.basename(image_path)}")

    try:
        # 预处理
        original_img, resized_img, input_img = preprocess_image(image_path)

        # 预测
        predictions = model.predict(input_img, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]

        # 获取top3预测
        top3_indices = np.argsort(predictions[0])[-3:][::-1]
        top3_confidences = predictions[0][top3_indices]

        print(f"✅ 预测结果: 类别{predicted_class} - {class_names_cn[predicted_class]}")
        print(f"📊 置信度: {confidence:.2%}")

        if confidence < 0.5:
            print("⚠️ 置信度较低！可能原因:")
            print("  1. 图像不是德国交通标志")
            print("  2. 图像质量差或尺寸不对")
            print("  3. 标志不在43个训练类别中")
            print("  4. 图像需要预处理（裁剪、调整大小）")

        print(f"\n🏆 前三名预测:")
        for i, (idx, conf) in enumerate(zip(top3_indices, top3_confidences)):
            print(f"  {i + 1}. 类别{idx}: {class_names_cn[idx]} ({conf:.2%})")

        # === 显示图像 ===
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # 原始图像
        axes[0].imshow(original_img)
        axes[0].set_title('原始图像', fontsize=12)
        axes[0].axis('off')

        # 预处理后的图像（模型看到的）
        axes[1].imshow(resized_img)
        axes[1].set_title('模型输入 (64x64)', fontsize=12)
        axes[1].axis('off')

        # 添加解释
        if original_img.shape[0] > 100 or original_img.shape[1] > 100:
            axes[1].text(32, 70, '训练图像就是64x64\n大图缩小后模糊是正常的',
                         ha='center', va='center', fontsize=10,
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5))

        # 预测结果条形图
        colors = ['#2ecc71', '#f39c12', '#e74c3c']  # 绿、橙、红
        bars = axes[2].barh(range(3), top3_confidences * 100, color=colors)
        axes[2].set_xlabel('置信度 (%)', fontsize=11)
        axes[2].set_yticks(range(3))
        axes[2].set_yticklabels([f'{class_names_cn[idx]}'
                                 for idx in top3_indices], fontsize=10)

        if confidence > 0.8:
            title_color = 'green'
        elif confidence > 0.5:
            title_color = 'orange'
        else:
            title_color = 'red'

        axes[2].set_title(f'预测结果\n最佳: {class_names_cn[predicted_class]}',
                          fontsize=12, color=title_color, fontweight='bold')
        axes[2].set_xlim([0, 100])

        # 在条形上显示数值
        for bar, conf in zip(bars, top3_confidences):
            width = bar.get_width()
            axes[2].text(width + 1, bar.get_y() + bar.get_height() / 2,
                         f'{conf:.1%}', ha='left', va='center', fontsize=9)

        plt.suptitle(f'交通标志识别结果', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('识别结果.png', dpi=150, bbox_inches='tight')
        plt.show()

        return predicted_class, confidence

    except Exception as e:
        print(f"❌ 错误: {e}")
        return None, 0


def test_with_sample_images():
    """使用测试集中的图像测试（真正的交通标志）"""
    print("\n🎯 从测试集随机选择图像测试...")

    try:
        # 加载测试数据
        X_test = np.load('processed_data/X_test.npy')
        y_test = np.load('processed_data/y_test.npy')

        # 随机选择5张
        indices = np.random.choice(len(X_test), 5, replace=False)

        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()

        correct_count = 0

        for i, idx in enumerate(indices):
            img = X_test[idx]
            true_label = y_test[idx]

            # 预测
            predictions = model.predict(img[np.newaxis, ...], verbose=0)
            pred_label = np.argmax(predictions[0])
            confidence = predictions[0][pred_label]

            # 显示图像
            axes[i].imshow(img)

            # 判断是否正确
            correct = pred_label == true_label
            color = 'green' if correct else 'red'

            if correct:
                correct_count += 1

            title = f'真: {class_names_cn[true_label]}\n'
            title += f'预测: {class_names_cn[pred_label]}\n'
            title += f'置信度: {confidence:.1%}'

            axes[i].set_title(title, color=color, fontsize=9)
            axes[i].axis('off')

            # 在图像上显示对错符号
            symbol = '✅' if correct else '❌'
            axes[i].text(5, 15, symbol, fontsize=12, color=color,
                         bbox=dict(boxstyle="circle,pad=0.3", facecolor="white", alpha=0.8))

        # 隐藏多余的子图
        for i in range(len(indices), len(axes)):
            axes[i].axis('off')

        accuracy = correct_count / len(indices)
        plt.suptitle(f'测试集随机样本 ({correct_count}/{len(indices)} 正确, 准确率: {accuracy:.1%})',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('测试集识别.png', dpi=150, bbox_inches='tight')
        plt.show()

        print(f"\n📊 随机测试结果: {correct_count}/{len(indices)} 正确 ({accuracy:.1%})")

    except Exception as e:
        print(f"❌ 测试错误: {e}")


def batch_test():
    """批量测试整个测试集"""
    print("\n📊 批量测试整个测试集...")

    try:
        X_test = np.load('processed_data/X_test.npy')
        y_test = np.load('processed_data/y_test.npy')
        y_test_onehot = tf.keras.utils.to_categorical(y_test, 43)

        # 评估
        loss, accuracy = model.evaluate(X_test, y_test_onehot, verbose=0)
        print(f"✅ 测试集准确率: {accuracy:.4f} ({accuracy:.2%})")
        print(f"📉 测试集损失: {loss:.4f}")

        # 预测所有样本
        predictions = model.predict(X_test, verbose=0)
        pred_labels = np.argmax(predictions, axis=1)

        # 计算每个类别的准确率
        class_correct = np.zeros(43)
        class_total = np.zeros(43)

        for i in range(len(y_test)):
            class_total[y_test[i]] += 1
            if pred_labels[i] == y_test[i]:
                class_correct[y_test[i]] += 1

        print("\n📈 各类别准确率 (前10个):")
        for i in range(min(10, 43)):
            if class_total[i] > 0:
                acc = class_correct[i] / class_total[i]
                stars = '★' * int(acc * 5) + '☆' * (5 - int(acc * 5))
                print(f"  类别{i:2d} {class_names_cn[i]:8s}: {stars} {acc:.1%}")

        # 显示混淆矩阵（简版）
        print("\n🔍 常见混淆情况:")
        confusion_pairs = []
        for i in range(len(y_test)):
            if pred_labels[i] != y_test[i]:
                confusion_pairs.append((y_test[i], pred_labels[i]))

        if confusion_pairs:
            from collections import Counter
            top_confusions = Counter(confusion_pairs).most_common(3)
            for (true, pred), count in top_confusions:
                print(f"  {class_names_cn[true]} → {class_names_cn[pred]}: {count}次")

        return accuracy

    except Exception as e:
        print(f"❌ 批量测试错误: {e}")
        return 0


def test_valid_image_folder():
    """测试valid_image文件夹"""
    valid_dir = 'valid_image'

    if not os.path.exists(valid_dir):
        print(f"📂 文件夹不存在: {valid_dir}/")
        print("💡 提示: 创建 valid_image/ 文件夹，放入要测试的图像")
        os.makedirs(valid_dir, exist_ok=True)
        print(f"✅ 已创建 {valid_dir}/ 文件夹")
        return

    images = [f for f in os.listdir(valid_dir)
              if f.endswith(('.png', '.jpg', '.jpeg', '.ppm', '.bmp'))]

    if not images:
        print(f"📭 {valid_dir}/ 文件夹是空的")
        print("💡 请放入要测试的交通标志图像")
        return

    print(f"\n📁 在 {valid_dir}/ 找到 {len(images)} 张图像:")

    results = []
    for img_file in images[:5]:  # 只测试前5张
        img_path = os.path.join(valid_dir, img_file)
        print(f"\n{'=' * 40}")
        print(f"测试: {img_file}")

        try:
            pred_class, confidence = predict_with_explanation(img_path)
            if pred_class is not None:
                results.append((img_file, pred_class, confidence, confidence > 0.5))
        except Exception as e:
            print(f"❌ 测试失败: {e}")

    if results:
        print(f"\n{'=' * 40}")
        print("📋 测试总结:")
        correct = sum(1 for _, _, conf, correct in results if correct)
        total = len(results)
        print(f"✅ 高置信度识别: {correct}/{total}")

        for img_file, pred_class, confidence, is_correct in results:
            status = "✅" if is_correct else "⚠️"
            print(f"  {status} {img_file}: {class_names_cn[pred_class]} ({confidence:.1%})")


def main():
    """主函数 - 中文版"""
    print("🎯 模型信息:")
    print(f"  测试准确率: 99.74%")
    print(f"  可识别: 43种德国交通标志")
    print(f"  训练样本: 39209张图像")

    while True:
        print("\n" + "=" * 60)
        print("🚦 德国交通标志识别系统")
        print("=" * 60)
        print("请选择操作:")
        print("1. 🖼️  识别单张图像")
        print("2. 🎯  测试真正的交通标志（从测试集）")
        print("3. 📁  测试valid_image文件夹")
        print("4. 📊  批量测试整个测试集")
        print("5. ℹ️  查看模型信息")
        print("6. 🚪  退出")
        print("=" * 60)

        choice = input("请输入选项 (1-6): ").strip()

        if choice == '1':
            img_path = input("请输入图像路径 (直接回车测试00065.png): ").strip()
            if not img_path:
                if os.path.exists('00065.png'):
                    img_path = '00065.png'
                    print(f"使用默认图像: {img_path}")
                else:
                    print("❌ 00065.png不存在，请手动输入路径")
                    continue

            if os.path.exists(img_path):
                predict_with_explanation(img_path)
            else:
                print(f"❌ 文件不存在: {img_path}")

        elif choice == '2':
            test_with_sample_images()

        elif choice == '3':
            test_valid_image_folder()

        elif choice == '4':
            batch_test()

        elif choice == '5':
            print("\nℹ️ 模型详细信息:")
            print(f"  输入形状: {model.input_shape}")
            print(f"  输出形状: {model.output_shape}")
            print(f"  总参数: {model.count_params():,}")
            print(f"  层数: {len(model.layers)}")
            print(f"  训练准确率: 99.79%")
            print(f"  验证准确率: 99.73%")
            print(f"  测试准确率: 99.74%")
            print(f"  训练样本: 27446张")
            print(f"  验证样本: 5881张")
            print(f"  测试样本: 5882张")

        elif choice == '6':
            print("\n👋 谢谢使用！")
            print("💾 生成的文件:")
            print("  - 识别结果.png (单张识别结果)")
            print("  - 测试集识别.png (随机测试结果)")
            print("  - my_traffic_classifier.keras (训练好的模型)")
            break

        else:
            print("❌ 无效选项，请重试")


# 添加这个，确保能直接运行
if __name__ == "__main__":
    # 检查TensorFlow警告，但继续运行
    import warnings

    warnings.filterwarnings('ignore', category=UserWarning)

    main()