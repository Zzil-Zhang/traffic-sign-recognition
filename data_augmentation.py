"""
高级数据增强策略实现
包含几何变换、模拟驾驶场景增强、以及Albumentations集成
"""

import numpy as np
import cv2
# 条件导入albumentations，避免网络连接问题
try:
    import albumentations as A
    ALBUMENTATIONS_AVAILABLE = True
except (ImportError, Exception) as e:
    print(f"警告: albumentations不可用 ({e})")
    ALBUMENTATIONS_AVAILABLE = False
    A = None
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import random
from pathlib import Path
import pandas as pd

class AdvancedDataAugmentation:
    """高级数据增强类，包含多种增强策略"""

    @staticmethod
    def visualize_augmentations(original_image, augmented_images, titles, save_path='augmentation_samples.png'):
        """可视化增强效果对比"""
        # 获取原图尺寸信息
        orig_h, orig_w = original_image.shape[:2]

        # 根据图像数量调整布局
        n_aug = len(augmented_images)
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))  # 2行5列

        # 显示原图（第一行第一个）
        axes[0, 0].imshow(original_image)
        axes[0, 0].set_title(f'Original\n{orig_w}x{orig_h}')
        axes[0, 0].axis('off')

        # 显示增强后的图像
        for i, (img, title) in enumerate(zip(augmented_images, titles), 1):
            row = (i-1) // 4  # 每行4个增强图像
            col = (i-1) % 4 + 1  # 从第一行第二列开始

            # 如果是小图像，使用最近邻插值放大显示以保持锐利
            if img.shape[0] < 128:
                display_img = cv2.resize(img, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            else:
                display_img = img

            axes[row, col].imshow(display_img)
            axes[row, col].set_title(f'{title}\n{img.shape[1]}x{img.shape[0]}')
            axes[row, col].axis('off')

        # 隐藏多余的子图
        total_cells = 10  # 2行×5列
        for i in range(len(augmented_images) + 1, total_cells):
            row = i // 5
            col = i % 5
            axes[row, col].axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✓ 增强样本对比图已保存: {save_path}")

    # ========== 几何增强 ==========
    @staticmethod
    def random_rotation(image, angle_range=(-20, 20)):
        """随机旋转"""
        angle = random.uniform(angle_range[0], angle_range[1])
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
        return rotated

    @staticmethod
    def random_translation(image, translate_range=(-0.1, 0.1)):
        """随机平移"""
        h, w = image.shape[:2]
        tx = random.uniform(translate_range[0], translate_range[1]) * w
        ty = random.uniform(translate_range[0], translate_range[1]) * h

        M = np.float32([[1, 0, tx], [0, 1, ty]])
        translated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
        return translated

    @staticmethod
    def random_scale(image, scale_range=(0.8, 1.2)):
        """随机缩放"""
        h, w = image.shape[:2]
        scale = random.uniform(scale_range[0], scale_range[1])

        # 计算缩放后的尺寸
        new_w, new_h = int(w * scale), int(h * scale)

        # 使用高质量插值调整图像大小
        scaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        # 如果缩放后尺寸不同，进行填充或裁剪
        if scale > 1:
            # 裁剪中心部分
            start_x = (new_w - w) // 2
            start_y = (new_h - h) // 2
            scaled = scaled[start_y:start_y + h, start_x:start_x + w]
        else:
            # 填充边界
            top = (h - new_h) // 2
            bottom = h - new_h - top
            left = (w - new_w) // 2
            right = w - new_w - left
            scaled = cv2.copyMakeBorder(scaled, top, bottom, left, right,
                                        cv2.BORDER_REFLECT_101)

        return scaled

    @staticmethod
    def random_shear(image, shear_range=(-0.2, 0.2)):
        """随机错切"""
        shear = random.uniform(shear_range[0], shear_range[1])
        h, w = image.shape[:2]

        M = np.float32([[1, shear, 0], [0, 1, 0]])
        sheared = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
        return sheared

    # ========== 驾驶场景增强 ==========
    @staticmethod
    def motion_blur(image, kernel_size=15):
        """运动模糊 - 模拟车辆移动"""
        # 对于小图像，减小核大小
        h, w = image.shape[:2]
        if min(h, w) < 64:
            kernel_size = min(kernel_size, 7)

        # 创建运动模糊核
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
        kernel = kernel / kernel_size

        blurred = cv2.filter2D(image, -1, kernel)
        return blurred

    @staticmethod
    def rain_effect(image, drop_count=30, drop_length=8, drop_width=1):
        """雨滴噪声 - 模拟雨天"""
        h, w, c = image.shape

        # 根据图像大小调整雨滴参数
        if min(h, w) < 64:
            drop_count = max(10, drop_count // 2)
            drop_length = max(3, drop_length // 2)

        rainy = image.copy()

        for _ in range(drop_count):
            # 随机生成雨滴位置
            x = random.randint(0, w - 1)
            y_start = random.randint(0, h // 2)  # 雨滴从上半部分开始

            # 绘制雨滴（白色线条）
            for i in range(drop_length):
                y = min(y_start + i, h - 1)
                if 0 <= x < w and 0 <= y < h:
                    rainy[y, x] = [255, 255, 255]

        # 混合原图和雨滴
        alpha = 0.7
        result = cv2.addWeighted(image, alpha, rainy, 1 - alpha, 0)
        return result

    @staticmethod
    def adjust_brightness_contrast(image, brightness=30, contrast=40):
        """调整亮度和对比度 - 模拟不同光照条件"""
        # 随机调整参数
        b = random.randint(-brightness, brightness)
        c = random.randint(-contrast, contrast)

        alpha = 1 + c / 100.0
        beta = b

        adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        return adjusted

    @staticmethod
    def shadow_effect(image, num_shadows=2):
        """阴影效果 - 模拟树木/建筑物遮挡"""
        h, w, c = image.shape

        min_dim = min(w, h)
        # 确保最小半径不超过最大半径
        min_radius = max(5, min_dim // 20)  # 至少5像素
        max_radius = max(min_radius + 1, min_dim // 4)  # 确保max > min

        # 如果图像太小，减少阴影数量
        if min_dim < 64:
            num_shadows = 1

        shadowed = image.copy().astype(np.float32)

        for _ in range(num_shadows):
            # 随机阴影位置和大小
            x = random.randint(0, w - 1)
            y = random.randint(0, h - 1)
            radius = random.randint(min_radius, max_radius)

            # 创建阴影遮罩（高斯模糊效果）
            mask = np.zeros((h, w), dtype=np.float32)

            # 使用高斯分布创建柔和的阴影
            for i in range(h):
                for j in range(w):
                    distance = np.sqrt((i - y)**2 + (j - x)**2)
                    if distance < radius:
                        mask[i, j] = 0.5 * (1 - distance / radius)  # 中心最暗，边缘渐变

            # 应用阴影
            for channel in range(c):
                shadowed[:, :, channel] = shadowed[:, :, channel] * (1 - mask * 0.7)

        return np.clip(shadowed, 0, 255).astype(np.uint8)

    # ========== 组合增强 ==========
    @staticmethod
    def apply_random_geometric_aug(image, prob=0.7):
        """应用随机几何增强"""
        if random.random() > prob:
            return image

        aug_methods = [
            ('rotate', lambda: AdvancedDataAugmentation.random_rotation(image)),
            ('translate', lambda: AdvancedDataAugmentation.random_translation(image)),
            ('scale', lambda: AdvancedDataAugmentation.random_scale(image)),
            ('shear', lambda: AdvancedDataAugmentation.random_shear(image))
        ]

        # 随机选择1-2种增强方法
        selected_methods = random.sample(aug_methods, k=random.randint(1, 2))

        result = image.copy()
        for name, method in selected_methods:
            result = method()

        return result

    @staticmethod
    def apply_random_scene_aug(image, prob=0.5):
        """应用随机场景增强"""
        if random.random() > prob:
            return image

        aug_methods = [
            ('motion_blur', lambda: AdvancedDataAugmentation.motion_blur(image)),
            ('brightness_contrast', lambda: AdvancedDataAugmentation.adjust_brightness_contrast(image)),
            ('shadow', lambda: AdvancedDataAugmentation.shadow_effect(image))
        ]

        # 只选一种场景增强
        selected_method = random.choice(aug_methods)
        result = selected_method[1]()

        # 50%概率添加雨滴效果
        if random.random() < 0.5:
            result = AdvancedDataAugmentation.rain_effect(result)

        return result

    @staticmethod
    def apply_combined_augmentation(image, geometric_prob=0.7, scene_prob=0.4):
        """应用组合增强（几何+场景）"""
        result = image.copy()

        # 几何增强
        result = AdvancedDataAugmentation.apply_random_geometric_aug(result, geometric_prob)

        # 场景增强
        result = AdvancedDataAugmentation.apply_random_scene_aug(result, scene_prob)

        return result

    # ========== Albumentations 增强 ==========
    @staticmethod
    def get_albumentations_transform(use_hard=False):
        """获取Albumentations增强管道"""
        if not ALBUMENTATIONS_AVAILABLE:
            print("警告: albumentations不可用，返回None")
            return None

        if use_hard:
            # 强增强
            transform = A.Compose([
                A.Rotate(limit=20, p=0.7),
                A.Transpose(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
                A.MotionBlur(blur_limit=7, p=0.3),
                A.OpticalDistortion(distort_limit=0.2, shift_limit=0.2, p=0.3),
                A.RandomRain(p=0.2),
                A.RandomShadow(p=0.2),
                A.CoarseDropout(max_holes=8, max_height=8, max_width=8, p=0.3),
                # 移除Normalize，避免数据类型问题
            ])
        else:
            # 中等增强
            transform = A.Compose([
                A.Rotate(limit=15, p=0.5),
                A.HorizontalFlip(p=0.3),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
                # 移除Normalize，避免数据类型问题
            ])

        return transform

    @staticmethod
    def apply_albumentations_aug(image, transform):
        """应用Albumentations增强"""
        if transform is None:
            return image

        augmented = transform(image=image)['image']
        return augmented


class PyTorchStyleAugmentation:
    """PyTorch风格的增强方法"""

    @staticmethod
    def get_train_transforms(image_size=64, augment=True):
        """获取训练时的变换管道"""
        if augment:
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        return transform

    @staticmethod
    def get_val_transforms(image_size=64):
        """获取验证/测试时的变换管道"""
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        return transform

    @staticmethod
    def apply_transform(image, transform):
        """应用单个变换"""
        return transform(image)

    @staticmethod
    def apply_transforms_to_dataset(images, labels, transform, augment=True):
        """对整个数据集应用变换"""
        transformed_images = []

        for i, img in enumerate(images):
            if augment or i < 5:  # 验证时只变换前几个示例
                transformed_img = transform(img)
            else:
                # 对于验证集，使用相同的变换但不应用增强
                val_transform = PyTorchStyleAugmentation.get_val_transforms(img.shape[0])
                transformed_img = val_transform(img)
            transformed_images.append(transformed_img.numpy())

        return np.array(transformed_images), labels
def test_augmentations():
    """测试增强效果 - 改进版本，修复小图像问题"""
    # 方案1: 使用指定路径的图像
    img_path = Path(r'00065.png')

    if img_path.exists():
        print(f"✓ 使用指定图像: {img_path}")
        # 读取原始图像
        original_img = cv2.imread(str(img_path))
        if original_img is None:
            print(f"错误: 无法读取图像 {img_path}")
            return

        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

        # 显示原始图像信息
        orig_h, orig_w = original_img.shape[:2]
        print(f"  原始尺寸: {orig_w}x{orig_h}")
        print(f"  数据类型: {original_img.dtype}")
        print(f"  数值范围: [{original_img.min()}, {original_img.max()}]")

        # 创建用于增强的64x64版本（使用高质量插值）
        img_for_aug = cv2.resize(original_img, (64, 64), interpolation=cv2.INTER_LANCZOS4)

        # 为了显示清晰，可以适当放大原图但不超过合理大小
        display_size = (256, 256) if orig_w > 256 else (orig_w, orig_h)
        display_original = cv2.resize(original_img, display_size, interpolation=cv2.INTER_LANCZOS4)

    else:
        print(f"⚠️ 指定图像不存在: {img_path}")
        print("使用测试图案...")
        # 创建测试图案
        display_original = create_test_traffic_sign()
        img_for_aug = cv2.resize(display_original, (64, 64), interpolation=cv2.INTER_LANCZOS4)

    # 测试各种增强
    aug = AdvancedDataAugmentation()

    augmented_images = []
    titles = []

    print("\n应用增强方法...")

    # 几何增强
    print("  几何增强...")
    augmented_images.append(aug.random_rotation(img_for_aug))
    titles.append('Rotation')

    augmented_images.append(aug.random_translation(img_for_aug))
    titles.append('Translation')

    augmented_images.append(aug.random_scale(img_for_aug))
    titles.append('Scale')

    augmented_images.append(aug.random_shear(img_for_aug))
    titles.append('Shear')

    # 场景增强
    print("  场景增强...")
    augmented_images.append(aug.motion_blur(img_for_aug))
    titles.append('Motion Blur')

    augmented_images.append(aug.rain_effect(img_for_aug))
    titles.append('Rain Effect')

    augmented_images.append(aug.adjust_brightness_contrast(img_for_aug))
    titles.append('Brightness/Contrast')

    augmented_images.append(aug.shadow_effect(img_for_aug))
    titles.append('Shadow Effect')

    # 组合增强
    print("  组合增强...")
    augmented_images.append(aug.apply_combined_augmentation(img_for_aug))
    titles.append('Combined Aug')

    # Albumentations增强（可选）
    if ALBUMENTATIONS_AVAILABLE:
        try:
            print("  尝试Albumentations增强...")
            transform = aug.get_albumentations_transform(use_hard=False)
            if transform:
                albu_img = aug.apply_albumentations_aug(img_for_aug, transform)
                augmented_images.append(albu_img)
                titles.append('Albumentations')
        except Exception as e:
            print(f"  Albumentations失败: {e}")

    # 验证增强结果
    print(f"\n验证增强结果:")
    for i, (title, aug_img) in enumerate(zip(titles, augmented_images)):
        print(f"  {title:20} - 尺寸: {aug_img.shape[1]}x{aug_img.shape[0]}, "
              f"数据类型: {aug_img.dtype}, 范围: [{aug_img.min()}, {aug_img.max()}]")

    # 可视化 - 使用改进的显示方法
    print(f"\n生成可视化对比图...")
    try:
        aug.visualize_augmentations(display_original, augmented_images, titles,
                                    save_path='augmentation_details_20251225_103725/data_augmentation_samples.png')
    except Exception as e:
        print(f"可视化失败: {e}")
        # 尝试简化版可视化
        simple_visualization(display_original, augmented_images, titles)

    # 额外保存增强前后的详细对比
    save_detailed_comparison(display_original, img_for_aug, augmented_images, titles)


def simple_visualization(original_image, augmented_images, titles):
    """简化版可视化"""
    print("使用简化版可视化...")

    # 显示原图
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(original_image)
    plt.title('Original')
    plt.axis('off')

    # 显示两个增强示例
    for i in range(min(2, len(augmented_images))):
        plt.subplot(1, 3, i+2)
        plt.imshow(augmented_images[i])
        plt.title(titles[i])
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('simple_augmentation_samples.png', dpi=150)
    plt.show()
    print("✓ 简化对比图已保存")


def save_detailed_comparison(original_display, original_small, augmented_images, titles):
    """保存详细的增强前后对比"""
    import os
    from datetime import datetime

    # 创建保存目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(f"augmentation_details_{timestamp}")
    save_dir.mkdir(exist_ok=True)

    print(f"\n保存详细对比到目录: {save_dir}")

    try:
        # 保存原图
        cv2.imwrite(str(save_dir / "01_original_high_quality.png"),
                    cv2.cvtColor(original_display, cv2.COLOR_RGB2BGR))

        cv2.imwrite(str(save_dir / "02_original_64x64.png"),
                    cv2.cvtColor(original_small, cv2.COLOR_RGB2BGR))

        # 保存每个增强结果
        for i, (img, title) in enumerate(zip(augmented_images, titles), 1):
            filename = f"{i+2:02d}_{title.replace(' ', '_').replace('/', '_')}.png"
            save_path = save_dir / filename

            # 确保图像格式正确
            if img.dtype != np.uint8:
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                else:
                    img = np.clip(img, 0, 255).astype(np.uint8)

            # 如果是3通道图像
            if len(img.shape) == 3 and img.shape[2] == 3:
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(save_path), img_bgr)
            else:
                cv2.imwrite(str(save_path), img)

            print(f"  保存: {filename}")

        # 创建对比说明文件
        with open(save_dir / "README.txt", "w") as f:
            f.write("数据增强结果对比\n")
            f.write("=" * 50 + "\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"原图尺寸: {original_display.shape[1]}x{original_display.shape[0]}\n")
            f.write(f"增强图尺寸: 64x64\n\n")
            f.write("文件说明:\n")
            f.write("  01_original_high_quality.png - 高质量原图\n")
            f.write("  02_original_64x64.png - 64x64原图（用于增强）\n")
            for i, title in enumerate(titles, 1):
                f.write(f"  {i+2:02d}_{title.replace(' ', '_')}.png - {title}增强\n")

        print(f"✓ 详细对比已保存到: {save_dir}")
    except Exception as e:
        print(f"保存详细对比失败: {e}")


def create_test_traffic_sign():
    """创建测试用交通标志图案"""
    img = np.zeros((256, 256, 3), dtype=np.uint8)

    # 画一个标准的停止标志（红色八角形）
    center = (128, 128)
    radius = 80

    # 八角形顶点
    pts = []
    for i in range(8):
        angle = i * 45 * np.pi / 180
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        pts.append([int(x), int(y)])

    # 填充八角形
    cv2.fillPoly(img, [np.array(pts)], (0, 0, 255))

    # 添加白色边框
    cv2.polylines(img, [np.array(pts)], True, (255, 255, 255), 3)

    # 添加"STOP"文字
    cv2.putText(img, "STOP", (80, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

    return img


if __name__ == '__main__':
    print("=" * 60)
    print("高级数据增强策略测试")
    print("=" * 60)

    test_augmentations()

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)