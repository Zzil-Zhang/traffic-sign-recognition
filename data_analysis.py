"""
GTSRB数据集分析脚本
分析数据集结构、类别分布、图像尺寸统计，并绘制类别分布条形图
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def analyze_dataset():
    """分析GTSRB数据集"""
    print("=" * 60)
    print("GTSRB数据集分析")
    print("=" * 60)
    
    # 读取训练数据CSV
    train_csv_path = Path('data/Train.csv')
    train_df = pd.read_csv(train_csv_path)
    
    print("\n1. 数据集基本信息")
    print("-" * 60)
    print(f"训练集总样本数: {len(train_df)}")
    print(f"类别数量: {train_df['ClassId'].nunique()}")
    print(f"类别范围: {train_df['ClassId'].min()} - {train_df['ClassId'].max()}")
    
    # 类别分布统计
    print("\n2. 类别分布统计")
    print("-" * 60)
    class_counts = train_df['ClassId'].value_counts().sort_index()
    print(f"每个类别的样本数:")
    print(class_counts)
    
    print(f"\n类别分布统计:")
    print(f"  最小样本数: {class_counts.min()}")
    print(f"  最大样本数: {class_counts.max()}")
    print(f"  平均样本数: {class_counts.mean():.2f}")
    print(f"  标准差: {class_counts.std():.2f}")
    print(f"  中位数: {class_counts.median():.2f}")
    
    # 判断数据是否平衡
    imbalance_ratio = class_counts.max() / class_counts.min()
    print(f"\n数据不平衡比例 (最大/最小): {imbalance_ratio:.2f}")
    if imbalance_ratio > 2.0:
        print("⚠️  数据集不平衡！")
    else:
        print("✓ 数据集相对平衡")
    
    # 图像尺寸统计
    print("\n3. 图像尺寸统计")
    print("-" * 60)
    print(f"图像宽度统计:")
    print(f"  最小值: {train_df['Width'].min()}")
    print(f"  最大值: {train_df['Width'].max()}")
    print(f"  平均值: {train_df['Width'].mean():.2f}")
    print(f"  中位数: {train_df['Width'].median():.2f}")
    
    print(f"\n图像高度统计:")
    print(f"  最小值: {train_df['Height'].min()}")
    print(f"  最大值: {train_df['Height'].max()}")
    print(f"  平均值: {train_df['Height'].mean():.2f}")
    print(f"  中位数: {train_df['Height'].median():.2f}")
    
    # 绘制类别分布条形图
    print("\n4. 生成类别分布可视化图表")
    print("-" * 60)
    
    plt.figure(figsize=(16, 8))
    bars = plt.bar(range(len(class_counts)), class_counts.values, 
                   color='steelblue', alpha=0.7, edgecolor='black')
    plt.xlabel('类别ID (ClassId)', fontsize=12, fontweight='bold')
    plt.ylabel('样本数量', fontsize=12, fontweight='bold')
    plt.title('GTSRB数据集类别分布', fontsize=14, fontweight='bold')
    plt.xticks(range(len(class_counts)), class_counts.index, rotation=45)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 添加数值标签
    for i, (idx, count) in enumerate(class_counts.items()):
        plt.text(i, count + 50, str(count), ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('class_distribution.png', dpi=300, bbox_inches='tight')
    print("✓ 类别分布条形图已保存: class_distribution.png")
    
    # 绘制图像尺寸分布散点图
    plt.figure(figsize=(10, 6))
    plt.scatter(train_df['Width'], train_df['Height'], alpha=0.3, s=10)
    plt.xlabel('图像宽度 (像素)', fontsize=12, fontweight='bold')
    plt.ylabel('图像高度 (像素)', fontsize=12, fontweight='bold')
    plt.title('GTSRB数据集图像尺寸分布', fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig('image_size_distribution.png', dpi=300, bbox_inches='tight')
    print("✓ 图像尺寸分布图已保存: image_size_distribution.png")
    
    # 保存类别分布统计到CSV
    class_stats = pd.DataFrame({
        'ClassId': class_counts.index,
        'SampleCount': class_counts.values
    })
    class_stats.to_csv('class_distribution_stats.csv', index=False, encoding='utf-8-sig')
    print("✓ 类别分布统计已保存: class_distribution_stats.csv")
    
    print("\n" + "=" * 60)
    print("数据分析完成！")
    print("=" * 60)
    
    return train_df, class_counts

if __name__ == '__main__':
    train_df, class_counts = analyze_dataset()

