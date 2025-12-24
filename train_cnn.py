"""
train_cnn.py - 交通标志CNN模型训练脚本
成员B的主要任务：训练深度CNN模型
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json

# Keras相关
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras.optimizers import Adam

# 自定义模块
from data_preprocessing import GTSRBDataLoader
from cnn_model import create_traffic_cnn_model, create_simple_cnn_model, create_reference_model

class CNNTrainer:
    """CNN训练器类，封装所有训练逻辑"""
    
    def __init__(self, model_name='traffic_cnn', image_size=(64, 64)):
        """
        初始化训练器
        model_name: 模型名称，用于保存文件
        image_size: 图像尺寸
        """
        self.model_name = model_name
        self.image_size = image_size
        self.model = None
        self.history = None
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 创建目录
        self.create_directories()
    
    def create_directories(self):
        """创建必要的目录"""
        directories = [
            'trained_models',
            'training_logs',
            'training_results',
            'training_curves'
        ]
        for dir_name in directories:
            os.makedirs(dir_name, exist_ok=True)
    
    def load_data(self):
        """加载数据 - 使用成员A的数据加载器"""
        print("=" * 60)
        print("步骤1: 加载数据")
        print("=" * 60)
        
        try:
            # 使用成员A的GTSRBDataLoader
            loader = GTSRBDataLoader(
                data_root='data',
                image_size=self.image_size,
                normalize='minmax'  # 像素值归一化到[0,1]
            )
            
            # 加载预处理数据
            X_train, X_val, X_test, y_train, y_val, y_test = loader.load_processed_data('processed_data')
            
            print(f"✓ 数据加载成功！")
            print(f"  训练集: {X_train.shape} - {len(y_train)} 张图片")
            print(f"  验证集: {X_val.shape} - {len(y_val)} 张图片")
            print(f"  测试集: {X_test.shape} - {len(y_test)} 张图片")
            print(f"  像素值范围: [{X_train.min():.3f}, {X_train.max():.3f}]")
            
            # 转换为one-hot编码（Keras需要）
            y_train_onehot = to_categorical(y_train, 43)
            y_val_onehot = to_categorical(y_val, 43)
            y_test_onehot = to_categorical(y_test, 43)
            
            return X_train, X_val, X_test, y_train_onehot, y_val_onehot, y_test_onehot
            
        except Exception as e:
            print(f"✗ 数据加载失败: {e}")
            print("请确保：")
            print("  1. 已运行 data_preprocessing.py 生成预处理数据")
            print("  2. processed_data/ 目录包含必要的数据文件")
            print("  3. data/ 目录包含原始数据集")
            return None
    
    def create_model(self, model_type='standard'):
        """
        创建CNN模型
        model_type: 'standard', 'simple', 或 'reference'
        """
        print("\n" + "=" * 60)
        print("步骤2: 创建CNN模型")
        print("=" * 60)
        
        input_shape = (self.image_size[0], self.image_size[1], 3)
        
        if model_type == 'simple':
            self.model = create_simple_cnn_model(input_shape)
            print("使用: 简单CNN模型")
        elif model_type == 'reference':
            self.model = create_reference_model(input_shape)
            print("使用: 参考项目改进模型")
        else:
            self.model = create_traffic_cnn_model(input_shape)
            print("使用: 标准CNN模型")
        
        # 编译模型
        optimizer = Adam(learning_rate=0.001)
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("模型结构:")
        self.model.summary()
        
        return self.model
    
    def train_model(self, X_train, y_train, X_val, y_val, epochs=30, batch_size=32):
        """
        训练模型
        """
        print("\n" + "=" * 60)
        print("步骤3: 训练模型")
        print("=" * 60)
        
        # 回调函数
        callbacks = self.get_callbacks()
        
        print(f"训练参数:")
        print(f"  Epochs: {epochs}")
        print(f"  Batch Size: {batch_size}")
        print(f"  训练样本: {len(X_train)}")
        print(f"  验证样本: {len(X_val)}")
        
        # 开始训练
        print("\n开始训练...")
        self.history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1  # 显示进度条
        )
        
        print("✓ 训练完成！")
        
        return self.history
    
    def get_callbacks(self):
        """获取训练回调函数"""
        # 模型保存路径
        model_path = f'trained_models/{self.model_name}_{self.timestamp}.keras'
        best_model_path = f'trained_models/{self.model_name}_best.keras'
        
        callbacks = [
            # 保存最佳模型
            ModelCheckpoint(
                filepath=best_model_path,
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1,
                # save_format='keras'
            ),
            # 提前停止（防止过拟合）
            EarlyStopping(
                monitor='val_accuracy',
                patience=10,  # 10个epoch没有改进就停止
                restore_best_weights=True,
                verbose=1
            ),
            # 动态调整学习率
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,    # 学习率减半
                patience=5,    # 5个epoch没有改进就调整
                min_lr=0.00001,
                verbose=1
            ),
            # TensorBoard日志
            TensorBoard(
                log_dir=f'training_logs/{self.timestamp}',
                histogram_freq=1,
                write_graph=True,
                write_images=True,
                update_freq='epoch'
            )
        ]
        
        return callbacks
    
    def save_model(self):
        """保存最终模型"""
        model_path = f'trained_models/{self.model_name}_final_{self.timestamp}.keras'
        self.model.save(model_path)
        print(f"✓ 最终模型已保存: {model_path}")
        
        # 同时保存一个简单名称的副本
        simple_path = 'my_traffic_classifier.keras'
        self.model.save(simple_path)
        print(f"✓ 模型已保存为: {simple_path}")
        
        return model_path
    
    def plot_training_curves(self):
        """绘制训练曲线"""
        if self.history is None:
            print("没有训练历史数据")
            return
        
        history = self.history.history
        
        plt.figure(figsize=(14, 5))
        
        # 1. 准确率曲线
        plt.subplot(1, 2, 1)
        plt.plot(history['accuracy'], label='训练准确率', linewidth=2)
        plt.plot(history['val_accuracy'], label='验证准确率', linewidth=2)
        plt.title('模型准确率', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        
        # 标记最佳准确率
        best_val_acc = max(history['val_accuracy'])
        best_epoch = history['val_accuracy'].index(best_val_acc)
        plt.scatter(best_epoch, best_val_acc, color='red', s=100, zorder=5)
        plt.text(best_epoch, best_val_acc-0.05, f'{best_val_acc:.3f}', 
                fontsize=11, ha='center', color='red')
        
        # 2. 损失曲线
        plt.subplot(1, 2, 2)
        plt.plot(history['loss'], label='训练损失', linewidth=2)
        plt.plot(history['val_loss'], label='验证损失', linewidth=2)
        plt.title('模型损失', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        curve_path = f'training_curves/training_history_{self.timestamp}.png'
        plt.savefig(curve_path, dpi=150, bbox_inches='tight')
        plt.savefig('training_results.png', dpi=150, bbox_inches='tight')  # 简单名称
        
        print(f"✓ 训练曲线已保存: {curve_path}")
        plt.show()
    
    def save_training_report(self, X_test, y_test):
        """保存训练报告"""
        if self.model is None:
            print("没有模型可以评估")
            return
        
        # 评估模型
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        train_accuracy = self.history.history['accuracy'][-1]
        val_accuracy = self.history.history['val_accuracy'][-1]
        best_val_accuracy = max(self.history.history['val_accuracy'])
        
        # 创建报告
        report = {
            'model_name': self.model_name,
            'timestamp': self.timestamp,
            'image_size': self.image_size,
            'training_samples': len(self.history.history['accuracy']) * 32,  # 估算
            'final_train_accuracy': float(train_accuracy),
            'final_val_accuracy': float(val_accuracy),
            'best_val_accuracy': float(best_val_accuracy),
            'test_accuracy': float(test_accuracy),
            'test_loss': float(test_loss),
            'training_time_epochs': len(self.history.history['accuracy'])
        }
        
        # 保存为JSON
        report_path = f'training_results/training_report_{self.timestamp}.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=4, ensure_ascii=False)
        
        # 打印报告
        print("\n" + "=" * 60)
        print("训练报告")
        print("=" * 60)
        print(f"模型名称: {report['model_name']}")
        print(f"训练时间: {report['timestamp']}")
        print(f"图像尺寸: {report['image_size']}")
        print(f"训练轮数: {report['training_time_epochs']}")
        print(f"最终训练准确率: {report['final_train_accuracy']:.4f}")
        print(f"最终验证准确率: {report['final_val_accuracy']:.4f}")
        print(f"最佳验证准确率: {report['best_val_accuracy']:.4f}")
        print(f"测试集准确率: {report['test_accuracy']:.4f}")
        print(f"测试集损失: {report['test_loss']:.4f}")
        
        # 判断是否达到目标
        target_accuracy = 0.85  # 85%的目标
        if report['test_accuracy'] >= target_accuracy:
            print(f"🎉 恭喜！测试准确率 ({report['test_accuracy']:.2%}) 达到目标 ({target_accuracy:.2%})")
        else:
            print(f"⚠️ 测试准确率 ({report['test_accuracy']:.2%}) 未达到目标 ({target_accuracy:.2%})")
        
        print(f"\n✓ 训练报告已保存: {report_path}")
        
        return report

def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("德国交通标志识别 - CNN模型训练")
    print("成员B任务：深度学习架构师 & 模型训练")
    print("=" * 60)
    
    # 1. 创建训练器
    trainer = CNNTrainer(
        model_name='traffic_sign_cnn',
        image_size=(64, 64)  # 与成员A的预处理保持一致
    )
    
    # 2. 加载数据
    data = trainer.load_data()
    if data is None:
        print("数据加载失败，请检查数据文件")
        return
    
    X_train, X_val, X_test, y_train, y_val, y_test = data
    
    # 3. 创建模型（可以选择不同的模型类型）
    # 可选: 'standard', 'simple', 'reference'
    trainer.create_model(model_type='standard')
    
    # 4. 训练模型
    trainer.train_model(
        X_train, y_train,
        X_val, y_val,
        epochs=30,      # 训练轮数
        batch_size=32   # 批大小
    )
    
    # 5. 保存模型
    trainer.save_model()
    
    # 6. 绘制训练曲线
    trainer.plot_training_curves()
    
    # 7. 保存训练报告
    trainer.save_training_report(X_test, y_test)
    
    print("\n" + "=" * 60)
    print("训练流程完成！")
    print("=" * 60)
    print("下一步:")
    print("1. 查看训练曲线: training_results.png")
    print("2. 查看训练报告: training_results/ 目录")
    print("3. 使用模型: my_traffic_classifier.keras")
    print("4. 可视化训练过程: tensorboard --logdir=training_logs")
    print("=" * 60)

if __name__ == "__main__":
    # 检查依赖
    try:
        import tensorflow as tf
        print(f"TensorFlow版本: {tf.__version__}")
    except ImportError:
        print("错误: 请先安装TensorFlow/Keras")
        print("运行: pip install tensorflow")
        exit(1)
    
    # 运行主程序
    main()