# train_cnn_paddle.py - 强制GPU版本
import os
import numpy as np
import paddle
import paddle.nn as nn
import paddle.optimizer as optim
from paddle.optimizer.lr import ReduceOnPlateau, CosineAnnealingDecay
from datetime import datetime
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

from data_utils import create_data_loaders
from cnn_model_paddle import create_model_by_type

class PaddleModelCheckpoint:
    """PaddlePaddle模型检查点回调"""
    def __init__(self, filepath, monitor='val_loss', save_best_only=True, mode='min', verbose=1):
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.mode = mode
        self.verbose = verbose
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return
            
        current = logs.get(self.monitor)
        if current is None:
            return
            
        if self.mode == 'min':
            should_save = current < self.best_value
        else:  # 'max'
            should_save = current > self.best_value
            
        if should_save or not self.save_best_only:
            if self.verbose > 0:
                print(f"保存模型: {self.filepath} ({self.monitor}: {current:.4f})")
                
            # 保存模型
            paddle.save(self.model.state_dict(), self.filepath)
            
            if should_save:
                self.best_value = current
    
    def set_model(self, model):
        self.model = model

class PaddleEarlyStopping:
    """PaddlePaddle提前停止回调"""
    def __init__(self, monitor='val_acc', patience=15, min_delta=0.001, restore_best_weights=True, verbose=1):
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        self.wait = 0
        self.stopped_epoch = 0
        self.best_value = float('-inf')
        self.best_weights = None
        
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return False
            
        current = logs.get(self.monitor)
        if current is None:
            return False
        
        # 如果是准确率，越大越好
        should_stop = current - self.best_value < self.min_delta
        
        if not should_stop:
            self.best_value = current
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = {k: v.clone() for k, v in self.model.state_dict().items()}
        else:
            self.wait += 1
            if self.verbose > 0:
                print(f"早停计数器: {self.wait}/{self.patience}")
            
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                if self.verbose > 0:
                    print(f"⚠️  提前停止在第 {epoch+1} 轮")
                if self.restore_best_weights and self.best_weights is not None:
                    print("恢复最佳模型权重...")
                    self.model.set_state_dict(self.best_weights)
                return True
        return False
    
    def set_model(self, model):
        self.model = model

class CNNTrainerPaddle:
    """CNN训练器类（强制GPU版本）"""
    
    def __init__(self, model_name='traffic_cnn_paddle', image_size=(64, 64)):
        self.model_name = model_name
        self.image_size = image_size
        self.model = None
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 强制使用GPU
        self.force_gpu_setup()
        
        # 设置随机种子
        self.setup_seeds()
        
        # 创建目录
        self.create_directories()
    
    def force_gpu_setup(self):
        """强制使用GPU，如果失败则报错退出"""
        print("=" * 60)
        print("🚀 强制GPU模式启动")
        print("=" * 60)
        
        try:
            # 方法1：优先使用依图GPU
            print("尝试使用依图GPU (iluvatar_gpu:0)...")
            paddle.set_device('iluvatar_gpu:0')
            device = paddle.device.get_device()
            print(f"✅ 成功使用依图GPU设备: {device}")
            
        except Exception as e1:
            print(f"依图GPU设置失败: {e1}")
            
            try:
                # 方法2：尝试其他GPU设备名称
                print("尝试其他GPU设备名称...")
                for device_name in ['gpu:0', 'gpu', 'cuda:0', 'cuda']:
                    try:
                        paddle.set_device(device_name)
                        device = paddle.device.get_device()
                        print(f"✅ 成功使用GPU设备: {device}")
                        break
                    except:
                        continue
                else:
                    raise Exception("所有GPU设备尝试失败")
                    
            except Exception as e2:
                print(f"所有GPU尝试失败: {e2}")
                print("❌ 错误：未检测到可用的GPU设备！")
                print("请检查：")
                print("1. 是否在支持GPU的环境中运行")
                print("2. GPU驱动是否正确安装")
                print("3. PaddlePaddle是否为GPU版本")
                print("=" * 60)
                raise SystemExit("程序终止：必须使用GPU环境")
    
    def setup_seeds(self):
        """设置随机种子"""
        paddle.seed(42)
        np.random.seed(42)
        random.seed(42)
    
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
    
    def create_model(self, model_type='simple', learning_rate=0.001):
        """创建模型"""
        print("\n" + "=" * 60)
        print("步骤2: 创建CNN模型")
        print("=" * 60)
        
        # 强制使用简单模型防止过拟合
        if model_type != 'simple':
            print(f"⚠️  自动将模型类型从 '{model_type}' 改为 'simple' 防止过拟合")
            model_type = 'simple'
        
        # 创建模型
        self.model = create_model_by_type(model_type=model_type)
        
        # 计算参数量
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if not p.stop_gradient)
        
        print(f"模型类型: {model_type}")
        print(f"总参数: {total_params:,}")
        print(f"可训练参数: {trainable_params:,}")
        
        # 使用AdamW优化器，添加权重衰减
        self.optimizer = optim.AdamW(
            learning_rate=learning_rate,
            parameters=self.model.parameters(),
            weight_decay=0.0005,  # 权重衰减防止过拟合
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8,
            grad_clip=nn.ClipGradByGlobalNorm(clip_norm=1.0)  # 梯度裁剪
        )
        
        # 使用余弦退火学习率调度
        self.scheduler = CosineAnnealingDecay(
            learning_rate=learning_rate,
            T_max=30,  # 总训练轮数
            eta_min=learning_rate * 0.01  # 最小学习率
        )
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss()
        
        print(f"优化器: AdamW with weight_decay=0.0005")
        print(f"学习率调度: CosineAnnealingDecay")
        print(f"初始学习率: {learning_rate}")
        print(f"⚠️  已启用防过拟合措施: L2正则化 + 学习率调度 + 梯度裁剪")
        
        return self.model
    
    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        # 创建进度条
        progress_bar = tqdm(train_loader, desc='训练', leave=False)
        
        for batch_idx, (data, target) in enumerate(progress_bar):
            # 前向传播
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            self.optimizer.clear_grad()
            
            # 统计
            total_loss += loss.item()
            predicted = output.argmax(axis=1)
            total += target.shape[0]
            batch_correct = (predicted == target).sum().item()
            correct += batch_correct
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{batch_correct/target.shape[0]:.2%}'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total if total > 0 else 0
        
        return avg_loss, accuracy
    
    def validate(self, val_loader):
        """验证"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        # 创建进度条
        progress_bar = tqdm(val_loader, desc='验证', leave=False)
        
        with paddle.no_grad():
            for data, target in progress_bar:
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                predicted = output.argmax(axis=1)
                total += target.shape[0]
                batch_correct = (predicted == target).sum().item()
                correct += batch_correct
                
                # 更新进度条
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{batch_correct/target.shape[0]:.2%}'
                })
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total if total > 0 else 0
        
        return avg_loss, accuracy
    
    def train_model(self, epochs=50, batch_size=32, model_type='simple', 
                   learning_rate=0.001, optimizer_type='adam'):
        """训练模型"""
        print("\n" + "=" * 60)
        print("步骤3: 训练模型 (GPU加速)")
        print("=" * 60)
        print("🚨 正在应用过拟合修复方案...")
        print("   1. 强制使用简单模型")
        print("   2. 添加L2正则化 (weight_decay=0.0005)")
        print("   3. 添加余弦退火学习率调度")
        print("   4. 梯度裁剪 (clip_norm=1.0)")
        print("   5. 强制GPU加速训练")
        print("=" * 60)
        
        # 加载数据
        print("📂 加载数据...")
        train_loader, val_loader, test_loader, data_info = create_data_loaders(
            batch_size=batch_size,
            augment_train=True  # 启用数据增强
        )
        
        if train_loader is None:
            print("❌ 数据加载失败，退出训练")
            return None
        
        X_train, y_train, X_val, y_val, X_test, y_test = data_info
        
        print(f"\n🔍 数据检查:")
        print(f"训练集样本数量: {len(X_train)}")
        print(f"验证集样本数量: {len(X_val)}")
        print(f"测试集样本数量: {len(X_test)}")
        print(f"训练批次数量: {len(train_loader)}")
        print(f"验证批次数量: {len(val_loader)}")
        
        # 创建模型
        self.create_model(model_type=model_type, learning_rate=learning_rate)
        
        # GPU性能测试
        print(f"\n⚡ GPU性能测试...")
        start_test = datetime.now()
        test_tensor = paddle.randn([128, 3, 64, 64])
        test_result = test_tensor * 2.0
        end_test = datetime.now()
        test_time = (end_test - start_test).total_seconds()
        print(f"GPU张量运算测试: {test_time:.4f}秒")
        
        # 创建回调函数
        checkpoint_path = f'trained_models/{self.model_name}_best_{self.timestamp}.pdparams'
        checkpoint = PaddleModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_acc',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        checkpoint.set_model(self.model)
        
        # 提前停止回调
        early_stopping = PaddleEarlyStopping(
            monitor='val_acc',
            patience=15,
            min_delta=0.001,
            restore_best_weights=True,
            verbose=1
        )
        early_stopping.set_model(self.model)
        
        print(f"\n🚀 开始GPU加速训练 (总轮数: {epochs})...")
        print(f"📊 当前设备: {paddle.device.get_device()}")
        print("-" * 60)
        
        start_time = datetime.now()
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # 验证
            val_loss, val_acc = self.validate(val_loader)
            
            # 更新学习率
            current_lr = self.scheduler.get_lr()
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(current_lr)
            
            # 调用回调函数
            logs = {'val_acc': val_acc, 'val_loss': val_loss}
            checkpoint.on_epoch_end(epoch, logs)
            
            # 检查是否提前停止
            if early_stopping.on_epoch_end(epoch, logs):
                print(f"⚠️  提前停止训练")
                break
            
            # 打印进度
            print(f"  训练 - 损失: {train_loss:.4f}, 准确率: {train_acc:.4f} ({train_acc:.2%})")
            print(f"  验证 - 损失: {val_loss:.4f}, 准确率: {val_acc:.4f} ({val_acc:.2%})")
            print(f"  学习率: {current_lr:.6f}")
            
            # 检查过拟合警告
            if epoch >= 5:  # 至少训练5轮后再检查
                gap = train_acc - val_acc
                if gap > 0.3:
                    print(f"⚠️  严重过拟合警告: 训练-验证差距={gap:.4f}")
                elif gap > 0.2:
                    print(f"⚠️  中度过拟合警告: 训练-验证差距={gap:.4f}")
            
            # 学习率调度
            self.scheduler.step()
        
        # 训练完成
        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds()
        
        print("\n" + "=" * 60)
        print("✅ GPU训练完成！")
        print("=" * 60)
        print(f"训练时间: {training_time:.2f}秒")
        print(f"平均每轮: {training_time/len(self.history['train_loss']):.2f}秒")
        
        # 分析过拟合程度
        self.analyze_overfitting()
        
        # 加载最佳模型
        best_model_path = f'trained_models/{self.model_name}_best_{self.timestamp}.pdparams'
        if os.path.exists(best_model_path):
            print(f"📂 加载最佳模型...")
            self.model.set_state_dict(paddle.load(best_model_path))
        
        # 保存最终模型
        self.save_model()
        
        # 评估模型
        test_acc = self.evaluate_model(test_loader)
        
        # 绘制训练曲线
        self.plot_training_curves()
        
        # 保存训练报告
        self.save_training_report(test_acc, len(X_test), training_time)
        
        return self.history
    
    def analyze_overfitting(self):
        """分析过拟合程度"""
        if not self.history['train_acc']:
            return
        
        final_train_acc = self.history['train_acc'][-1]
        final_val_acc = self.history['val_acc'][-1]
        gap = final_train_acc - final_val_acc
        
        # 找到最佳验证准确率
        best_val_acc = max(self.history['val_acc'])
        best_val_epoch = self.history['val_acc'].index(best_val_acc) + 1
        
        print(f"\n📊 过拟合分析:")
        print(f"  最终训练准确率: {final_train_acc:.4f} ({final_train_acc:.2%})")
        print(f"  最终验证准确率: {final_val_acc:.4f} ({final_val_acc:.2%})")
        print(f"  训练-验证差距: {gap:.4f}")
        print(f"  最佳验证准确率: {best_val_acc:.4f} ({best_val_acc:.2%}) - 第{best_val_epoch}轮")
        
        if gap > 0.3:
            print(f"  🔴 严重过拟合！建议:")
            print(f"    1. 进一步增加权重衰减到0.001")
            print(f"    2. 增加Dropout率")
            print(f"    3. 减少训练轮数")
        elif gap > 0.2:
            print(f"  🟡 中等过拟合")
        elif gap > 0.1:
            print(f"  🟢 轻微过拟合")
        else:
            print(f"  ✅ 优秀的泛化能力！")
    
    def save_model(self):
        """保存最终模型"""
        model_path = f'trained_models/{self.model_name}_final_{self.timestamp}.pdparams'
        paddle.save(self.model.state_dict(), model_path)
        
        # 同时保存一个简单名称的副本
        simple_path = 'my_traffic_classifier_paddle.pdparams'
        paddle.save(self.model.state_dict(), simple_path)
        
        print(f"✅ 最终模型已保存: {model_path}")
        print(f"✅ 模型已保存为: {simple_path}")
        
        return model_path
    
    def evaluate_model(self, test_loader):
        """评估模型"""
        print("\n" + "=" * 60)
        print("步骤4: 评估模型")
        print("=" * 60)
        
        if test_loader is None:
            print("❌ 测试数据加载器为空")
            return 0.0
        
        self.model.eval()
        correct = 0
        total = 0
        test_loss = 0
        
        print("📊 评估测试集...")
        
        with paddle.no_grad():
            for data, target in tqdm(test_loader, desc='测试'):
                output = self.model(data)
                loss = self.criterion(output, target)
                
                test_loss += loss.item()
                predicted = output.argmax(axis=1)
                total += target.shape[0]
                correct += (predicted == target).sum().item()
        
        avg_test_loss = test_loss / len(test_loader)
        test_acc = correct / total if total > 0 else 0
        
        print(f"测试集损失: {avg_test_loss:.4f}")
        print(f"测试集准确率: {test_acc:.4f} ({test_acc:.2%})")
        print(f"测试样本数: {total}")
        
        return test_acc
    
    def plot_training_curves(self):
        """绘制训练曲线"""
        if not self.history['train_acc'] or len(self.history['train_acc']) < 2:
            print("训练历史数据不足，无法绘制曲线")
            return
        
        plt.figure(figsize=(15, 5))
        
        # 1. 准确率曲线
        plt.subplot(1, 3, 1)
        epochs = range(1, len(self.history['train_acc']) + 1)
        
        plt.plot(epochs, self.history['train_acc'], 'b-', label='训练准确率', linewidth=2)
        plt.plot(epochs, self.history['val_acc'], 'r-', label='验证准确率', linewidth=2)
        
        # 标记最佳验证准确率
        best_val_acc = max(self.history['val_acc'])
        best_epoch = self.history['val_acc'].index(best_val_acc)
        plt.scatter(best_epoch + 1, best_val_acc, color='red', s=100, zorder=5)
        plt.text(best_epoch + 1, best_val_acc - 0.05, f'最佳: {best_val_acc:.3f}', 
                fontsize=10, ha='center', color='red')
        
        plt.title('准确率曲线', fontsize=12, fontweight='bold')
        plt.xlabel('训练轮数', fontsize=11)
        plt.ylabel('准确率', fontsize=11)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.ylim([0, 1.0])
        
        # 2. 损失曲线
        plt.subplot(1, 3, 2)
        plt.plot(epochs, self.history['train_loss'], 'b-', label='训练损失', linewidth=2)
        plt.plot(epochs, self.history['val_loss'], 'r-', label='验证损失', linewidth=2)
        plt.title('损失曲线', fontsize=12, fontweight='bold')
        plt.xlabel('训练轮数', fontsize=11)
        plt.ylabel('损失', fontsize=11)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # 3. 学习率曲线
        plt.subplot(1, 3, 3)
        plt.plot(epochs, self.history['lr'], 'g-', label='学习率', linewidth=2)
        plt.title('学习率变化', fontsize=12, fontweight='bold')
        plt.xlabel('训练轮数', fontsize=11)
        plt.ylabel('学习率', fontsize=11)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        plt.tight_layout()
        
        # 保存图片
        curve_path = f'training_curves/training_history_{self.timestamp}.png'
        plt.savefig(curve_path, dpi=150, bbox_inches='tight')
        plt.savefig('training_results_paddle.png', dpi=150, bbox_inches='tight')
        
        print(f"✅ 训练曲线已保存: {curve_path}")
        plt.show()
    
    def save_training_report(self, test_acc, test_samples, training_time):
        """保存训练报告"""
        if not self.history['train_acc']:
            print("没有训练历史数据")
            return
        
        # 收集数据
        train_acc = self.history['train_acc'][-1]
        val_acc = self.history['val_acc'][-1]
        best_val_acc = max(self.history['val_acc'])
        epochs = len(self.history['train_acc'])
        gap = train_acc - val_acc
        
        # 评估过拟合程度
        if gap > 0.3:
            overfitting_status = '严重过拟合'
            status_emoji = '🔴'
        elif gap > 0.2:
            overfitting_status = '中等过拟合'
            status_emoji = '🟡'
        elif gap > 0.1:
            overfitting_status = '轻微过拟合'
            status_emoji = '🟢'
        else:
            overfitting_status = '优秀泛化'
            status_emoji = '✅'
        
        # 创建报告
        report = {
            'model_name': self.model_name,
            'timestamp': self.timestamp,
            'image_size': self.image_size,
            'training_time_seconds': float(training_time),
            'training_epochs': int(epochs),
            'final_train_accuracy': float(train_acc),
            'final_val_accuracy': float(val_acc),
            'train_val_gap': float(gap),
            'best_val_accuracy': float(best_val_acc),
            'test_accuracy': float(test_acc),
            'test_samples': int(test_samples),
            'overfitting_status': overfitting_status,
            'status_emoji': status_emoji,
            'device': str(paddle.device.get_device()),
            'parameters': {
                'batch_size': 32,
                'learning_rate': 0.001,
                'weight_decay': 0.0005,
                'patience': 15
            }
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
        print(f"训练设备: {report['device']}")
        print(f"图像尺寸: {report['image_size']}")
        print(f"训练轮数: {report['training_epochs']}")
        print(f"训练用时: {report['training_time_seconds']:.2f}秒")
        print(f"最终训练准确率: {report['final_train_accuracy']:.4f} ({report['final_train_accuracy']:.2%})")
        print(f"最终验证准确率: {report['final_val_accuracy']:.4f} ({report['final_val_accuracy']:.2%})")
        print(f"训练-验证差距: {report['train_val_gap']:.4f} - {status_emoji} {overfitting_status}")
        print(f"最佳验证准确率: {report['best_val_accuracy']:.4f} ({report['best_val_accuracy']:.2%})")
        print(f"测试集准确率: {report['test_accuracy']:.4f} ({report['test_accuracy']:.2%})")
        print(f"测试样本数: {report['test_samples']}")
        
        # 根据结果给出建议
        print(f"\n{status_emoji} 建议:")
        if overfitting_status == '严重过拟合':
            print("  1. 进一步增加权重衰减到0.001")
            print("  2. 增加Dropout率")
            print("  3. 使用更简单的模型")
            print("  4. 增加数据增强强度")
        elif overfitting_status == '中等过拟合':
            print("  1. 略微增加权重衰减")
            print("  2. 增加早停耐心值")
            print("  3. 调整数据增强参数")
        elif overfitting_status == '轻微过拟合':
            print("  1. 模型状态良好")
            print("  2. 可以尝试调整学习率")
        else:
            print("  1. 模型泛化能力优秀")
            print("  2. 可以尝试微调参数获得更好结果")
        
        print(f"\n✅ 训练报告已保存: {report_path}")
        
        return report

def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("德国交通标志识别 - CNN模型训练 (强制GPU加速版)")
    print("=" * 60)
    
    # 创建训练器
    trainer = CNNTrainerPaddle(
        model_name='traffic_sign_cnn_paddle',
        image_size=(64, 64)
    )
    
    # 训练模型 - 使用稳定参数
    history = trainer.train_model(
        epochs=50,  # 增加最大轮数，但会有早停
        batch_size=32,
        model_type='simple',  # 强制使用简单模型
        learning_rate=0.001,
        optimizer_type='adam'
    )
    
    if history is not None:
        print("\n" + "=" * 60)
        print("✅ 训练流程完成！")
        print("=" * 60)
        print("强制GPU模式总结:")
        print("✅ 1. 强制使用GPU，不会退回到CPU")
        print("✅ 2. 如果GPU不可用，程序会报错退出")
        print("✅ 3. 使用依图GPU加速卡 (iluvatar_gpu:0)")
        print("=" * 60)
        print("过拟合修复措施:")
        print("✅ 1. 强制使用简单模型 (减少参数)")
        print("✅ 2. 添加L2正则化 (weight_decay=0.0005)")
        print("✅ 3. 添加余弦退火学习率调度")
        print("✅ 4. 梯度裁剪 (clip_norm=1.0)")
        print("✅ 5. 改进早停策略 (patience=15)")
        print("✅ 6. 随机种子设置 (确保可重复性)")
        print("=" * 60)
        print("输出文件:")
        print("1. 训练曲线: training_results_paddle.png")
        print("2. 详细曲线: training_curves/ 目录")
        print("3. 训练报告: training_results/ 目录")
        print("4. 最佳模型: trained_models/ 目录")
        print("5. 简版模型: my_traffic_classifier_paddle.pdparams")
        print("=" * 60)
        print("下一步:")
        print("1. 查看训练曲线分析模型表现")
        print("2. 使用测试集验证模型性能")
        print("3. 根据报告调整参数进一步优化")
        print("=" * 60)

if __name__ == "__main__":
    print(f"PaddlePaddle版本: {paddle.__version__}")
    print(f"NumPy版本: {np.__version__}")
    
    # 检查数据目录
    if not os.path.exists('processed_data'):
        print("📁 创建processed_data目录...")
        os.makedirs('processed_data', exist_ok=True)
        print("将使用示例数据进行测试")
    
    # 运行主程序
    main()