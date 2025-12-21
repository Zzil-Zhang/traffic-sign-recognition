"""
cnn_model.py - CNN模型定义
成员B的任务：定义交通标志识别的CNN模型
要求：必须包含Batch Normalization和Dropout
"""

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation

def create_traffic_cnn_model(input_shape=(64, 64, 3), num_classes=43):
    """
    创建交通标志识别的CNN模型
    参数：
        input_shape: 输入图像形状 (高度, 宽度, 通道数)
        num_classes: 类别数量，德国交通标志有43类
    返回：
        Keras Sequential模型
    """
    model = Sequential(name="TrafficSignCNN")
    
    # ========== 第一卷积块 ==========
    # Conv2D + BatchNorm + ReLU + MaxPooling + Dropout
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
    model.add(BatchNormalization())  # 作业要求：Batch Normalization
    model.add(Activation('relu'))    # ReLU激活函数
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))         # 作业要求：Dropout防止过拟合
    
    # ========== 第二卷积块 ==========
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # ========== 第三卷积块 ==========
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # ========== 全连接层 ==========
    model.add(Flatten())
    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))  # 更高的Dropout率防止过拟合
    
    # ========== 输出层 ==========
    model.add(Dense(num_classes, activation='softmax'))
    
    return model

def create_simple_cnn_model(input_shape=(64, 64, 3), num_classes=43):
    """
    创建一个更简单的CNN模型（如果上面的太复杂）
    同样包含BatchNorm和Dropout
    """
    model = Sequential(name="SimpleTrafficCNN")
    
    # 卷积层1
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))
    
    # 卷积层2
    model.add(Conv2D(64, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))
    
    # 全连接层
    model.add(Flatten())
    model.add(Dense(128))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    # 输出层
    model.add(Dense(num_classes, activation='softmax'))
    
    return model

def create_reference_model(input_shape=(64, 64, 3), num_classes=43):
    """
    基于参考项目traffic_sign.py的模型，但添加了BatchNorm
    """
    model = Sequential(name="ReferenceWithBatchNorm")
    
    # 参考项目的结构，但添加BatchNorm
    model.add(Conv2D(16, (3, 3), input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(Conv2D(32, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(64, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(num_classes, activation='softmax'))
    
    return model

# 测试代码：如果直接运行这个文件，显示模型结构
if __name__ == "__main__":
    print("测试模型创建...")
    model = create_traffic_cnn_model(input_shape=(64, 64, 3))
    model.summary()