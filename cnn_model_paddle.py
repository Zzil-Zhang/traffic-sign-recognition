# cnn_model_paddle.py
import paddle.nn as nn
import paddle

class TrafficCNNPaddle(nn.Layer):
    def __init__(self, num_classes=43):
        super(TrafficCNNPaddle, self).__init__()
        
        self.conv1 = nn.Conv2D(3, 32, 5, padding=2)
        self.bn1 = nn.BatchNorm2D(32)
        self.pool1 = nn.MaxPool2D(2, 2)
        self.dropout1 = nn.Dropout(0.2)
        
        self.conv2 = nn.Conv2D(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2D(64)
        self.pool2 = nn.MaxPool2D(2, 2)
        self.dropout2 = nn.Dropout(0.2)
        
        self.conv3 = nn.Conv2D(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2D(128)
        self.pool3 = nn.MaxPool2D(2, 2)
        self.dropout3 = nn.Dropout(0.3)
        
        self.conv4 = nn.Conv2D(128, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2D(256)
        self.pool4 = nn.MaxPool2D(2, 2)
        self.dropout4 = nn.Dropout(0.3)
        
        self.global_avg_pool = nn.AdaptiveAvgPool2D((1, 1))
        
        self.fc1 = nn.Linear(256, 256)
        self.bn5 = nn.BatchNorm1D(256)
        self.dropout5 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(256, num_classes)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = self.dropout3(x)
        
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)
        x = self.dropout4(x)
        
        x = self.global_avg_pool(x)
        x = paddle.flatten(x, 1)
        
        x = self.relu(self.bn5(self.fc1(x)))
        x = self.dropout5(x)
        x = self.fc2(x)
        
        return x

class SimpleCNNPaddle(nn.Layer):
    def __init__(self, num_classes=43):
        super(SimpleCNNPaddle, self).__init__()
        
        self.conv1 = nn.Conv2D(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2D(32)
        self.pool1 = nn.MaxPool2D(2, 2)
        self.dropout1 = nn.Dropout(0.1)
        
        self.conv2 = nn.Conv2D(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2D(64)
        self.pool2 = nn.MaxPool2D(2, 2)
        self.dropout2 = nn.Dropout(0.1)
        
        self.conv3 = nn.Conv2D(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2D(128)
        self.pool3 = nn.MaxPool2D(2, 2)
        self.dropout3 = nn.Dropout(0.2)
        
        self.global_avg_pool = nn.AdaptiveAvgPool2D((1, 1))
        
        self.fc1 = nn.Linear(128, 128)
        self.bn4 = nn.BatchNorm1D(128)
        self.dropout4 = nn.Dropout(0.4)
        
        self.fc2 = nn.Linear(128, num_classes)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.dropout1(x)
        
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.dropout2(x)
        
        x = self.pool3(self.relu(self.bn3(self.conv3(x))))
        x = self.dropout3(x)
        
        x = self.global_avg_pool(x)
        x = paddle.flatten(x, 1)
        
        x = self.relu(self.bn4(self.fc1(x)))
        x = self.dropout4(x)
        x = self.fc2(x)
        return x

class ReferenceModelPaddle(nn.Layer):
    def __init__(self, num_classes=43):
        super(ReferenceModelPaddle, self).__init__()
        
        self.conv1 = nn.Conv2D(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2D(16)
        
        self.conv2 = nn.Conv2D(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2D(32)
        self.pool1 = nn.MaxPool2D(2, 2)
        self.dropout1 = nn.Dropout(0.1)
        
        self.conv3 = nn.Conv2D(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2D(64)
        self.pool2 = nn.MaxPool2D(2, 2)
        self.dropout2 = nn.Dropout(0.2)
        
        self.global_avg_pool = nn.AdaptiveAvgPool2D((1, 1))
        
        self.fc1 = nn.Linear(64, 128)
        self.bn4 = nn.BatchNorm1D(128)
        self.dropout3 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(128, num_classes)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        x = self.global_avg_pool(x)
        x = paddle.flatten(x, 1)
        
        x = self.relu(self.bn4(self.fc1(x)))
        x = self.dropout3(x)
        x = self.fc2(x)
        return x

def create_traffic_cnn_model(input_shape=(64, 64, 3), num_classes=43, **kwargs):
    model = TrafficCNNPaddle(num_classes=num_classes)
    return model

def create_simple_cnn_model(input_shape=(64, 64, 3), num_classes=43, **kwargs):
    model = SimpleCNNPaddle(num_classes=num_classes)
    return model

def create_reference_model(input_shape=(64, 64, 3), num_classes=43, **kwargs):
    model = ReferenceModelPaddle(num_classes=num_classes)
    return model

def create_model_by_type(model_type='standard', num_classes=43, **kwargs):
    if model_type == 'simple':
        return create_simple_cnn_model(num_classes=num_classes, **kwargs)
    elif model_type == 'reference':
        return create_reference_model(num_classes=num_classes, **kwargs)
    else:
        return create_traffic_cnn_model(num_classes=num_classes, **kwargs)

def create_stable_model(model_type='simple', num_classes=43):
    if model_type == 'simple':
        model = SimpleCNNPaddle(num_classes=num_classes)
    elif model_type == 'reference':
        model = ReferenceModelPaddle(num_classes=num_classes)
    else:
        model = TrafficCNNPaddle(num_classes=num_classes)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if not p.stop_gradient)
    print(f"总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    return model

if __name__ == "__main__":
    
    model1 = create_traffic_cnn_model()
    print(f"标准模型参数: {sum(p.numel() for p in model1.parameters()):,}")
    
    model2 = create_simple_cnn_model()
    print(f"简单模型参数: {sum(p.numel() for p in model2.parameters()):,}")
    
    model3 = create_reference_model()
    print(f"参考模型参数: {sum(p.numel() for p in model3.parameters()):,}")
    
    test_input = paddle.randn([1, 3, 64, 64])
    print(f"标准模型输出形状: {model1(test_input).shape}")
    print(f"简单模型输出形状: {model2(test_input).shape}")
    print(f"参考模型输出形状: {model3(test_input).shape}")