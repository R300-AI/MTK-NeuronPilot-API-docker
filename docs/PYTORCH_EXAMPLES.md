# PyTorch to ONNX Converter Examples

This document provides examples of PyTorch model class definitions that can be used with the web interface.

**Note**: You only need to provide the model class definition. The system will automatically instantiate the model using the class name and input shape you specify in the form.

## Example 1: Simple Linear Model

**Model Class Name**: `SimpleModel`  
**Input Shape**: `(1, 10)`

```python
import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)
    
    def forward(self, x):
        return self.fc(x)
```

## Example 2: Multi-layer Neural Network

**Model Class Name**: `MLP`  
**Input Shape**: `(1, 784)`

```python
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 10)
        )
    
    def forward(self, x):
        return self.layers(x)
```

## Example 3: Convolutional Neural Network

**Model Class Name**: `SimpleCNN`  
**Input Shape**: `(1, 3, 32, 32)`

```python
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        return x
```

## Example 4: ResNet-like Block

**Model Class Name**: `SimpleResNet`  
**Input Shape**: `(1, 3, 224, 224)`

```python
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(residual)
        out = self.relu(out)
        return out

class SimpleResNet(nn.Module):
    def __init__(self):
        super(SimpleResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = ResidualBlock(64, 64)
        self.layer2 = ResidualBlock(64, 128, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, 1000)
    
    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
```

## How to Use:

1. **Model Class Definition**: Copy and paste one of the above model class definitions into the code editor.

2. **Model Class Name**: Enter the exact class name (e.g., `SimpleModel`, `MLP`, `SimpleCNN`, `SimpleResNet`) in the "Model Class Name" field.

3. **Input Shape**: Enter the input tensor shape in the format `(batch_size, ...other_dimensions)`:
   - For tabular data: `(1, num_features)`
   - For images: `(1, channels, height, width)`
   - For sequences: `(1, sequence_length, feature_dim)`

4. **Output File Name**: Choose a name for your output ONNX file.

## Notes:

- **No Instantiation Required**: Don't create model instances or define input_shape variables in your code. The system handles this automatically.
- **Multiple Classes**: You can define multiple classes (like helper classes), but specify the main model class name in the form.
- **Import Statements**: Always include necessary imports (`torch`, `torch.nn`).
- **Standard Operations**: Use standard PyTorch operations for better ONNX compatibility.
