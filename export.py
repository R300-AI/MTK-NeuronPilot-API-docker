# Simple PyTorch Model Example
import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(10, 32)
        self.fc2 = nn.Linear(32, 10)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
model = SimpleModel()
model.eval()
dummy_input = torch.randn(1, 10)
torch.onnx.export(model, dummy_input, 'model.onnx', opset_version=11)
