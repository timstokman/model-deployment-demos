import torch
from torch import nn
 
# Define a model
model = nn.Sequential(
	nn.Linear(200, 100),
	nn.ReLU(),
	nn.Linear(100, 50),
	nn.ReLU(),
	nn.Linear(50, 1),
	nn.Sigmoid()
)

# Usually you would either train the model here, or load a checkpoint..

# Export with example input
export_output = torch.onnx.export(model, torch.randn(200), "my_model.onnx", opset_version=10, do_constant_folding=True, input_names=['input'], output_names=['output'])