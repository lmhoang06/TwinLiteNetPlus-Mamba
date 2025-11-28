import torch
from torch.onnx import export
from model.model import TwinLiteNetPlus
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--config", type=str, default="small")
args = parser.parse_args()

model = TwinLiteNetPlus(args)
model.load_state_dict(torch.load("pretrained/small.pth"))
model.eval()

input = torch.randn(1, 3, 384, 640)
export(model, input, "model.onnx", verbose=False)

print("Model exported to model.onnx")

