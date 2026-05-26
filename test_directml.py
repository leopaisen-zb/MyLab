"""Quick DirectML model inference test"""
import sys, os, json, time
sys.path.insert(0, r'h:\BaiduNetdiskDownload\mylab(1)\mylab(1)\matgen_app')
os.environ["MATGEN_DEVICE"] = "directml"

results = []

import torch
import torch_directml as dml
dml_dev = dml.device()
results.append(f"DirectML: {dml_dev}")

# Quick matmul test on GPU
N = 512
a = torch.randn(N, N, device=dml_dev)
b = torch.randn(N, N, device=dml_dev)
start = time.time()
c = torch.matmul(a, b)
elapsed = time.time() - start
results.append(f"Matmul {N}x{N}: {elapsed:.3f}s on {c.device}")

# Test with a simple model structure (just Linear layers)
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(128, 256)
        self.fc2 = torch.nn.Linear(256, 64)
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

model = SimpleModel().to(dml_dev).eval()
x = torch.randn(32, 128, device=dml_dev)
start = time.time()
with torch.no_grad():
    y = model(x)
elapsed = time.time() - start
results.append(f"Simple forward pass: {elapsed:.4f}s, output shape: {y.shape}, device: {y.device}")

results.append("SUCCESS: DirectML inference verified!")
output_path = r'h:\BaiduNetdiskDownload\mylab(1)\mylab(1)\test_output.txt'
with open(output_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(results))
print("Done")
