import os
os.environ['MATGEN_DEVICE'] = 'directml'

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

try:
    import torch_directml
    dml = torch_directml.device()
    print(f"DirectML device: {dml}")
    # Try a simple matmul
    a = torch.randn(100, 100, device=dml)
    b = torch.randn(100, 100, device=dml)
    c = torch.matmul(a, b)
    print(f"DirectML matmul works! Result shape: {c.shape}")
except Exception as e:
    print(f"DirectML error: {e}")
    import traceback
    traceback.print_exc()
