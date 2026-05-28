import os, sys
sys.path.insert(0, 'matgen_app')
os.environ['MATGEN_DEVICE'] = 'cpu'

output = open('matgen_app/load_simple_out.txt', 'w')
output.write('Loading model...\n')
output.flush()

import torch
from transformers import AutoModelForCausalLM

try:
    m = AutoModelForCausalLM.from_pretrained(
        'Qwen/Qwen2.5-7B-Instruct',
        torch_dtype=torch.bfloat16,
        device_map='cpu'
    )
    output.write(f'Model loaded successfully!\n')
    output.write(f'Params: {sum(p.numel() for p in m.parameters())/1e9:.1f}B\n')
    output.flush()
except Exception as e:
    output.write(f'Error: {e}\n')
    import traceback
    traceback.print_exc(file=output)
    output.flush()
output.close()
print("Done - check load_simple_out.txt")