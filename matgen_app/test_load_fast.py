import os, sys
sys.path.insert(0, 'matgen_app')
os.environ['MATGEN_DEVICE'] = 'cpu'

output = open('matgen_app/load_fast_out.txt', 'w')
output.write('Loading with float16...\n')
output.flush()

import torch
from transformers import AutoModelForCausalLM

try:
    output.write('Starting load...\n')
    output.flush()
    m = AutoModelForCausalLM.from_pretrained(
        'Qwen/Qwen2.5-7B-Instruct',
        torch_dtype=torch.float16,  # float16 instead of bfloat16
        device_map='cpu',
        low_cpu_mem_usage=True,  # Reduce memory during loading
    )
    output.write(f'Model loaded! Params: {sum(p.numel() for p in m.parameters())/1e9:.1f}B\n')
    output.flush()
except Exception as e:
    output.write(f'Error: {e}\n')
    import traceback
    traceback.print_exc(file=output)
    output.flush()
output.close()
print("Done")