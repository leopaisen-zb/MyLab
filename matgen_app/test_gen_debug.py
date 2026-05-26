import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ['MATGEN_DEVICE'] = 'directml'

output_file = PROJECT_ROOT / "debug_output.txt"
with open(output_file, "w") as f:
    f.write("=== DEBUG TEST ===\n")
    f.flush()

    # Test 1: Check transformers version
    import transformers
    f.write(f"Transformers version: {transformers.__version__}\n")
    f.flush()

    # Test 2: Check torch
    import torch
    f.write(f"PyTorch version: {torch.__version__}\n")
    f.write(f"CUDA available: {torch.cuda.is_available()}\n")
    f.flush()

    # Test 3: Check DirectML
    try:
        import torch_directml
        dml_dev = torch_directml.device()
        f.write(f"DirectML device: {dml_dev}\n")
        f.flush()
    except Exception as e:
        f.write(f"DirectML error: {e}\n")
        f.flush()

    # Test 4: Try loading the base model
    f.write("\nLoading base model Qwen/Qwen2.5-7B-Instruct...\n")
    f.flush()
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        base = "Qwen/Qwen2.5-7B-Instruct"
        f.write(f"Loading tokenizer from {base}...\n")
        f.flush()
        tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
        f.write(f"Tokenizer loaded. EOS token: {tokenizer.eos_token_id}\n")
        f.flush()

        f.write("Loading base model (this may take a while)...\n")
        f.flush()
        base_model = AutoModelForCausalLM.from_pretrained(
            base,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        f.write(f"Base model loaded on device: {base_model.device}\n")
        f.flush()
    except Exception as e:
        f.write(f"Error loading base model: {e}\n")
        import traceback
        traceback.print_exc(file=f)
        f.flush()

print(f"Debug output written to {output_file}")