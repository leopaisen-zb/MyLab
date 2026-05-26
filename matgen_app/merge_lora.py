"""
Merge LoRA adapter with base Qwen2.5-7B model.
After merging, we can use the merged model directly or convert to Ollama format.
"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ['MATGEN_DEVICE'] = 'cpu'  # CPU for merging (memory intensive)

output_file = PROJECT_ROOT / "merge_output.txt"
with open(output_file, "w") as f:
    f.write("=== MERGING LoRA WITH BASE MODEL ===\n")
    f.flush()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    LORA_CKPT = PROJECT_ROOT.parent / "RAG" / "RAG" / "checkpoints" / "qwen_vasp_lora" / "checkpoint-3114"
    MERGED_OUTPUT = PROJECT_ROOT.parent / "models" / "qwen_vasp_merged"

    f.write(f"LoRA checkpoint: {LORA_CKPT}\n")
    f.write(f"Output path: {MERGED_OUTPUT}\n")
    f.flush()

    # Create output directory
    MERGED_OUTPUT.mkdir(parents=True, exist_ok=True)

    f.write("\nLoading base model Qwen2.5-7B-Instruct...\n")
    f.flush()

    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-7B-Instruct",
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            trust_remote_code=True,
        )
        f.write(f"Base model loaded. Size: {sum(p.numel() for p in base_model.parameters())/1e9:.1f}B params\n")
        f.flush()

        f.write("\nLoading LoRA adapter...\n")
        f.flush()
        model = PeftModel.from_pretrained(base_model, str(LORA_CKPT))
        f.write(f"LoRA loaded. LoRA params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M\n")
        f.flush()

        f.write("\nMerging LoRA into base model (this may take a while)...\n")
        f.flush()
        merged_model = model.merge_and_unload()
        f.write("Merge complete!\n")
        f.flush()

        f.write("\nSaving merged model...\n")
        f.flush()
        merged_model.save_pretrained(MERGED_OUTPUT)
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
        tokenizer.save_pretrained(MERGED_OUTPUT)
        f.write(f"Merged model saved to: {MERGED_OUTPUT}\n")
        f.flush()

        # Check size
        import shutil
        total_size = sum(f.stat().st_size for f in MERGED_OUTPUT.rglob('*') if f.is_file()) / 1e9
        f.write(f"Total size: {total_size:.2f} GB\n")
        f.flush()

    except Exception as e:
        f.write(f"Error: {e}\n")
        import traceback
        traceback.print_exc(file=f)
        f.flush()

print(f"Output: {output_file}")
