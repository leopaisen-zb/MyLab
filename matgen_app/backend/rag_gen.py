"""
HEA-Gen LLM backend: text prompt -> POSCAR text
Uses qwen_vasp_lora/checkpoint-3114 (direct POSCAR generation, not code generation).
Set MATGEN_DEMO=1 to use a canned POSCAR without loading the LLM (for demos/testing).
"""

import os
from pathlib import Path


def get_device():
    """Get torch device based on MATGEN_DEVICE env var."""
    override = os.environ.get("MATGEN_DEVICE", "auto")
    if override == "directml":
        import torch_directml
        return torch_directml.device()
    elif override == "cuda" and os.environ.get("CUDA_VISIBLE_DEVICES"):
        import torch
        return torch.device("cuda")
    elif override == "mps":
        import torch
        return torch.device("mps")
    else:
        import torch
        return torch.device("cpu")

# Demo POSCAR — real IrPdPtRhRu HEA surface structure from training set
_DEMO_POSCAR = """\
Rh Pd Rh Pd Rh Pd Rh  H
1.0000000000000000
   5.4011  0.0000  0.0000
   2.7005  4.6775  0.0000
   0.0000  0.0000  24.4100
Rh  Pd  Rh  Pd  Rh  Pd  Rh  H
1   1   3   1   3   1   2   1
Cartesian
  0.0000  0.0000  9.9999
  1.3502  2.3387  9.9999
  2.7005  0.0000  9.9999
  4.0508  2.3387  9.9999
  5.4011  3.1183 12.2050
  4.0508  0.7795 12.2050
  2.7005  3.1183 12.2050
  1.3502  0.7795 12.2050
  2.7310  1.5183 14.4000
  4.0507  3.8618 14.4499
  5.3805  1.5473 14.3526
  6.7499  3.8767 14.3978
  5.3738  4.6622 15.3140
"""

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # mylab(1)/
LORA_CKPT = PROJECT_ROOT / "RAG" / "RAG" / "checkpoints" / "qwen_vasp_lora" / "checkpoint-3114"
SYSTEM_INSTRUCTION = (
    "You are a materials science expert. Given the physical and chemical properties "
    "of a hydrogen storage material and several reference structures with similar properties, "
    "generate the corresponding atomic structure in VASP POSCAR format."
)

_model = None
_tokenizer = None


def load_model(base_model_name_or_path: str = None):
    global _model, _tokenizer

    if _model is not None and _tokenizer is not None:
        return _model, _tokenizer

    if not LORA_CKPT.exists():
        raise FileNotFoundError(
            f"LoRA checkpoint not found: {LORA_CKPT}\n"
            "Ensure the checkpoint directory exists before loading the model."
        )

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise ImportError(
            "transformers and torch are required. Install with:\n"
            "  pip install transformers torch"
        ) from e

    try:
        from peft import PeftModel
    except ImportError as e:
        raise ImportError(
            "peft is required for LoRA adapter loading. Install with:\n"
            "  pip install peft"
        ) from e

    base = base_model_name_or_path or "Qwen/Qwen2.5-7B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)

    # Get device for model loading
    dml_device = get_device()

    try:
        # Use float16 for DirectML, bfloat16 for CUDA
        if dml_device.type == "privateuseone":
            torch_dtype = torch.float16
            # Load on CPU first then move to DirectML
            base_model = AutoModelForCausalLM.from_pretrained(
                base,
                torch_dtype=torch_dtype,
                device_map="cpu",
                trust_remote_code=True,
            )
            base_model = base_model.to(dml_device)
        else:
            torch_dtype = torch.bfloat16
            base_model = AutoModelForCausalLM.from_pretrained(
                base,
                torch_dtype=torch_dtype,
                device_map="auto",
                trust_remote_code=True,
            )
        model = PeftModel.from_pretrained(base_model, str(LORA_CKPT))
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load model from '{base}' with LoRA adapter at '{LORA_CKPT}'.\n"
            f"Original error: {exc}\n"
            "If the base model is not cached locally, ensure internet access or provide a "
            "local path via base_model_name_or_path."
        ) from exc

    model.eval()
    _model = model
    _tokenizer = tokenizer
    return _model, _tokenizer


def clean_poscar(raw_text: str) -> str:
    lines = raw_text.splitlines()

    # Strip markdown code fences
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            continue
        cleaned.append(line)

    # Find scale factor line (a bare float, typically "1.0000000000000000")
    # POSCAR structure: line 0 = title/elements, line 1 = scale factor
    scale_idx = None
    for i, line in enumerate(cleaned):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            float(stripped)
            # Confirm it looks like a scale factor (positive float, no spaces)
            if " " not in stripped and float(stripped) > 0:
                scale_idx = i
                break
        except ValueError:
            continue

    if scale_idx is not None and scale_idx > 0:
        # Title line is immediately before the scale factor
        start = scale_idx - 1
        cleaned = cleaned[start:]

    # Strip trailing explanatory prose: drop lines after the last coordinate line
    # Coordinate lines contain 3 floats separated by spaces
    last_coord = None
    for i, line in enumerate(cleaned):
        parts = line.split()
        if len(parts) == 3:
            try:
                float(parts[0])
                float(parts[1])
                float(parts[2])
                last_coord = i
            except ValueError:
                pass

    if last_coord is not None:
        cleaned = cleaned[: last_coord + 1]

    return "\n".join(cleaned).strip()


def generate(
    prompt: str,
    max_new_tokens: int = 1024,
    temperature: float = 0.0,
    top_p: float = 1.0,
    base_model_name_or_path: str = None,
    use_rag: bool = False,
    rag_top_k: int = 3,
) -> str:
    if os.environ.get("MATGEN_DEMO", "0") == "1":
        return _DEMO_POSCAR.strip()

    # RAG enhancement: retrieve similar reference structures
    if use_rag:
        try:
            from backend.rag_retrieve import retrieve, build_rag_prompt_minimal
            refs = retrieve(prompt, top_k=rag_top_k)
            if refs:
                enhanced_prompt = build_rag_prompt_minimal(prompt, refs)
                prompt_to_use = enhanced_prompt
            else:
                prompt_to_use = prompt
        except Exception:
            prompt_to_use = prompt
    else:
        prompt_to_use = prompt

    model, tokenizer = load_model(base_model_name_or_path)

    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": prompt_to_use},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    gen_kwargs = dict(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
    )

    if temperature > 0.0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p
    else:
        gen_kwargs["do_sample"] = False

    import torch

    with torch.no_grad():
        output_ids = model.generate(**gen_kwargs)

    new_tokens = output_ids[0][input_ids.shape[-1]:]
    raw_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return clean_poscar(raw_text)
