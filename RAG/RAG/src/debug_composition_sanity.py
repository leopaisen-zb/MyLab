from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from pymatgen.core import Lattice, Structure

import eval_sandbox as es
from unsloth import FastLanguageModel


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "rag_data"
    / "text2struct_rag"
)

TEST_JSON = DATA_DIR / "dataset_test_rag_code.json"
EVAL_JSON_B = DATA_DIR / "eval_sandbox_FINAL_promptB.json"


def _load_candidates(
    max_samples: int = 5,
) -> List[int]:
    """从 B 结果里挑出 is_valid_structure=1 但 composition=0 的若干样本 idx。"""
    with EVAL_JSON_B.open("r", encoding="utf-8") as f:
        results = json.load(f)["results"]
    cands: List[int] = [
        r["idx"]
        for r in results
        if r.get("is_valid_structure", 0) == 1
        and r.get("is_composition_match", 0) == 0
    ]
    return cands[:max_samples]


def _build_single_prompt(
    sample: Dict[str, Any],
    tokenizer: Any,
) -> str:
    trimmed = es.trim_input_if_needed(
        sample.get("input", ""),
        sample.get("instruction", ""),
        tokenizer,
    )
    return es.build_code_prompt(
        sample.get("instruction", ""),
        trimmed,
        tokenizer,
    )


def _run_single_inference(
    prompt: str,
    model: Any,
    tokenizer: Any,
) -> str:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=es.MAX_SEQ_LENGTH,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=es.MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    input_len: int = inputs["input_ids"].shape[1]
    generated_ids = outputs[:, input_len:]
    gen_text: str = tokenizer.decode(
        generated_ids[0],
        skip_special_tokens=True,
    )
    return gen_text


def _parse_structure_from_text(gen_text: str) -> Optional[Structure]:
    """复用 eval_sandbox 里的正则解析逻辑，但返回 Structure，方便查看 composition。"""
    import ast
    import re

    try:
        lat_match = re.search(
            r"lattice_matrix\s*=\s*(\[\s*\[.*?\]\s*\])",
            gen_text,
            re.DOTALL,
        )
        sp_match = re.search(
            r"species\s*=\s*(\[.*?\])",
            gen_text,
            re.DOTALL,
        )
        coord_match = re.search(
            r"coords\s*=\s*(\[\s*\[.*?\]\s*\])",
            gen_text,
            re.DOTALL,
        )
        if not (lat_match and sp_match and coord_match):
            return None

        lattice_matrix = ast.literal_eval(lat_match.group(1))
        species = ast.literal_eval(sp_match.group(1))
        coords = ast.literal_eval(coord_match.group(1))

        if len(species) < len(coords):
            species = species + [species[-1]] * (len(coords) - len(species))
        elif len(species) > len(coords):
            species = species[: len(coords)]

        return Structure(Lattice(lattice_matrix), species, coords)
    except Exception:
        return None


def _get_formulas(
    gt_struct: Optional[Structure],
    gen_struct: Optional[Structure],
) -> Tuple[str, str]:
    if gt_struct is None:
        gt = "<None>"
    else:
        gt = f"{gt_struct.composition.reduced_formula} | {gt_struct.composition}"

    if gen_struct is None:
        gen = "<None>"
    else:
        gen = f"{gen_struct.composition.reduced_formula} | {gen_struct.composition}"
    return gt, gen


def main() -> None:
    if not TEST_JSON.exists():
        raise FileNotFoundError(f"Test JSON not found: {TEST_JSON}")
    if not EVAL_JSON_B.exists():
        raise FileNotFoundError(f"Eval JSON (B) not found: {EVAL_JSON_B}")

    print("加载候选样本 idx ...")
    candidate_indices = _load_candidates(max_samples=5)
    print(f"候选样本 idx: {candidate_indices}")

    with TEST_JSON.open("r", encoding="utf-8") as f:
        test_data = json.load(f)

    print("\n加载 LoRA 模型（与 eval_sandbox 一致）...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(es.LORA_PATH),
        max_seq_length=es.MAX_SEQ_LENGTH,
        load_in_4bit=False,
        dtype=torch.bfloat16,
    )
    FastLanguageModel.for_inference(model)
    print("模型加载完成。开始逐条 sanity check：\n")

    for idx in candidate_indices:
        sample = test_data[idx]
        gt_struct = es.get_ground_truth_structure(sample.get("output", ""))
        prompt = _build_single_prompt(sample, tokenizer)
        gen_text = _run_single_inference(prompt, model, tokenizer)
        gen_struct = _parse_structure_from_text(gen_text)

        gt_formula, gen_formula = _get_formulas(gt_struct, gen_struct)
        metrics = es.evaluate_single_prediction(
            gen_text,
            gt_struct,
        )

        print("=" * 70)
        print(f"idx = {idx}")
        print(f"ground truth formula : {gt_formula}")
        print(f"generated formula    : {gen_formula}")
        print(f"metrics              : {metrics}")
        print()


if __name__ == "__main__":
    main()

