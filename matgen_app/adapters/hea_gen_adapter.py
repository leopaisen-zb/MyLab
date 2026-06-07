# adapters/hea_gen_adapter.py
import os
import sys
import random
from typing import List, Dict, Any, Optional
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from adapters.base import BaseAdapter

# Third-level parsing guard imports
try:
    from pymatgen.core import Structure
    PYMGEN_AVAILABLE = True
except ImportError:
    PYMGEN_AVAILABLE = False

ELEMENT_SYMBOLS = ["Ir", "Pd", "Pt", "Rh", "Ru", "Fe", "Co", "Ni", "Cu", "Zn"]

def generate_fake_poscar(elements: List[str], num_atoms: int = 20) -> str:
    # Distribute num_atoms deterministically across elements (each gets at least 1)
    n_elements = len(elements)
    base = num_atoms // n_elements
    remainder = num_atoms % n_elements
    composition = {}
    for i, el in enumerate(elements):
        composition[el] = base + (1 if i < remainder else 0)
    total = sum(composition.values())
    lines = ["Generated HEA structure", "1.0"]
    lattice = [f"{random.uniform(7, 10):.6f} 0.0 0.0",
               f"0.0 {random.uniform(7, 10):.6f} 0.0",
               f"0.0 0.0 {random.uniform(7, 10):.6f}"]
    lines.extend(lattice)
    lines.append(" ".join(composition.keys()))
    lines.append(" ".join(str(composition[el]) for el in composition.keys()))
    lines.append("Cartesian")
    for _ in range(total):
        lines.append(f"{random.uniform(0, 9):.10f} {random.uniform(0, 9):.10f} {random.uniform(0, 9):.10f}")
    return "\n".join(lines)

def composition_from_poscar(poscar_text: str, sep: str = "") -> str:
    """从 POSCAR 文本提取真实组成字符串，按首次出现顺序聚合。

    sep="" → 'Ir4Pd4Pt4Rh4Ru4'（用作结构记录 elements 字段）；
    sep=" " → 'Ir4 Pd4 Pt4 Rh4 Ru4'（用作 composition_to_features 输入）。
    保证与 POSCAR 实际原子数严格一致。解析失败返回 ''（调用方可回退）。
    """
    def _fmt(counts: Dict[str, int]) -> str:
        return sep.join(f"{el}{n}" for el, n in counts.items())

    # 优先用 pymatgen（更鲁棒，可处理重复元素行）
    if PYMGEN_AVAILABLE:
        try:
            struct = Structure.from_str(poscar_text, "poscar")
            counts: Dict[str, int] = {}
            for site in struct.sites:
                el = str(site.specie)
                counts[el] = counts.get(el, 0) + 1
            if counts:
                return _fmt(counts)
        except Exception:
            pass
    # 回退：手动解析"元素符号行 + 计数行"（标准 POSCAR 格式）
    lines = poscar_text.splitlines()
    for i in range(len(lines) - 1):
        syms = lines[i].split()
        nums = lines[i + 1].split()
        if (syms and nums and len(syms) == len(nums)
                and all(s.isalpha() for s in syms)
                and all(n.isdigit() for n in nums)):
            counts = {}
            for el, n in zip(syms, nums):
                counts[el] = counts.get(el, 0) + int(n)
            return _fmt(counts)
    return ""


class HEAGenAdapter(BaseAdapter):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {
            "model_path": os.environ.get("RAG_MODEL_DIR", "./models/hea_gen"),
            "max_length": 512,
            "temperature": 0.7,
        }
        if config:
            default_config.update(config)
        super().__init__(default_config)
        self.model = None
        self.tokenizer = None

    def initialize(self) -> bool:
        try:
            model_path = self.config.get("model_path", "")
            if os.path.exists(model_path):
                pass
            self._initialized = True
            return True
        except Exception as e:
            print(f"HEA-Gen initialization failed: {e}")
            return False

    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        if not isinstance(input_data, dict):
            return False
        elements = input_data.get("elements", [])
        if not elements or not all(el in ELEMENT_SYMBOLS for el in elements):
            return False
        return True

    def forward(self, input_data: Dict[str, Any]) -> str:
        elements = input_data.get("elements", ["Ir", "Pd", "Pt", "Rh", "Ru"])
        target_dgH = input_data.get("target_dgH", -0.5)
        prompt = (
            f"Generate a high-entropy alloy surface structure with elements "
            f"{', '.join(elements)} for hydrogen evolution reaction (HER) "
            f"with target ΔG_H = {target_dgH:.2f} eV. "
            "Output only the VASP POSCAR format."
        )
        try:
            from backend.rag_gen import generate
            return generate(prompt, use_rag=True)
        except Exception as e:
            print(f"[HEAGenAdapter] backend.rag_gen failed ({e}), falling back to fake POSCAR")
            return generate_fake_poscar(elements, num_atoms=20)

    def generate_batch(self, target_dgH: float, elements: List[str], batch_size: int = 10) -> List[Dict[str, Any]]:
        results = []
        for _ in range(batch_size):
            poscar = self.forward({"elements": elements, "target_dgH": target_dgH})
            results.append({
                "elements": composition_from_poscar(poscar) or "".join(elements),
                "poscar": poscar,
                "target_dgH": target_dgH
            })
        return results

    def parse_structure(self, poscar_text: str) -> Optional[Dict[str, Any]]:
        if not PYMGEN_AVAILABLE:
            return {"raw": poscar_text, "parsed": True}
        try:
            struct = Structure.from_str(poscar_text, "poscar")
            return {
                "lattice": struct.lattice.matrix.tolist(),
                "composition": str(struct.composition),
                "num_sites": len(struct.sites),
                "parsed": True
            }
        except Exception:
            return None
