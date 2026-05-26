r"""
将 Text-to-3D (VASP POSCAR 文本) 数据集转换为 Text-to-Code (Python 晶体构建函数) 数据集。

- 输入:
  - D:\mylab\RAG\RAG\data\rag_data\text2struct_rag\dataset_train_rag.json
  - D:\mylab\RAG\RAG\data\rag_data\text2struct_rag\dataset_test_rag.json
- 输出:
  - D:\mylab\RAG\RAG\data\rag_data\text2struct_rag\dataset_train_rag_code.json
  - D:\mylab\RAG\RAG\data\rag_data\text2struct_rag\dataset_test_rag_code.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pymatgen.core import Structure
from tqdm import tqdm

# ============================================================
# 路径配置
# ============================================================
DATA_DIR = Path(r"D:\mylab\RAG\RAG\data\rag_data\text_rag")
# 兼容旧目录名：text2struct_rag
if not DATA_DIR.exists():
    legacy_dir = Path(r"D:\mylab\RAG\RAG\data\rag_data\text2struct_rag")
    if legacy_dir.exists():
        DATA_DIR = legacy_dir

TRAIN_INPUT = DATA_DIR / "dataset_train_rag.json"
TEST_INPUT = DATA_DIR / "dataset_test_rag.json"
TRAIN_OUTPUT = DATA_DIR / "dataset_train_rag_code.json"
TEST_OUTPUT = DATA_DIR / "dataset_test_rag_code.json"

RESULT_DIR = Path(r"D:\mylab\RAG\RAG\result")
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def _fmt_num(val: float) -> str:
    """将浮点数按 round(val, 4) 规则压缩为字符串。"""
    v = round(float(val), 4)
    if abs(v) < 1e-12:
        v = 0.0
    return f"{v:.4f}".rstrip("0").rstrip(".") or "0"


def _format_matrix(rows: list[list[float]], indent: str = "        ") -> str:
    """将二维浮点数组格式化为 Python 列表文本。"""
    lines: list[str] = []
    for row in rows:
        row_txt = ", ".join(_fmt_num(x) for x in row)
        lines.append(f"{indent}[{row_txt}],")
    return "\n".join(lines)


def _format_coords(rows: list[list[float]], indent: str = "        ") -> str:
    """将分数坐标数组格式化为 Python 列表文本。"""
    lines: list[str] = []
    for row in rows:
        row_txt = ", ".join(_fmt_num(x) for x in row)
        lines.append(f"{indent}[{row_txt}],")
    return "\n".join(lines)


def vasp_to_python_code(vasp_text: str) -> str:
    """
    将 VASP POSCAR 文本转换为标准 Python 函数代码字符串。

    目标模板：
    def generate_structure():
        from pymatgen.core import Lattice, Structure

        lattice_matrix = [
            [x1, y1, z1],
            [x2, y2, z2],
            [x3, y3, z3]
        ]
        lattice = Lattice(lattice_matrix)

        species = ['Elem1', 'Elem2', ...]

        coords = [
            [u1, v1, w1],
            [u2, v2, w2],
            ...
        ]

        structure = Structure(lattice, species, coords)
        return structure
    """
    structure = Structure.from_str(vasp_text, fmt="poscar")

    lattice_matrix: list[list[float]] = structure.lattice.matrix.tolist()
    species: list[str] = [str(s) for s in structure.species]
    frac_coords: list[list[float]] = structure.frac_coords.tolist()

    lattice_block = _format_matrix(lattice_matrix, indent="        ").rstrip(",")
    coords_block = _format_coords(frac_coords, indent="        ").rstrip(",")
    species_str = repr(species)

    code = (
        "def generate_structure():\n"
        "    from pymatgen.core import Lattice, Structure\n"
        "    \n"
        "    lattice_matrix = [\n"
        f"{lattice_block}\n"
        "    ]\n"
        "    lattice = Lattice(lattice_matrix)\n"
        "    \n"
        f"    species = {species_str}\n"
        "    \n"
        "    coords = [\n"
        f"{coords_block}\n"
        "    ]\n"
        "    \n"
        "    structure = Structure(lattice, species, coords)\n"
        "    return structure"
    )
    return code


def process_dataset(input_path: Path, output_path: Path) -> dict[str, int]:
    """
    读取 JSON 数据集并将 output 字段从 VASP 文本转为 Python 代码。

    - 转换成功：替换 output
    - 转换失败：打印 Warning，并剔除该条数据
    """
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    with input_path.open("r", encoding="utf-8") as f:
        raw_data = json.load(f)

    if not isinstance(raw_data, list):
        raise ValueError(f"数据格式错误，期望 list[dict]，实际为: {type(raw_data)}")

    cleaned_data: list[dict[str, Any]] = []
    dropped = 0

    for idx, item in enumerate(tqdm(raw_data, desc=f"Converting {input_path.name}", ncols=100), start=1):
        if not isinstance(item, dict):
            print(f"[Warning] {input_path.name} 第 {idx} 条不是字典，已剔除。")
            dropped += 1
            continue

        vasp_text = item.get("output", "")
        if not isinstance(vasp_text, str) or not vasp_text.strip():
            print(f"[Warning] {input_path.name} 第 {idx} 条 output 为空或非字符串，已剔除。")
            dropped += 1
            continue

        try:
            code_text = vasp_to_python_code(vasp_text)
        except Exception as exc:
            print(f"[Warning] {input_path.name} 第 {idx} 条 VASP 解析失败，已剔除。原因: {exc}")
            dropped += 1
            continue

        new_item = dict(item)
        new_item["output"] = code_text
        cleaned_data.append(new_item)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)

    stats = {
        "input_count": len(raw_data),
        "output_count": len(cleaned_data),
        "dropped_count": dropped,
    }

    print(
        f"[Done] {input_path.name} -> {output_path.name} | "
        f"输入: {stats['input_count']} | 输出: {stats['output_count']} | 剔除: {stats['dropped_count']}"
    )
    return stats


def main() -> None:
    train_stats = process_dataset(TRAIN_INPUT, TRAIN_OUTPUT)
    test_stats = process_dataset(TEST_INPUT, TEST_OUTPUT)

    report = {
        "train": train_stats,
        "test": test_stats,
        "train_input": str(TRAIN_INPUT),
        "train_output": str(TRAIN_OUTPUT),
        "test_input": str(TEST_INPUT),
        "test_output": str(TEST_OUTPUT),
    }

    report_path = RESULT_DIR / "convert_to_code_dataset_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[Report] 处理报告已保存: {report_path}")


if __name__ == "__main__":
    main()
