"""候选结构横向对比视图"""
import ase.io
import io
import numpy as np
from typing import Optional

def parse_poscar(poscar_text: str) -> Optional[dict]:
    """解析 POSCAR，返回结构参数字典。"""
    try:
        atoms = ase.io.read(io.StringIO(poscar_text), format="vasp")
        lattice = atoms.cell.array
        a, b, c = np.linalg.norm(lattice, axis=1)
        alpha = np.degrees(np.arccos(np.dot(lattice[1], lattice[2]) / (b * c)))
        beta = np.degrees(np.arccos(np.dot(lattice[0], lattice[2]) / (a * c)))
        gamma = np.degrees(np.arccos(np.dot(lattice[0], lattice[1]) / (a * b)))
        elem_counts = {}
        for n in atoms.numbers:
            elem_counts[n] = elem_counts.get(n, 0) + 1
        from ase.data import atomic_numbers
        elem_names = {z: atomic_numbers[z] for z in elem_counts}
        from ase.neighborlist import neighbor_list
        n1, n2, d = neighbor_list('ijd', atoms, cutoff=10.0)
        min_dist = float(d.min()) if len(d) > 0 else float('inf')
        return {
            "a": round(a, 4),
            "b": round(b, 4),
            "c": round(c, 4),
            "alpha": round(alpha, 2),
            "beta": round(beta, 2),
            "gamma": round(gamma, 2),
            "volume": round(atoms.get_volume(), 2),
            "density": round(atoms.get_density(), 4),
            "num_atoms": len(atoms),
            "elements": {elem_names[z]: c for z, c in elem_counts.items()},
            "min_distance": round(min_dist, 3),
        }
    except Exception:
        return None

def compare_two(poscar_a: str, poscar_b: str) -> dict:
    """对比两个 POSCAR，返回差异字典。"""
    info_a = parse_poscar(poscar_a)
    info_b = parse_poscar(poscar_b)
    if info_a is None or info_b is None:
        return {"error": "无法解析 POSCAR"}
    diff = {}
    for key in ["a", "b", "c", "alpha", "beta", "gamma", "volume", "density", "num_atoms", "min_distance"]:
        if key in info_a and key in info_b:
            diff[key] = round(info_b[key] - info_a[key], 4)
    return {
        "a_info": info_a,
        "b_info": info_b,
        "diff": diff,
    }
