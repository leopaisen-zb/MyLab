"""
Eqv2-Lite predictor: POSCAR text -> delta_G_H (eV)
"""

import sys
from pathlib import Path
import json
import torch
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # mylab(1)/
EQ_SRC = PROJECT_ROOT / "equiformer" / "src"
EQ_CKPT = PROJECT_ROOT / "equiformer" / "checkpionts" / "best_standalone_equiformer_v2_model.pt"
NORM_STATS_PATH = PROJECT_ROOT / "RAG" / "RAG" / "rag_datasets" / "custom_hydrogen" / "normalization_stats.json"

MAX_RADIUS = 12.0
MAX_NEIGHBORS = 20

_model = None
_norm_stats = None
_device = None


def get_device() -> torch.device:
    """
    自动检测可用设备，优先级：CUDA > MPS > DirectML > CPU。
    DirectML 通过 torch_directml 包支持 AMD/Intel/NVIDIA GPU。
    """
    import os
    override = os.environ.get("MATGEN_DEVICE", "auto")
    if override == "cuda":
        return torch.device("cuda")
    if override == "mps":
        return torch.device("mps")
    if override == "cpu":
        return torch.device("cpu")
    if override == "directml":
        import torch_directml
        return torch_directml.device()
    # auto: 依次检测 CUDA, MPS, DirectML
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    # 检查 DirectML (torch_directml 包)
    try:
        import torch_directml
        dml_dev = torch_directml.device()
        if dml_dev is not None:
            return dml_dev
    except (ImportError, Exception):
        pass
    return torch.device("cpu")


def load_model():
    """Lazy-load StandaloneEquiformerV2 + normalization stats. Returns (model, stats, device)."""
    global _model, _norm_stats, _device
    if _model is not None:
        return _model, _norm_stats, _device

    sys.path.insert(0, str(EQ_SRC))
    from standalone_equiformer_v2 import StandaloneEquiformerV2

    with open(NORM_STATS_PATH) as f:
        _norm_stats = json.load(f)

    _device = get_device()

    # Hyperparameters match the checkpoint saved by equiformer/src/train_equiformer.py
    # (verified from evaluate_equiformer_v2.py create_model() and training config)
    _model = StandaloneEquiformerV2(
        max_radius=MAX_RADIUS,
        max_neighbors=MAX_NEIGHBORS,
        max_num_elements=90,
        num_layers=6,
        sphere_channels=128,
        attn_hidden_channels=64,
        num_heads=8,
        attn_alpha_channels=32,
        attn_value_channels=16,
        ffn_hidden_channels=256,
        lmax_list=[4],
        mmax_list=[2],
        grid_resolution=18,
        edge_channels=128,
        use_atom_edge_embedding=True,
        share_atom_edge_embedding=False,
        alpha_drop=0.1,
        drop_path_rate=0.05,
        proj_drop=0.0,
    )

    ckpt = torch.load(str(EQ_CKPT), map_location="cpu", weights_only=False)
    # Checkpoint is a raw state_dict (saved via torch.save(model.state_dict(), ...))
    state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    _model.load_state_dict(state_dict, strict=False)
    _model.to(_device)
    _model.eval()

    return _model, _norm_stats, _device


def poscar_to_data(poscar_text: str):
    """Parse POSCAR text -> SimpleData for model inference."""
    import ase.io
    import io as _io
    from scipy.spatial import cKDTree

    sys.path.insert(0, str(EQ_SRC))
    from standalone_equiformer_v2 import SimpleData

    atoms = ase.io.read(_io.StringIO(poscar_text), format="vasp")
    positions = atoms.positions.astype(np.float32)  # (N, 3)
    atomic_numbers = atoms.numbers                   # (N,)
    N = len(atoms)

    tree = cKDTree(positions)
    src_list, dst_list = [], []
    for i in range(N):
        dists, idxs = tree.query(positions[i], k=min(MAX_NEIGHBORS + 1, N), distance_upper_bound=MAX_RADIUS)
        for d, j in zip(dists, idxs):
            if j == N:
                continue
            if j == i:
                continue
            if d <= MAX_RADIUS:
                src_list.append(i)
                dst_list.append(j)

    if len(src_list) == 0:
        raise ValueError("No edges found within cutoff. Check POSCAR coordinates.")

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    pos_t = torch.tensor(positions, dtype=torch.float32)
    edge_distance = torch.norm(pos_t[edge_index[1]] - pos_t[edge_index[0]], dim=1)

    data = SimpleData(
        pos=pos_t,
        atomic_numbers=torch.tensor(atomic_numbers, dtype=torch.long),
        batch=torch.zeros(N, dtype=torch.long),
        natoms=torch.tensor([N]),
        edge_index=edge_index,
        edge_distance=edge_distance,
        y=torch.zeros(1),
    )
    return data


def predict(poscar_text: str) -> dict:
    """
    Main entry point: POSCAR text -> {'dg_h': float, 'num_atoms': int, 'device': str}
    Raises ValueError if POSCAR cannot be parsed or model fails.
    """
    model, stats, device = load_model()

    data = poscar_to_data(poscar_text)

    data.pos = data.pos.to(device)
    data.atomic_numbers = data.atomic_numbers.to(device)
    data.batch = data.batch.to(device)
    data.natoms = data.natoms.to(device)
    data.edge_index = data.edge_index.to(device)
    data.edge_distance = data.edge_distance.to(device)

    with torch.no_grad():
        raw = model(data)

    raw_val = raw.item()
    dg_h = raw_val * stats["target_std"] + stats["target_mean"]

    return {
        "dg_h": round(dg_h, 6),
        "raw_normalized": round(raw_val, 6),
        "num_atoms": data.natoms.item(),
        "device": str(device),
    }
