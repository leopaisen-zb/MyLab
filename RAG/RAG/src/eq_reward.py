"""
Eqv2-Lite 进程内预测器（奖励模型后端）：POSCAR 文本 -> 氢吸附自由能 ΔG_H (eV)。

本文件是 matgen_app/backend/eq_predict.py 的移植版，所有路径重指到 RAG 项目可达位置，
供 GRPO 强化学习的奖励函数（reward_dgh.py）进程内直接调用，无需 HTTP 服务。

加载的是 6 层 / 21.45M 的 standalone EquiformerV2 权重
（equiformer/checkpionts/best_standalone_equiformer_v2_model.pt），与论文正文
“4.69M/3 层 Eqv2-Lite”口径不同——见 plan 风险 1。

路径可用环境变量覆盖（GPU 训练机用）：
  EQ_SRC_PATH      standalone_equiformer_v2.py 所在目录（默认 ../../equiformer/src）
  EQ_CKPT_PATH     权重 .pt 路径
  EQ_NORM_STATS    归一化统计 json 路径
  MATGEN_DEVICE    cuda|mps|cpu|directml|auto（默认 auto；奖励模型建议 cpu，避免抢 7B 的显存）
"""

import os
import sys
import io as _io
import json
from pathlib import Path

import numpy as np
import torch

# ----------------------------------------------------------------------
# 路径解析：__file__ = mylab(1)/RAG/RAG/src/eq_reward.py
#   parent          -> src
#   parent.parent   -> RAG/RAG          (RAG 项目根，含 rag_datasets/)
#   parent.parent.parent.parent -> mylab(1)  (含 equiformer/)
# ----------------------------------------------------------------------
_HERE = Path(__file__).resolve()
RAG_ROOT = _HERE.parent.parent                      # mylab(1)/RAG/RAG
MYLAB_ROOT = _HERE.parent.parent.parent.parent      # mylab(1)

EQ_SRC = Path(os.environ.get("EQ_SRC_PATH", MYLAB_ROOT / "equiformer" / "src"))
EQ_CKPT = Path(os.environ.get(
    "EQ_CKPT_PATH",
    MYLAB_ROOT / "equiformer" / "checkpionts" / "best_standalone_equiformer_v2_model.pt",
))
NORM_STATS_PATH = Path(os.environ.get(
    "EQ_NORM_STATS",
    RAG_ROOT / "rag_datasets" / "custom_hydrogen" / "normalization_stats.json",
))

MAX_RADIUS = 12.0
MAX_NEIGHBORS = 20

_model = None
_norm_stats = None
_device = None


def get_device() -> torch.device:
    """自动检测设备，优先级 CUDA > MPS > DirectML > CPU；可用 MATGEN_DEVICE 覆盖。"""
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
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    try:
        import torch_directml
        dml_dev = torch_directml.device()
        if dml_dev is not None:
            return dml_dev
    except Exception:
        pass
    return torch.device("cpu")


def load_model():
    """惰性加载 StandaloneEquiformerV2 + 归一化统计，返回 (model, stats, device)。模块级缓存。"""
    global _model, _norm_stats, _device
    if _model is not None:
        return _model, _norm_stats, _device

    if str(EQ_SRC) not in sys.path:
        sys.path.insert(0, str(EQ_SRC))
    from standalone_equiformer_v2 import StandaloneEquiformerV2

    with open(NORM_STATS_PATH) as f:
        _norm_stats = json.load(f)

    _device = get_device()

    # 超参严格对齐 6 层 ckpt（照搬 matgen_app/backend/eq_predict.load_model）
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
    state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    _model.load_state_dict(state_dict, strict=False)
    _model.to(_device)
    _model.eval()

    return _model, _norm_stats, _device


def poscar_to_data(poscar_text: str):
    """POSCAR 文本 -> SimpleData。退化结构（半径内无邻接边）抛 ValueError。"""
    import ase.io
    from scipy.spatial import cKDTree

    if str(EQ_SRC) not in sys.path:
        sys.path.insert(0, str(EQ_SRC))
    from standalone_equiformer_v2 import SimpleData

    atoms = ase.io.read(_io.StringIO(poscar_text), format="vasp")
    positions = atoms.positions.astype(np.float32)
    atomic_numbers = atoms.numbers
    N = len(atoms)

    # 注意：非周期半径图（无 PBC 镜像），小晶胞可能少算近邻——见 plan 风险 3
    tree = cKDTree(positions)
    src_list, dst_list = [], []
    for i in range(N):
        dists, idxs = tree.query(
            positions[i], k=min(MAX_NEIGHBORS + 1, N), distance_upper_bound=MAX_RADIUS
        )
        for d, j in zip(np.atleast_1d(dists), np.atleast_1d(idxs)):
            if j == N or j == i:
                continue
            if d <= MAX_RADIUS:
                src_list.append(i)
                dst_list.append(int(j))

    if len(src_list) == 0:
        raise ValueError("No edges found within cutoff. Check POSCAR coordinates.")

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    pos_t = torch.tensor(positions, dtype=torch.float32)
    edge_distance = torch.norm(pos_t[edge_index[1]] - pos_t[edge_index[0]], dim=1)

    return SimpleData(
        pos=pos_t,
        atomic_numbers=torch.tensor(atomic_numbers, dtype=torch.long),
        batch=torch.zeros(N, dtype=torch.long),
        natoms=torch.tensor([N]),
        edge_index=edge_index,
        edge_distance=edge_distance,
        y=torch.zeros(1),
    )


def predict(poscar_text: str) -> dict:
    """POSCAR 文本 -> {'dg_h': float(eV), 'raw_normalized': float, 'num_atoms': int, 'device': str}。
    解析失败 / 退化结构会抛 ValueError（由调用方决定惩罚）。"""
    model, stats, device = load_model()
    data = poscar_to_data(poscar_text)

    for attr in ("pos", "atomic_numbers", "batch", "natoms", "edge_index", "edge_distance"):
        setattr(data, attr, getattr(data, attr).to(device))

    with torch.no_grad():
        raw = model(data)

    raw_val = raw.item()
    dg_h = raw_val * stats["target_std"] + stats["target_mean"]
    return {
        "dg_h": round(dg_h, 6),
        "raw_normalized": round(raw_val, 6),
        "num_atoms": int(data.natoms.item()),
        "device": str(device),
    }


def predict_batch(poscar_list):
    """一批 POSCAR -> list；每条成功返回 predict() 的 dict，失败（解析/退化/异常）返回 None。
    奖励函数据此把坏结构判为惩罚，不让单条坏结构拖垮整批。"""
    out = []
    for p in poscar_list:
        if not p:
            out.append(None)
            continue
        try:
            out.append(predict(p))
        except Exception:
            out.append(None)
    return out


if __name__ == "__main__":
    # 冒烟自测：从 stdin 读 POSCAR 文本
    txt = sys.stdin.read()
    print(predict(txt))
