# ---------- src/equiformer_v2/nets/tabular_branch.py ----------
from __future__ import annotations
import json
from typing import List, Optional, Dict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

class TabularStandardizer:
    def __init__(self):
        self.mean_: Dict[str, float] = {}
        self.std_: Dict[str, float] = {}

    def fit(self, df: pd.DataFrame, cols: List[str]):
        self.mean_ = {c: float(df[c].mean()) for c in cols}
        self.std_  = {c: float(df[c].std(ddof=0)) for c in cols}
        for c in cols:
            if self.std_[c] == 0 or np.isnan(self.std_[c]):
                self.std_[c] = 1.0

    def transform(self, x: pd.DataFrame | np.ndarray | torch.Tensor, cols: Optional[List[str]]=None):
        if isinstance(x, pd.DataFrame):
            cols = cols or list(self.mean_.keys())
            arr = x[cols].to_numpy(dtype=np.float32)
        elif isinstance(x, np.ndarray):
            arr = x.astype(np.float32)
        elif isinstance(x, torch.Tensor):
            return (x - torch.tensor(list(self.mean_.values()), device=x.device, dtype=x.dtype)) / \
                   torch.tensor(list(self.std_.values()), device=x.device, dtype=x.dtype)
        else:
            raise TypeError("Unsupported type for transform.")
        mean = np.array([self.mean_[c] for c in self.mean_.keys()], dtype=np.float32)
        std  = np.array([self.std_[c]  for c in self.std_.keys()], dtype=np.float32)
        return (arr - mean) / std

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"mean": self.mean_, "std": self.std_}, f, ensure_ascii=False, indent=2)

    def load(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        self.mean_ = {k: float(v) for k, v in obj["mean"].items()}
        self.std_  = {k: float(v) for k, v in obj["std"].items()}

class TabularMLP(nn.Module):
    def __init__(self, in_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # (B, 64)

class FusionHead(nn.Module):
    """
    支持两种融合：
    - concat: [h_struct; h_feat] -> MLP -> y
    - gate:   g=σ(MLP(h_feat))；h = h_struct * g + h_feat -> MLP -> y
    若 h_struct 为 None，则使用"标量回退"：将 y_struct 通过 Linear(1,64) 投影得到 64 维再融合。
    """
    def __init__(self, struct_dim: int = 256, fusion: str = "concat", use_struct_scalar_fallback: bool = True):
        super().__init__()
        assert fusion in ("concat", "gate")
        self.fusion = fusion
        self.use_struct_scalar_fallback = use_struct_scalar_fallback
        self.struct_proj_from_scalar = nn.Linear(1, 64)
        if fusion == "concat":
            self.mlp = nn.Sequential(
                nn.Linear(struct_dim + 64, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(128, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(64, 1),
            )
        else:  # gate
            self.gate = nn.Sequential(nn.Linear(64, 1))
            self.mlp = nn.Sequential(
                nn.Linear(struct_dim, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(64, 1),
            )

    def forward(self,
                h_feat: torch.Tensor,
                h_struct: Optional[torch.Tensor] = None,
                y_struct_scalar: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = h_feat.size(0)
        if h_struct is None:
            if not self.use_struct_scalar_fallback or y_struct_scalar is None:
                raise ValueError("h_struct and y_struct_scalar are both None.")
            h_struct = self.struct_proj_from_scalar(y_struct_scalar.view(B, 1))  # (B,64)
            if self.fusion == "concat":
                x = torch.cat([h_struct, h_feat], dim=-1)  # (B,128)
                # 动态调整MLP第一层的输入维度
                if x.size(-1) != self.mlp[0].in_features:
                    self.mlp[0] = nn.Linear(x.size(-1), 128).to(x.device)
                return self.mlp(x)
            else:
                g = torch.sigmoid(self.gate(h_feat))  # (B,1)
                h = h_struct * g + h_feat  # broadcast (B,64)
                # 对于gate模式，需要将h投影到struct_dim
                if h.size(-1) != 256:  # 默认struct_dim=256
                    h = F.pad(h, (0, 256 - h.size(-1)))
                return self.mlp(h)

        # 有结构表示的标准路径
        struct_dim = h_struct.size(-1)
        if self.fusion == "concat":
            # 若 struct_dim != 256 也可工作，因为 Linear 的 in_features 已固定在 __init__；保持默认 256 即可
            if h_struct.size(-1) != self.mlp[0].in_features - 64:
                # 动态适配：重新初始化第一层
                in_features = h_struct.size(-1) + 64
                self.mlp[0] = nn.Linear(in_features, 128).to(h_struct.device)
            x = torch.cat([h_struct, h_feat], dim=-1)
            return self.mlp(x)
        else:
            if self.mlp[0].in_features != struct_dim:
                self.mlp[0] = nn.Linear(struct_dim, 64).to(h_struct.device)
            g = torch.sigmoid(self.gate(h_feat))
            # 将 h_feat pad/投影到 struct_dim 再做门控融合
            if h_feat.size(-1) != struct_dim:
                # 简单 pad；也可换线性投影
                h_feat = F.pad(h_feat, (0, struct_dim - h_feat.size(-1)))
            h = h_struct * g + h_feat
            return self.mlp(h)
# ---------- end of file ----------
