"""
GRPO 组合奖励函数：格式有效性 bonus + 物理接近度（Eqv2-Lite 预测 ΔG_H 对齐目标）。

进程内直接调 eq_reward（无 HTTP）。trl 0.24 GRPOTrainer 期望的奖励签名：
    reward_fn(prompts, completions, **kwargs) -> list[float]
数据集的额外列（target_dgh）由 trl 原样以 kwargs 转发，与 completions 对齐。

格式判定的两个工具函数 extract_pure_vasp / check_validity 从 src/eval_metrics.py
**逐字复制**而来（保持与评估同口径）。之所以不直接 import eval_metrics：该模块顶部
`import unsloth`，会把 unsloth 依赖拉进奖励路径，导致无法在纯 equiformer_v2 环境本地验证。
两函数为纯字符串逻辑、无外部依赖，复制零风险。
"""

import math
from typing import List


# ============================================================
# 以下两函数逐字复制自 src/eval_metrics.py（extract_pure_vasp:197 / check_validity:225）
# 修改请同步两处，保持奖励与评估口径一致。
# ============================================================
def _split_non_empty_lines(text: str) -> List[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def extract_pure_vasp(raw_text: str) -> str:
    """针对大模型“话痨”行为剥壳，提取纯净 VASP 文本。"""
    if "```" in raw_text:
        parts = raw_text.split("```")
        if len(parts) >= 2:
            content = parts[1]
            stripped = content.lstrip()
            if stripped.lower().startswith("vasp"):
                stripped = stripped[4:]
            return stripped.strip()

    lines = raw_text.splitlines()
    for i, line in enumerate(lines):
        if line.strip().startswith("1.000"):
            start_idx = max(0, i - 1)
            return "\n".join(lines[start_idx:]).strip()

    return raw_text.strip()


def check_validity(gen_text: str) -> bool:
    """检查是否符合基本 VASP POSCAR 格式（≥8 行 / 3 行晶格 / Cartesian|Direct / ≥1 坐标行）。"""
    try:
        lines = _split_non_empty_lines(gen_text)
        if len(lines) < 8:
            return False

        lattice_ok = 0
        for i in range(2, 5):
            if i >= len(lines):
                break
            tokens = lines[i].split()
            if len(tokens) < 3:
                continue
            try:
                _ = [float(t) for t in tokens[:3]]
                lattice_ok += 1
            except Exception:
                continue
        if lattice_ok != 3:
            return False

        coord_type_idx = None
        for idx, line in enumerate(lines):
            lower = line.lower()
            if lower == "cartesian" or lower == "direct":
                coord_type_idx = idx
                break
        if coord_type_idx is None:
            return False

        has_coord_line = False
        for line in lines[coord_type_idx + 1:]:
            tokens = line.split()
            if len(tokens) < 3:
                continue
            try:
                _ = [float(t) for t in tokens[:3]]
                has_coord_line = True
                break
            except Exception:
                continue

        return has_coord_line
    except Exception:
        return False


# ============================================================
# 组合奖励
# ============================================================
def _completion_to_text(c) -> str:
    """trl 可能把 completion 传成 str，或 [{'role','content'}] 的消息列表，统一成纯文本。"""
    if isinstance(c, str):
        return c
    if isinstance(c, list) and c and isinstance(c[0], dict):
        return c[0].get("content", "")
    return str(c)


def make_reward_fn(tau: float = 0.05, fmt_bonus: float = 0.1,
                   degenerate_penalty: float = -1.0):
    """构造 trl 签名的奖励函数。

    评分：
      格式合规  -> + fmt_bonus
      可打分    -> + exp(-|dg_h_pred - target| / tau)   (∈(0,1]，完美匹配=1)
      不合规/退化/Eqv2-Lite 打分失败 -> degenerate_penalty（覆盖上面，最低分）

    tau 是关键旋钮（eV）：太小信号稀疏、太大不区分。起步 0.05，可扫 {0.03,0.05,0.1}。
    """
    # 延迟 import：本模块在纯 equiformer_v2 环境（无 unsloth）也能 import，
    # eq_reward 仅在真正调用奖励时才加载模型。
    from eq_reward import predict_batch

    def reward_dgh(prompts=None, completions=None, **kwargs) -> List[float]:
        targets = kwargs["target_dgh"]  # list[float]，与 completions 对齐（trl 转发）
        comps = completions if completions is not None else []

        # 1) 剥壳 + 格式判定
        poscars, valid_flags = [], []
        for c in comps:
            text = _completion_to_text(c)
            vasp = extract_pure_vasp(text)
            ok = check_validity(vasp)
            valid_flags.append(ok)
            poscars.append(vasp if ok else "")  # 空串 -> predict_batch 直接返回 None

        # 2) 一次性批量打分（坏结构返回 None）
        preds = predict_batch(poscars)

        # 3) 组合奖励
        rewards: List[float] = []
        for ok, pred, tgt in zip(valid_flags, preds, targets):
            if (not ok) or (pred is None) or (pred.get("dg_h") is None):
                rewards.append(degenerate_penalty)
                continue
            err = abs(float(pred["dg_h"]) - float(tgt))
            r = fmt_bonus + math.exp(-err / tau)
            rewards.append(r)
        return rewards

    return reward_dgh
