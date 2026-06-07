"""
展示层中英映射 —— 仅用于 UI 友好展示。

设计原则（遵循"数据完整性优先"范式）：底层 DB / API / 状态机的英文枚举值是
**单一真相源**，保持不变；中文标签只是从英文源**派生**的展示层映射，绝不回写。
新增状态/字段时，必须同步在此补一条（由 tests/test_labels.py 的完整性不变式守护）。
"""

# 状态值 → 中文（键必须与 core/state_machine.StructureState 的取值一致）
STATE_LABELS = {
    "generated": "已生成",
    "rejected_precheck": "预检驳回",
    "predicted": "已预测",
    "filtered_in": "初筛通过",
    "filtered_out": "初筛淘汰",
    "dft_verified": "DFT已验证",
    "validated": "专家确认",
    "rejected": "已驳回",
    "exported_for_training": "已导出训练",
}

# 记录字段名 → 中文
FIELD_LABELS = {
    "uuid": "结构ID",
    "task_id": "任务ID",
    "status": "状态",
    "elements": "元素组成",
    "poscar": "结构文件",
    "predicted_dgH": "预测ΔG_H(eV)",
    "target_dgH": "目标ΔG_H(eV)",
    "dft_dg_h": "DFT ΔG_H(eV)",
    "dft_energy": "DFT能量(eV)",
    "dft_status": "DFT状态",
    "decision": "审查决定",
    "error": "错误信息",
    "created_at": "创建时间",
    "updated_at": "更新时间",
    "metadata": "元数据",
    "parsed_structure": "解析结构",
}


def state_label(value: str) -> str:
    """状态英文值 → 中文；未知值原样返回（不丢信息）。"""
    return STATE_LABELS.get(value, value)


def field_label(key: str) -> str:
    """字段名 → 中文；未知键原样返回。"""
    return FIELD_LABELS.get(key, key)


def localize_record(record: dict) -> dict:
    """结构记录 → 中文字段名 + 中文状态值（仅用于展示，不改原记录）。"""
    out = {}
    for k, v in record.items():
        out[field_label(k)] = state_label(v) if k == "status" else v
    return out


def localize_distribution(dist: dict) -> dict:
    """状态分布 {英文状态: 计数} → {中文状态: 计数}。"""
    return {state_label(k): v for k, v in dist.items()}
