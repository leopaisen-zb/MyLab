#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用Standalone EquiformerV2模型对训练集和测试集进行预测，
筛选ΔGH接近0 eV的优质HER催化材料，输出商用级别的筛选结果表格。
"""

import os
import sys
import json
import pickle
import io
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader

# 添加项目路径
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from standalone_equiformer_v2 import (
    StandaloneEquiformerV2,
    HydrogenDataset,
    custom_collate_fn,
    SimpleData,
)

# 确保 pickle 能正确反序列化
sys.modules['__main__'] = type(sys)('__main__')
sys.modules['__main__'].SimpleData = SimpleData


def load_normalization_stats(stats_file):
    with open(stats_file, 'r') as f:
        return json.load(f)


def denormalize(values, stats):
    return values * stats['target_std'] + stats['target_mean']


@torch.no_grad()
def predict_all(model, dataloader, device, stats):
    """对整个数据集进行预测，返回 (predictions, true_values, sample_info_list)"""
    model.eval()
    preds_list, true_list, info_list = [], [], []
    sample_idx = 0

    for batch in dataloader:
        batch.pos = batch.pos.to(device)
        batch.atomic_numbers = batch.atomic_numbers.to(device)
        batch.edge_index = batch.edge_index.to(device)
        batch.edge_distance = batch.edge_distance.to(device)
        batch.y = batch.y.to(device)
        batch.batch = batch.batch.to(device)
        batch.natoms = batch.natoms.to(device)

        outputs = model(batch)
        denorm_preds = denormalize(outputs.cpu().numpy(), stats)
        true_vals = batch.y.cpu().numpy()

        natoms_np = batch.natoms.cpu().numpy()

        for i in range(len(natoms_np)):
            preds_list.append(float(denorm_preds[i]))
            true_list.append(float(true_vals[i]))
            # 收集原子信息
            start = int(np.sum(natoms_np[:i]))
            end = start + int(natoms_np[i])
            atom_nums = batch.atomic_numbers.cpu().numpy()[start:end]
            info_list.append({
                'sample_idx': sample_idx,
                'natoms': int(natoms_np[i]),
                'atomic_numbers': atom_nums.tolist(),
            })
            sample_idx += 1

    return np.array(preds_list), np.array(true_list), info_list


# 元素符号表
ELEMENT_SYMBOLS = [
    '', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
    'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
    'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
    'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
    'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn',
    'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U',
]


def get_composition(atomic_numbers):
    """从原子序数列表获取化学组成式"""
    from collections import Counter
    counts = Counter(atomic_numbers)
    # 按原子序数排序
    sorted_elems = sorted(counts.items())
    formula_parts = []
    for z, n in sorted_elems:
        sym = ELEMENT_SYMBOLS[z] if z < len(ELEMENT_SYMBOLS) else f"Z{z}"
        if n == 1:
            formula_parts.append(sym)
        else:
            formula_parts.append(f"{sym}{n}")
    return ''.join(formula_parts)


def get_unique_elements(atomic_numbers):
    """获取去重的元素符号列表"""
    unique_z = sorted(set(atomic_numbers))
    return [ELEMENT_SYMBOLS[z] if z < len(ELEMENT_SYMBOLS) else f"Z{z}" for z in unique_z]


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 路径设置
    data_dir = project_root / "datasets" / "custom_hydrogen"
    stats_path = data_dir / "normalization_stats.json"
    # standalone 模型 checkpoint
    model_path = project_root / "checkpionts" / "best_standalone_equiformer_v2_model.pt"

    if not model_path.exists():
        print(f"[WARN] Standalone checkpoint not found at {model_path}")
        print("       Falling back to enhanced checkpoint...")
        model_path = project_root / "checkpionts" / "best_enhanced_equiformer_v2.pt"

    output_dir = Path(r"d:\mylab\Jiang")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载归一化统计
    stats = load_normalization_stats(str(stats_path))
    print(f"Normalization: mean={stats['target_mean']:.4f}, std={stats['target_std']:.4f}")

    # 加载 checkpoint
    print(f"Loading checkpoint: {model_path}")
    checkpoint = torch.load(str(model_path), map_location=device, weights_only=False)

    # 判断是 standalone 还是 enhanced checkpoint
    is_standalone = 'standalone' in str(model_path).lower()

    # 判断 checkpoint 格式: raw state_dict vs wrapped dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        config = checkpoint.get('config', {})
        val_mae = checkpoint.get('val_mae', 'N/A')
    else:
        # Raw OrderedDict state_dict
        state_dict = checkpoint
        config = {}
        val_mae = 'N/A'

    if is_standalone:
        from standalone_equiformer_v2 import StandaloneEquiformerV2
        model = StandaloneEquiformerV2(
            max_radius=config.get('max_radius', 12.0),
            max_neighbors=config.get('max_neighbors', 20),
            max_num_elements=config.get('max_num_elements', 90),
            num_layers=config.get('num_layers', 6),
            sphere_channels=config.get('sphere_channels', 128),
            attn_hidden_channels=config.get('attn_hidden_channels', 64),
            num_heads=config.get('num_heads', 8),
            attn_alpha_channels=config.get('attn_alpha_channels', 32),
            attn_value_channels=config.get('attn_value_channels', 16),
            ffn_hidden_channels=config.get('ffn_hidden_channels', 256),
            lmax_list=config.get('lmax_list', [4]),
            mmax_list=config.get('mmax_list', [2]),
            grid_resolution=config.get('grid_resolution', 18),
            edge_channels=config.get('edge_channels', 128),
            use_atom_edge_embedding=config.get('use_atom_edge_embedding', True),
            share_atom_edge_embedding=config.get('share_atom_edge_embedding', False),
            alpha_drop=0.0,  # inference: 关闭dropout
            drop_path_rate=0.0,
            proj_drop=0.0,
            norm_type=config.get('norm_type', 'layer_norm_sh'),
        ).to(device)
    else:
        from enhanced_equiformer_v2 import EnhancedEquiformerV2
        model = EnhancedEquiformerV2(
            max_radius=config.get('max_radius', 12.0),
            max_neighbors=config.get('max_neighbors', 20),
            max_num_elements=config.get('max_num_elements', 90),
            num_layers=config.get('num_layers', 3),
            sphere_channels=config.get('sphere_channels', 64),
            attn_hidden_channels=config.get('attn_hidden_channels', 32),
            num_heads=config.get('num_heads', 4),
            attn_alpha_channels=config.get('attn_alpha_channels', 32),
            attn_value_channels=config.get('attn_value_channels', 8),
            ffn_hidden_channels=config.get('ffn_hidden_channels', 64),
            lmax_list=config.get('lmax_list', [4]),
            mmax_list=config.get('mmax_list', [2]),
            grid_resolution=config.get('grid_resolution', 12),
            edge_channels=config.get('edge_channels', 64),
            use_atom_edge_embedding=config.get('use_atom_edge_embedding', True),
            share_atom_edge_embedding=config.get('share_atom_edge_embedding', False),
            alpha_drop=0.0,
            drop_path_rate=0.0,
            proj_drop=0.0,
            norm_type=config.get('norm_type', 'layer_norm_sh'),
        ).to(device)

    model.load_state_dict(state_dict)
    model.eval()
    print(f"Model loaded. Val MAE from training: {val_mae}")

    # ==================== 加载训练集 & 测试集 ====================
    batch_size = 16

    datasets_to_scan = {}
    for split_name in ['train', 'test']:
        lmdb_path = data_dir / f"{split_name}.lmdb"
        if lmdb_path.exists():
            ds = HydrogenDataset(str(lmdb_path))
            loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                                collate_fn=custom_collate_fn, num_workers=0)
            datasets_to_scan[split_name] = (ds, loader)
        else:
            print(f"[WARN] {lmdb_path} not found, skipping {split_name} set.")

    # ==================== 运行预测 ====================
    all_rows = []
    for split_name, (ds, loader) in datasets_to_scan.items():
        print(f"\n{'='*60}")
        print(f"Running predictions on {split_name} set ({len(ds)} samples)...")
        preds, trues, infos = predict_all(model, loader, device, stats)

        # 统计
        err = np.abs(preds - trues)
        mae = np.mean(err)
        rmse = np.sqrt(np.mean((preds - trues) ** 2))
        print(f"  {split_name} MAE = {mae:.4f} eV, RMSE = {rmse:.4f} eV")

        for i in range(len(preds)):
            composition = get_composition(infos[i]['atomic_numbers'])
            elements = get_unique_elements(infos[i]['atomic_numbers'])
            all_rows.append({
                'dataset': split_name,
                'sample_id': infos[i]['sample_idx'],
                'composition': composition,
                'elements': ', '.join(elements),
                'num_atoms': infos[i]['natoms'],
                'true_deltaGH_eV': round(float(trues[i]), 4),
                'pred_deltaGH_eV': round(float(preds[i]), 4),
                'abs_error_eV': round(float(err[i]), 4),
            })
        ds.close()

    df_all = pd.DataFrame(all_rows)
    print(f"\nTotal samples scanned: {len(df_all)}")

    # ==================== 筛选商用级 HER 催化材料 ====================
    # 对于HER催化剂，ΔGH (氢吸附自由能) 越接近0 eV 性能越好
    # 商用级别标准: |ΔGH| < 0.1 eV (极优)，拓展到 0.2 eV (优秀)
    # 同时要求模型预测与真实值一致 (误差小)

    # 使用预测值来筛选 (真实场景中没有真实标签，靠模型预测)
    # 但这里我们有标签，采用预测值和真实值都满足条件的交集以确保可靠性

    EXCELLENT_THRESHOLD = 0.10  # |ΔGH| < 0.10 eV — 极优 (Pt级别)
    GOOD_THRESHOLD = 0.20       # |ΔGH| < 0.20 eV — 商用优秀
    MAX_PRED_ERROR = 0.15       # 预测误差需小于此阈值，确保可信度

    df_all['abs_pred_deltaGH'] = df_all['pred_deltaGH_eV'].abs()
    df_all['abs_true_deltaGH'] = df_all['true_deltaGH_eV'].abs()

    # 商用筛选: 预测值和真实值的|ΔGH|都在好的范围内，且预测可靠
    mask_commercial = (
        (df_all['abs_pred_deltaGH'] < GOOD_THRESHOLD) &
        (df_all['abs_true_deltaGH'] < GOOD_THRESHOLD) &
        (df_all['abs_error_eV'] < MAX_PRED_ERROR)
    )
    df_commercial = df_all[mask_commercial].copy()
    df_commercial['grade'] = df_commercial['abs_true_deltaGH'].apply(
        lambda x: '★★★ 极优 (Pt级)' if x < EXCELLENT_THRESHOLD else '★★ 优秀'
    )
    df_commercial = df_commercial.sort_values('abs_true_deltaGH').reset_index(drop=True)
    df_commercial.index = df_commercial.index + 1  # 从1开始编号
    df_commercial.index.name = '排名'

    # 选择输出列
    output_cols = [
        'dataset', 'sample_id', 'composition', 'elements', 'num_atoms',
        'true_deltaGH_eV', 'pred_deltaGH_eV', 'abs_error_eV', 'grade',
    ]
    df_out = df_commercial[output_cols]

    # ==================== 保存结果 ====================
    # 1. 筛选结果表格 (CSV)
    csv_path = output_dir / "HER_commercial_materials_screening.csv"
    df_out.to_csv(csv_path, encoding='utf-8-sig')
    print(f"\n筛选结果已保存: {csv_path}")
    print(f"共筛选出 {len(df_out)} 种商用级 HER 催化材料")
    print(f"  其中 极优(|ΔGH|<{EXCELLENT_THRESHOLD}eV): "
          f"{(df_commercial['abs_true_deltaGH'] < EXCELLENT_THRESHOLD).sum()} 种")
    print(f"  其中 优秀({EXCELLENT_THRESHOLD}≤|ΔGH|<{GOOD_THRESHOLD}eV): "
          f"{((df_commercial['abs_true_deltaGH'] >= EXCELLENT_THRESHOLD) & (df_commercial['abs_true_deltaGH'] < GOOD_THRESHOLD)).sum()} 种")

    # 2. 全部预测结果 (供后续分析)
    full_csv_path = output_dir / "all_predictions.csv"
    df_all.to_csv(full_csv_path, index=False, encoding='utf-8-sig')
    print(f"全部预测结果已保存: {full_csv_path}")

    # 3. 筛选结果 Excel (更美观)
    xlsx_path = output_dir / "HER_commercial_materials_screening.xlsx"
    try:
        with pd.ExcelWriter(str(xlsx_path), engine='openpyxl') as writer:
            df_out.to_excel(writer, sheet_name='商用级HER催化材料')
            # 统计概览
            summary_data = {
                '指标': [
                    '总扫描样本数', '筛选后材料数',
                    f'极优 (|ΔGH|<{EXCELLENT_THRESHOLD}eV)',
                    f'优秀 ({EXCELLENT_THRESHOLD}≤|ΔGH|<{GOOD_THRESHOLD}eV)',
                    '筛选标准: |ΔGH|阈值', '筛选标准: 最大预测误差',
                    '模型', '归一化均值', '归一化标准差',
                ],
                '值': [
                    len(df_all), len(df_out),
                    int((df_commercial['abs_true_deltaGH'] < EXCELLENT_THRESHOLD).sum()),
                    int(((df_commercial['abs_true_deltaGH'] >= EXCELLENT_THRESHOLD) & (df_commercial['abs_true_deltaGH'] < GOOD_THRESHOLD)).sum()),
                    f'< {GOOD_THRESHOLD} eV', f'< {MAX_PRED_ERROR} eV',
                    str(model_path.name),
                    f"{stats['target_mean']:.4f}",
                    f"{stats['target_std']:.4f}",
                ],
            }
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='筛选概览', index=False)
        print(f"Excel报告已保存: {xlsx_path}")
    except ImportError:
        print("[INFO] openpyxl未安装，跳过Excel输出。CSV文件已正常保存。")

    # 4. 打印 Top-20 结果
    print(f"\n{'='*80}")
    print("                 商用级 HER 催化材料 Top-20 排名")
    print(f"{'='*80}")
    print(f"筛选标准: |ΔGH| < {GOOD_THRESHOLD} eV, 预测误差 < {MAX_PRED_ERROR} eV")
    print(f"{'─'*80}")
    top20 = df_out.head(20)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    pd.set_option('display.max_colwidth', 30)
    print(top20.to_string())
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
