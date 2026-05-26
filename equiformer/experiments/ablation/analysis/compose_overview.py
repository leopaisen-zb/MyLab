#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
综合消融概览可视化（稳健版）

- 自动遍历 experiments/2025-09-10_run1/ablation/*/seed=*/logs/enhanced_equiformer_v2_test_results.json
  聚合 grid_resolution（Lmax）、num_layers 与模块(AR、S2、LN)指标
- 模块结果优先读取 experiments/ablation/results_modules_corrected.csv（若存在）
- 缺失值使用 NaN 并在绘图前过滤，避免 -1 被当作真实值
- 输出一张 2x2 综合图到 experiments/ablation/plots/ablation_overview.png
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def safe_float(x) -> float:
    try:
        v = float(x)
        if np.isfinite(v):
            return v
    except Exception:
        pass
    return np.nan


def harvest_ablation_rows(ablation_root: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if not ablation_root.exists():
        return pd.DataFrame()

    for cfg_dir in ablation_root.iterdir():
        if not cfg_dir.is_dir():
            continue
        for seed_dir in cfg_dir.glob('seed=*'):
            logs = seed_dir / 'logs' / 'enhanced_equiformer_v2_test_results.json'
            if not logs.exists():
                continue
            # 从目录名解析参数键值
            parts = cfg_dir.name.split('__')
            kv = {}
            for p in parts:
                if '=' in p:
                    k, v = p.split('=', 1)
                    kv[k] = v
            try:
                obj = json.loads(logs.read_text(encoding='utf-8'))
            except Exception:
                continue

            row = {
                'grid_resolution': safe_float(kv.get('grid_resolution')),
                'num_layers': safe_float(kv.get('num_layers')),
                'attn_renorm': safe_float(kv.get('attn_renorm')),
                'sep_s2': safe_float(kv.get('sep_s2')),
                'sep_ln': safe_float(kv.get('sep_ln')),
                'test_mae': safe_float(obj.get('test_mae')),
                'test_rmse': safe_float(obj.get('test_rmse')),
                'test_loss': safe_float(obj.get('test_loss')),
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    return df


def load_module_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    # 规范列名为数值
    for c in ['attn_renorm', 'sep_s2', 'sep_ln', 'test_mae', 'test_rmse', 'test_loss']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def main():
    project = Path('.').resolve()
    ablation_root = project / 'experiments' / '2025-09-10_run1' / 'ablation'
    out_dir = project / 'experiments' / 'ablation' / 'plots'
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 汇总ablation目录
    df_all = harvest_ablation_rows(ablation_root)

    # 2) 尝试加载修正的模块CSV（优先生效）
    df_mod_csv = load_module_csv(project / 'experiments' / 'ablation' / 'results_modules_corrected.csv')

    # 3) 创建画布
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # 顶左：Lmax（grid_resolution）
    ax = axes[0, 0]
    if not df_all.empty and 'grid_resolution' in df_all.columns:
        dfg = df_all.dropna(subset=['grid_resolution', 'test_mae']).groupby('grid_resolution')['test_mae']
        if len(dfg) > 0:
            xs = sorted(dfg.mean().index.tolist())
            means = [dfg.mean()[x] for x in xs]
            stds = [dfg.std()[x] for x in xs]
            ax.errorbar(xs, means, yerr=stds, marker='o', capsize=4)
            ax.set_xlabel('Grid Resolution')
            ax.set_ylabel('MAE (eV)')
            ax.set_title('Lmax消融实验结果')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, '无有效Lmax数据', ha='center', va='center')
    else:
        ax.text(0.5, 0.5, '无有效Lmax数据', ha='center', va='center')

    # 顶右：层数（num_layers）
    ax = axes[0, 1]
    if not df_all.empty and 'num_layers' in df_all.columns:
        dfg = df_all.dropna(subset=['num_layers', 'test_mae']).groupby('num_layers')['test_mae']
        if len(dfg) > 0:
            xs = sorted(dfg.mean().index.tolist())
            means = [dfg.mean()[x] for x in xs]
            stds = [dfg.std()[x] for x in xs]
            ax.errorbar(xs, means, yerr=stds, marker='o', color='r', capsize=4)
            ax.set_xlabel('Number of Layers')
            ax.set_ylabel('MAE (eV)')
            ax.set_title('层数消融实验结果')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, '无有效层数数据', ha='center', va='center')
    else:
        ax.text(0.5, 0.5, '无有效层数数据', ha='center', va='center')

    # 底左：模块消融（AR、S2、LN）
    ax = axes[1, 0]
    # 首选CSV，其次从df_all聚合
    if not df_mod_csv.empty:
        dm = df_mod_csv.copy()
    else:
        dm = df_all.copy()
    if not dm.empty and set(['attn_renorm', 'sep_s2', 'sep_ln']).issubset(dm.columns):
        dm = dm.dropna(subset=['attn_renorm', 'sep_s2', 'sep_ln', 'test_mae'])
        grouped = dm.groupby(['attn_renorm', 'sep_s2', 'sep_ln'])['test_mae'].agg(['mean', 'std']).reset_index()
        if not grouped.empty:
            labels = [f"AR={int(a)}, S2={int(s)}, LN={int(l)}" for a, s, l in zip(grouped['attn_renorm'], grouped['sep_s2'], grouped['sep_ln'])]
            ax.bar(range(len(grouped)), grouped['mean'], yerr=grouped['std'], capsize=4, color='#2E86AB', alpha=0.8)
            ax.set_xticks(range(len(grouped)))
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_ylabel('MAE (eV)')
            ax.set_title('模块消融实验结果')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, '无有效模块数据', ha='center', va='center')
    else:
        ax.text(0.5, 0.5, '无有效模块数据', ha='center', va='center')

    # 底右：最佳配置对比（基模、最佳Equiformer、最佳层数、最佳模块）
    ax = axes[1, 1]
    items: List[tuple[str, float]] = []

    # 基础EquiformerV2（若存在）
    base_json = project / 'experiments' / '2025-09-10_run1' / 'logs' / 'enhanced_equiformer_v2_test_results.json'
    if base_json.exists():
        try:
            obj = json.loads(base_json.read_text(encoding='utf-8'))
            mae = safe_float(obj.get('test_mae'))
            if np.isfinite(mae):
                items.append(('EquiformerV2 (基础)', mae))
        except Exception:
            pass

    # 最佳Equiformer（从df_all最小MAE）
    if not df_all.empty:
        m = df_all['test_mae'].min()
        if np.isfinite(m):
            items.append(('EquiformerV2 (最佳)', m))

    # 层数=4（若存在）
    if not df_all.empty and (df_all['num_layers'] == 4).any():
        m = df_all.loc[df_all['num_layers'] == 4, 'test_mae'].min()
        if np.isfinite(m):
            items.append(('Layers (4)', m))

    # 最佳模块（来自CSV或聚合）
    if not dm.empty and {'attn_renorm', 'sep_s2', 'sep_ln'}.issubset(dm.columns):
        grp = dm.groupby(['attn_renorm', 'sep_s2', 'sep_ln'])['test_mae'].mean().reset_index()
        if not grp.empty:
            m = grp['test_mae'].min()
            if np.isfinite(m):
                items.append(('Best Modules', m))

    if items:
        names = [k for k, _ in items]
        values = [v for _, v in items]
        bars = ax.bar(range(len(values)), values, color=['#f2c14e', '#5dd39e', '#4d7ea8', '#e94f37'][:len(values)], alpha=0.85)
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(names, rotation=20, ha='right')
        ax.set_ylabel('MAE (eV)')
        ax.set_title('最佳配置对比')
        ax.grid(True, alpha=0.3)
        for i, (b, v) in enumerate(zip(bars, values)):
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.002, f"{v:.6f}", ha='center', va='bottom', fontsize=9)
    else:
        ax.text(0.5, 0.5, '无可对比数据', ha='center', va='center')

    plt.tight_layout()
    out_file = out_dir / 'ablation_overview.png'
    plt.savefig(out_file, dpi=200, bbox_inches='tight')
    print(f'✅ 综合消融概览图已保存: {out_file}')


if __name__ == '__main__':
    main()


