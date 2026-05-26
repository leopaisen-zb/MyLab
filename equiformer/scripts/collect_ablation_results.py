#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
遍历 ablation 实验目录，汇总各配置与种子的指标，输出 CSV 与图表。
使用：
  python scripts/collect_ablation_results.py --root D:\\myproject\\equiformer_v2\\experiments\\2025-09-10_run1\\ablation
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def read_metrics_json(metrics_path: Path) -> Dict[str, Any]:
    try:
        with open(metrics_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def walk_experiments(root: Path) -> List[Tuple[Path, Dict[str, Any]]]:
    rows: List[Tuple[Path, Dict[str, Any]]] = []
    for metrics_file in root.rglob('logs/enhanced_equiformer_v2_test_results.json'):
        m = read_metrics_json(metrics_file)
        if not m:
            continue
        rows.append((metrics_file, m))
    return rows


def parse_config_from_dir(dir_path: Path) -> Dict[str, Any]:
    # 从路径中解析简单的 k=v 片段，形如 ablation/num_layers=3__sphere_channels=64/seed=0/
    cfg: Dict[str, Any] = {}
    parts = list(dir_path.parts)
    try:
        if 'ablation' in parts:
            idx = parts.index('ablation')
            # 下一级通常为 "k=v__k=v" 形式
            if idx + 1 < len(parts):
                kvs = parts[idx + 1]
                for seg in kvs.split('__'):
                    if '=' in seg:
                        k, v = seg.split('=', 1)
                        cfg[k] = v
            # seed 目录
            for p in parts:
                if p.startswith('seed='):
                    cfg['seed'] = p.split('=', 1)[-1]
    except Exception:
        pass
    return cfg


def aggregate_by_config(rows: List[Tuple[Path, Dict[str, Any]]]) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    for metrics_file, m in rows:
        exp_dir = metrics_file.parent.parent  # logs -> experiment dir
        cfg = parse_config_from_dir(exp_dir)
        rec = {**cfg, **m}
        rec['experiment_dir'] = str(exp_dir)
        records.append(rec)
    df = pd.DataFrame(records)
    # 将数字列转换为 float
    for c in df.columns:
        if c not in ['experiment_dir'] and c not in ['seed']:
            try:
                df[c] = pd.to_numeric(df[c])
            except Exception:
                pass
    return df


def group_and_summarize(df: pd.DataFrame, key_cols: List[str]) -> pd.DataFrame:
    value_cols = [c for c in df.columns if c not in key_cols + ['seed', 'experiment_dir']]
    aggs: Dict[str, List[str]] = {c: ['mean', 'std'] for c in value_cols}
    g = df.groupby(key_cols, dropna=False).agg(aggs)
    # 展平多层列
    g.columns = [f"{v}_{stat}" for v, stat in g.columns]
    g = g.reset_index()
    return g


def plot_mae_rmse(summary: pd.DataFrame, key_cols: List[str], out_png: Path):
    # 简单对比，将不同配置（拼接key）在 X 轴显示 MAE/RMSE
    plot_df = summary.copy()
    plot_df['config'] = plot_df[key_cols].astype(str).agg('|'.join, axis=1)
    mae_col, rmse_col = 'test_mae_mean', 'test_rmse_mean'
    mae_std, rmse_std = 'test_mae_std', 'test_rmse_std'
    plt.figure(figsize=(max(8, len(plot_df) * 0.8), 6))
    x = np.arange(len(plot_df))
    w = 0.35
    plt.bar(x - w/2, plot_df[mae_col], yerr=plot_df[mae_std], width=w, label='MAE', alpha=0.8)
    plt.bar(x + w/2, plot_df[rmse_col], yerr=plot_df[rmse_std], width=w, label='RMSE', alpha=0.8)
    plt.xticks(x, plot_df['config'], rotation=45, ha='right')
    plt.ylabel('eV')
    plt.title('Ablation: MAE/RMSE')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description='Collect ablation results to CSV and plots')
    ap.add_argument('--root', type=str, required=True)
    ap.add_argument('--out', type=str, default=None, help='Output CSV path (default: <root>/ablation_summary.csv)')
    args = ap.parse_args()

    root = Path(args.root)
    rows = walk_experiments(root)
    if not rows:
        print('No metrics found under', root)
        return
    df = aggregate_by_config(rows)

    # 选取关键列作为配置键（若不存在则自动回退为空）
    key_cols = [c for c in df.columns if c in (
        'num_layers','sphere_channels','attn_hidden_channels','num_heads','ffn_hidden_channels',
        'grid_resolution','edge_channels','lmax_list','mmax_list','use_atom_edge_embedding','share_atom_edge_embedding'
    )]
    if not key_cols:
        # 回退：仅按 experiment_dir 聚合（基本不会跨 seed）
        key_cols = ['experiment_dir']

    summary = group_and_summarize(df, key_cols)
    out_csv = Path(args.out) if args.out else (root / 'ablation_summary.csv')
    summary.to_csv(out_csv, index=False)
    print('Saved:', out_csv)

    # 绘图
    out_png = out_csv.with_name('ablation_mae_rmse.png')
    try:
        plot_mae_rmse(summary, key_cols, out_png)
        print('Saved:', out_png)
    except Exception as e:
        print('Plot failed:', e)


if __name__ == '__main__':
    main()


