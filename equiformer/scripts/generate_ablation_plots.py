#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate three publication-quality figures (English-only):
1) Module Ablation Plot
2) Ablation Summary Plot (all experiment variants)
3) New Network vs Baseline Comparison Plot

Inputs (auto-detected if available):
- experiments/all/public_cleaned.csv
- experiments/all/modules_cleaned.csv
- experiments/2025-09-10_run1/ablation/*/seed=*/logs/enhanced_equiformer_v2_test_results.json
- experiments/*tabfusion_run*/metrics.json

Outputs:
- experiments/now_result/module_ablation.png
- experiments/now_result/ablation_summary.png
- experiments/now_result/new_vs_baseline.png

Design goals:
- English-only labels, value annotations (3 decimals), clean seaborn/matplotlib style
- Choose metric with clearer separation (by variance among Loss, RMSE; R^2 optional if predictions available)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
sns.set_context("talk", font_scale=0.9)
sns.set_style("whitegrid")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_modules_df(project: Path) -> pd.DataFrame:
    # Prefer cleaned module CSV if exists
    p1 = project / 'experiments' / 'all' / 'modules_cleaned.csv'
    if p1.exists():
        df = pd.read_csv(p1)
        return df
    # fallback to corrected source
    p2 = project / 'experiments' / 'ablation' / 'results_modules_corrected.csv'
    if p2.exists():
        df = pd.read_csv(p2)
        for c in ['test_mae', 'test_rmse', 'test_loss']:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        df = df[(df['test_mae'] > 0) & (df['test_mae'] <= 0.3) & (df['test_rmse'] > 0) & (df['test_rmse'] <= 0.4) & (df['test_loss'] > 0)]
        return df
    return pd.DataFrame()


def figure_module_ablation(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        print('[Module] No data, skip.')
        return
    df = df.copy()
    # Build label
    df['label'] = df.apply(lambda r: f"AR={int(r['attn_renorm'])}, S2={int(r['sep_s2'])}, LN={int(r['sep_ln'])}", axis=1)
    # Choose metric with larger variance (Loss vs RMSE)
    metrics = [('test_loss', 'Loss'), ('test_rmse', 'RMSE')]
    variances = [(df[m].var(), m, n) for m, n in metrics if m in df.columns]
    metric_col, metric_name = ('test_mae', 'MAE')
    if variances:
        metric_col, metric_name = sorted(variances, key=lambda x: x[0], reverse=True)[0][1:]
    # Sort by performance (ascending if lower is better)
    df = df.sort_values(metric_col, ascending=True)

    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x='label', y=metric_col, data=df, palette='viridis')
    ax.set_xlabel('Module Configuration')
    ax.set_ylabel(metric_name)
    ax.set_title('Module Ablation')
    # Annotate values
    for p in ax.patches:
        v = p.get_height()
        ax.annotate(f"{v:.3f}", (p.get_x() + p.get_width()/2, v), ha='center', va='bottom', fontsize=9)
    plt.xticks(rotation=25, ha='right')
    plt.tight_layout()
    ensure_dir(out_path.parent)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print('[Module] Saved:', out_path)


def harvest_ablation_variants(project: Path) -> pd.DataFrame:
    root = project / 'experiments' / '2025-09-10_run1' / 'ablation'
    rows: List[Dict[str, Any]] = []
    if not root.exists():
        return pd.DataFrame()
    for cfg in root.iterdir():
        if not cfg.is_dir():
            continue
        kv = {}
        for part in cfg.name.split('__'):
            if '=' in part:
                k, v = part.split('=', 1)
                kv[k] = v
        for j in cfg.glob('seed=*/logs/enhanced_equiformer_v2_test_results.json'):
            try:
                obj = json.loads(j.read_text(encoding='utf-8'))
            except Exception:
                continue
            row = {**kv,
                   'test_mae': obj.get('test_mae'),
                   'test_rmse': obj.get('test_rmse'),
                   'test_loss': obj.get('test_loss')}
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # Numeric casting & cleaning
    for c in ['num_layers', 'sphere_channels', 'num_heads', 'grid_resolution', 'edge_channels', 'eSCN', 'attn_renorm', 'sep_s2', 'sep_ln', 'test_mae', 'test_rmse', 'test_loss']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df[(df['test_mae'] > 0) & (df['test_mae'] <= 0.3) & (df['test_rmse'] > 0) & (df['test_rmse'] <= 0.4) & (df['test_loss'] > 0)]
    # Build label
    def cfg_label(r):
        parts = []
        for k in ['num_layers', 'sphere_channels', 'num_heads', 'grid_resolution', 'edge_channels']:
            if k in df.columns and not np.isnan(r.get(k, np.nan)):
                parts.append(f"{k}={int(r[k])}")
        for k in ['attn_renorm', 'sep_s2', 'sep_ln']:
            if k in df.columns and not np.isnan(r.get(k, np.nan)):
                parts.append(f"{k}={int(r[k])}")
        return ', '.join(parts)
    df['label'] = df.apply(cfg_label, axis=1)
    return df


def figure_ablation_summary(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        print('[Summary] No data, skip.')
        return
    # Choose metric with larger separation among Loss/RMSE
    metrics = [('test_loss', 'Loss'), ('test_rmse', 'RMSE')]
    variances = [(df[m].var(), m, n) for m, n in metrics if m in df.columns]
    metric_col, metric_name = ('test_mae', 'MAE')
    if variances:
        metric_col, metric_name = sorted(variances, key=lambda x: x[0], reverse=True)[0][1:]
    df = df.sort_values(metric_col, ascending=True)
    # Limit number of configs to keep the plot readable (top 20)
    df_plot = df.head(20)

    plt.figure(figsize=(14, 7))
    ax = sns.barplot(x='label', y=metric_col, data=df_plot, palette='crest')
    ax.set_xlabel('Experiment Variant (Top 20 by performance)')
    ax.set_ylabel(metric_name)
    ax.set_title('Ablation Summary (Top performers)')
    for p in ax.patches:
        v = p.get_height()
        ax.annotate(f"{v:.3f}", (p.get_x()+p.get_width()/2, v), ha='center', va='bottom', fontsize=8)
    plt.xticks(rotation=25, ha='right')
    plt.tight_layout()
    ensure_dir(out_path.parent)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print('[Summary] Saved:', out_path)


def load_public_cleaned(project: Path) -> pd.DataFrame:
    p = project / 'experiments' / 'all' / 'public_cleaned.csv'
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()


def figure_new_vs_baseline(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        print('[Compare] No data, skip.')
        return
    # Select baseline and best fusion (prefer gate)
    baseline = df[df['category'] == 'baseline'].sort_values('loss').head(1)
    fusion = df[df['category'].isin(['fusion_gate', 'fusion_concat'])].sort_values('loss').head(1)
    if baseline.empty or fusion.empty:
        print('[Compare] Missing baseline or fusion rows.')
        return
    labels = ['Baseline', 'New Fusion']
    losses = [float(baseline['loss'].values[0]), float(fusion['loss'].values[0])]
    colors = ['#1f77b4', '#ff7f0e']  # blue, orange

    plt.figure(figsize=(6, 5))
    bars = plt.bar(labels, losses, color=colors, alpha=0.9)
    for b, v in zip(bars, losses):
        plt.text(b.get_x()+b.get_width()/2, v, f"{v:.3f}", ha='center', va='bottom', fontsize=10)
    plt.ylabel('Loss (lower is better)')
    plt.title('New Fusion Network vs Baseline')
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    ensure_dir(out_path.parent)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print('[Compare] Saved:', out_path)


def main():
    project = Path('.').resolve()
    out_dir = project / 'experiments' / 'now_result'
    ensure_dir(out_dir)

    # 1) Module Ablation
    mod_df = load_modules_df(project)
    figure_module_ablation(mod_df, out_dir / 'module_ablation.png')

    # 2) Ablation Summary (harvest ablation variants)
    ab_df = harvest_ablation_variants(project)
    figure_ablation_summary(ab_df, out_dir / 'ablation_summary.png')

    # 3) New vs Baseline (public_cleaned)
    public_df = load_public_cleaned(project)
    # harmonize columns if needed
    if not public_df.empty:
        # ensure lowercase keys
        cols = {c: c.lower() for c in public_df.columns}
        public_df.rename(columns=cols, inplace=True)
    figure_new_vs_baseline(public_df, out_dir / 'new_vs_baseline.png')

    print('\nAll figures have been saved under:', out_dir)


if __name__ == '__main__':
    main()


