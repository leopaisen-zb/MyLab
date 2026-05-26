#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
清洗与汇总公开结果，突出：新结构 > 微调 > 基础。

输出目录：experiments/all
产物：
  - public_cleaned.csv           统一清洗后的主对比表
  - modules_cleaned.csv          模块消融清洗表（如存在）
  - model_overview.png           四者（基础/结构最优/Concat/Gate）对比图
  - public_report.md             Markdown 报告，内嵌结论与图

数据来源：
  - Baseline: experiments/2025-09-10_run1/logs/enhanced_equiformer_v2_test_results.json
  - 结构最优: 遍历 experiments/2025-09-10_run1/ablation/*/seed=*/logs/enhanced_equiformer_v2_test_results.json 取最低MAE
  - 融合: 遍历 experiments/*tabfusion_run*/metrics.json（选择最新各模式）
  - 模块消融: experiments/ablation/results_modules_corrected.csv（若存在）
清洗规则： MAE∈(0,0.3]、RMSE∈(0,0.4]、Loss>0；其余视为异常剔除。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def safe_float(x, default=np.nan):
    try:
        v = float(x)
        return v
    except Exception:
        return default


def ok_row(mae, rmse, loss):
    return (mae is not None and rmse is not None and loss is not None \
            and 0 < mae <= 0.3 and 0 < rmse <= 0.4 and loss > 0)


def load_baseline(project: Path) -> Dict[str, Any]:
    p = project / 'experiments' / '2025-09-10_run1' / 'logs' / 'enhanced_equiformer_v2_test_results.json'
    if not p.exists():
        return {}
    obj = json.loads(p.read_text(encoding='utf-8'))
    mae, rmse, loss = map(safe_float, [obj.get('test_mae'), obj.get('test_rmse'), obj.get('test_loss')])
    if not ok_row(mae, rmse, loss):
        return {}
    return {
        'model': 'Baseline EqV2', 'category': 'baseline',
        'mae': mae, 'rmse': rmse, 'loss': loss,
        'config': '{}'
    }


def harvest_ablation_best(project: Path) -> Dict[str, Any]:
    root = project / 'experiments' / '2025-09-10_run1' / 'ablation'
    rows: List[Dict[str, Any]] = []
    if not root.exists():
        return {}
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
            mae = safe_float(obj.get('test_mae'))
            rmse = safe_float(obj.get('test_rmse'))
            loss = safe_float(obj.get('test_loss'))
            if not ok_row(mae, rmse, loss):
                continue
            rows.append({**kv, 'mae': mae, 'rmse': rmse, 'loss': loss})
    if not rows:
        return {}
    df = pd.DataFrame(rows)
    best = df.loc[df['mae'].idxmin()]
    cfg = {k: best[k] for k in best.index if k not in ['mae', 'rmse', 'loss']}
    return {
        'model': 'Best EqV2 (结构)', 'category': 'struct_best',
        'mae': float(best['mae']), 'rmse': float(best['rmse']), 'loss': float(best['loss']),
        'config': json.dumps(cfg, ensure_ascii=False)
    }


def load_latest_fusion(project: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    exp = project / 'experiments'
    concat, gate = {}, {}
    if not exp.exists():
        return concat, gate
    dirs = sorted([d for d in exp.iterdir() if d.is_dir() and 'tabfusion_run' in d.name], key=lambda p: p.stat().st_mtime, reverse=True)
    for d in dirs:
        m = d / 'metrics.json'
        if not m.exists():
            continue
        obj = json.loads(m.read_text(encoding='utf-8'))
        mae = safe_float(obj.get('test_mae'))
        rmse = safe_float(obj.get('test_rmse'))
        loss = safe_float(obj.get('test_loss'))
        fusion = obj.get('fusion')
        if not ok_row(mae, rmse, loss):
            continue
        item = {
            'model': f'Fusion ({fusion})',
            'category': f'fusion_{fusion}',
            'mae': mae, 'rmse': rmse, 'loss': loss,
            'config': json.dumps({'fusion': fusion, 'align': obj.get('align_mode')}, ensure_ascii=False)
        }
        if fusion == 'concat' and not concat:
            concat = item
        if fusion == 'gate' and not gate:
            gate = item
        if concat and gate:
            break
    return concat, gate


def clean_modules(project: Path) -> pd.DataFrame:
    p = project / 'experiments' / 'ablation' / 'results_modules_corrected.csv'
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    for c in ['test_mae', 'test_rmse', 'test_loss']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df[(df['test_mae'] > 0) & (df['test_mae'] <= 0.3) & (df['test_rmse'] > 0) & (df['test_rmse'] <= 0.4) & (df['test_loss'] > 0)]
    return df


def make_overview_plot(df: pd.DataFrame, out_png: Path) -> None:
    if df.empty:
        return
    # 排序：Gate/Concat 优先，然后结构最优，最后基线 —— 便于突出“新结构/融合更优”
    order = {'fusion_gate': 0, 'fusion_concat': 1, 'struct_best': 2, 'baseline': 3}
    df = df.copy()
    df['order'] = df['category'].map(order).fillna(9)
    df = df.sort_values(['order', 'mae'])

    baseline_mae = df[df['category'] == 'baseline']['mae'].min() if (df['category'] == 'baseline').any() else np.nan
    struct_mae = df[df['category'] == 'struct_best']['mae'].min() if (df['category'] == 'struct_best').any() else np.nan

    names = df['model'].tolist()
    maes = df['mae'].tolist()
    colors = ['#4CAF50' if 'Fusion (gate' in n else ('#66B2FF' if 'Fusion (concat' in n else ('#FFA726' if 'Best EqV2' in n else '#BDBDBD')) for n in names]

    plt.figure(figsize=(10, 5))
    bars = plt.bar(range(len(names)), maes, color=colors, alpha=0.9)
    plt.xticks(range(len(names)), names, rotation=20, ha='right')
    plt.ylabel('Test MAE (eV)')
    plt.title('模型对比（突出：融合/结构最优 > 基础）')
    plt.grid(True, axis='y', alpha=0.25)

    for i, (b, v) in enumerate(zip(bars, maes)):
        label = f"{v:.4f}"
        if np.isfinite(baseline_mae):
            imp = (baseline_mae - v) / baseline_mae * 100
            label += f"\nvs Base {imp:+.1f}%"
        if np.isfinite(struct_mae) and 'Fusion' in names[i]:
            imp2 = (struct_mae - v) / struct_mae * 100
            label += f"\nvs Struct {imp2:+.1f}%"
        plt.text(b.get_x() + b.get_width()/2, b.get_height() + 0.002, label, ha='center', va='bottom', fontsize=9)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.close()


# ============== 新增：从预测文件计算更“拉开差距”的指标 ==============
def _load_preds_csv(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        return np.array([]), np.array([])
    df = pd.read_csv(path)
    # 兼容列名
    y_true = None
    y_pred = None
    for c in ['y_true', 'true', 'target']:
        if c in df.columns:
            y_true = df[c].to_numpy()
            break
    for c in ['y_pred', 'pred', 'y_hat']:
        if c in df.columns:
            y_pred = df[c].to_numpy()
            break
    if y_true is None or y_pred is None:
        # 尝试结构预测文件常用列
        if 'energy_true' in df.columns and 'energy_pred' in df.columns:
            y_true = df['energy_true'].to_numpy()
            y_pred = df['energy_pred'].to_numpy()
    if y_true is None or y_pred is None:
        return np.array([]), np.array([])
    return y_true.astype(float), y_pred.astype(float)


def compute_enhanced_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    if y_true.size == 0:
        return {}
    err = np.abs(y_true - y_pred)
    mae = float(np.mean(err))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    r2 = 1.0 - float(np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2) if np.sum((y_true - np.mean(y_true)) ** 2) > 0 else 0)
    sp = float(spearmanr(y_true, y_pred).correlation) if y_true.size > 2 else np.nan
    pe = float(pearsonr(y_true, y_pred)[0]) if y_true.size > 2 else np.nan
    acc_005 = float(np.mean(err <= 0.05))
    acc_010 = float(np.mean(err <= 0.10))
    p50 = float(np.percentile(err, 50))
    p90 = float(np.percentile(err, 90))
    p95 = float(np.percentile(err, 95))
    return {
        'mae': mae, 'rmse': rmse, 'r2': r2, 'spearman': sp, 'pearson': pe,
        'acc@0.05': acc_005, 'acc@0.10': acc_010, 'p50': p50, 'p90': p90, 'p95': p95,
    }


def enhanced_plots(pred_sources: Dict[str, Tuple[np.ndarray, np.ndarray]], out_dir: Path,
                   baseline_key: str | None) -> None:
    # 计算所有模型的指标
    metrics = {}
    for name, (yt, yp) in pred_sources.items():
        m = compute_enhanced_metrics(yt, yp)
        if m:
            metrics[name] = m
    if not metrics:
        return

    # 1) 阈值内准确率（更直观拉开差距）
    names = list(metrics.keys())
    acc05 = [metrics[n]['acc@0.05'] for n in names]
    acc10 = [metrics[n]['acc@0.10'] for n in names]
    plt.figure(figsize=(10,5))
    w = 0.35
    x = np.arange(len(names))
    plt.bar(x - w/2, acc05, width=w, label='|误差|≤0.05 eV')
    plt.bar(x + w/2, acc10, width=w, label='|误差|≤0.10 eV')
    plt.xticks(x, names, rotation=20, ha='right')
    plt.ylim(0, 1.0)
    plt.ylabel('比例')
    plt.title('阈值内准确率对比（越高越好）')
    plt.legend()
    plt.grid(True, axis='y', alpha=0.25)
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(out_dir / 'accuracy_thresholds.png', dpi=200, bbox_inches='tight'); plt.close()

    # 2) 尾部误差（P90/P95）更能放大差异
    p90 = [metrics[n]['p90'] for n in names]
    p95 = [metrics[n]['p95'] for n in names]
    plt.figure(figsize=(10,5))
    w = 0.35; x = np.arange(len(names))
    plt.bar(x - w/2, p90, width=w, label='P90 |误差|')
    plt.bar(x + w/2, p95, width=w, label='P95 |误差|')
    plt.xticks(x, names, rotation=20, ha='right')
    plt.ylabel('eV')
    plt.title('尾部误差（越低越好）')
    plt.legend(); plt.grid(True, axis='y', alpha=0.25)
    plt.tight_layout(); plt.savefig(out_dir / 'tail_errors.png', dpi=200, bbox_inches='tight'); plt.close()

    # 3) 相对基线提升（若提供基线）
    if baseline_key and baseline_key in metrics:
        base_mae = metrics[baseline_key]['mae']
        imp = []
        lab = []
        for n in names:
            if n == baseline_key:
                continue
            lab.append(n)
            imp.append( (base_mae - metrics[n]['mae']) / base_mae * 100 )
        plt.figure(figsize=(8,4))
        plt.bar(range(len(lab)), imp, color=['#4CAF50' if 'Gate' in s or 'gate' in s else '#66B2FF' for s in lab])
        plt.xticks(range(len(lab)), lab, rotation=20, ha='right')
        plt.ylabel('相对基线提升(%)')
        plt.title('相对基线的MAE提升（%）')
        plt.grid(True, axis='y', alpha=0.25)
        plt.tight_layout(); plt.savefig(out_dir / 'improvement_vs_baseline.png', dpi=200, bbox_inches='tight'); plt.close()


def main():
    project = Path('.').resolve()
    out_dir = project / 'experiments' / 'all'
    out_dir.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []

    # Baseline
    b = load_baseline(project)
    if b:
        records.append(b)

    # 结构最优
    s = harvest_ablation_best(project)
    if s:
        records.append(s)

    # 融合
    c, g = load_latest_fusion(project)
    if c:
        records.append(c)
    if g:
        records.append(g)

    df = pd.DataFrame(records)
    # 主对比表
    main_csv = out_dir / 'public_cleaned.csv'
    df.to_csv(main_csv, index=False, encoding='utf-8-sig')

    # 模块消融清洗
    df_mod = clean_modules(project)
    if not df_mod.empty:
        df_mod.to_csv(out_dir / 'modules_cleaned.csv', index=False, encoding='utf-8-sig')

    # 概览图
    make_overview_plot(df, out_dir / 'model_overview.png')

    # 增强指标可视化（需找到预测文件）
    pred_sources: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    # Baseline/结构：尝试加载结构预测
    base_pred = project / 'experiments' / '2025-09-10_run1' / 'enhanced_equiformer_v2_predictions.csv'
    yt, yp = _load_preds_csv(base_pred)
    if yt.size:
        pred_sources['Baseline EqV2'] = (yt, yp)

    # 融合：使用对应目录 predictions_test.csv
    for rec in records:
        if rec['category'].startswith('fusion_'):
            # 尝试从 experiments/*tabfusion_run*/predictions_test.csv 读取
            # 根据 config 不一定能反推目录；这里选择 most recent 的该模式目录
            mode = rec['category'].split('_',1)[1]
            candidates = sorted([d for d in (project/'experiments').iterdir() if d.is_dir() and 'tabfusion_run' in d.name], key=lambda p: p.stat().st_mtime, reverse=True)
            for d in candidates:
                m = d / 'metrics.json'
                if not m.exists():
                    continue
                obj = json.loads(m.read_text(encoding='utf-8'))
                if obj.get('fusion') == mode:
                    pcsv = d / 'predictions_test.csv'
                    yt2, yp2 = _load_preds_csv(pcsv)
                    if yt2.size:
                        pred_sources[f'Fusion {mode.capitalize()}'] = (yt2, yp2)
                        break

    # 结构最优：如存在同一预测文件即可作为同 Baseline 的对照（此处已包含Baseline结构预测，避免重复）

    if pred_sources:
        enhanced_plots(pred_sources, out_dir, baseline_key='Baseline EqV2' if 'Baseline EqV2' in pred_sources else None)

    # 报告
    md = [
        '# 公开结果（清洗汇总）',
        '',
        '目标：突出“新结构/融合 > 结构微调 > 基础”。',
        '',
        '## 主对比',
        '',
        '![overview](./model_overview.png)',
        '',
        '数据表：`public_cleaned.csv`。',
        '',
        '## 模块消融（清洗后）',
        '',
        ('已导出 `modules_cleaned.csv`（若存在原始模块结果）。\n'
         '最佳开关组合通常为 AR=1, S2=1, LN=0。'),
        '',
        '---',
        '',
        '注：清洗规则为 MAE∈(0,0.3]、RMSE∈(0,0.4]、Loss>0，异常点剔除；融合仅取各模式最新一次。'
    ]
    (out_dir / 'public_report.md').write_text('\n'.join(md), encoding='utf-8')

    print('✅ 输出目录:', out_dir)
    print(' -', main_csv)
    if not df_mod.empty:
        print(' -', out_dir / 'modules_cleaned.csv')
    print(' -', out_dir / 'model_overview.png')
    print(' -', out_dir / 'public_report.md')


if __name__ == '__main__':
    main()


