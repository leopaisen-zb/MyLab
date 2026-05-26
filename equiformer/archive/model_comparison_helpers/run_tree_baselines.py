#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classic ML baselines (tree and related regressors) for ΔG_H prediction.

Purpose (for the paper):
1) Reproduce strong classical baselines using ONLY the 10 engineered features
   from the previous work (file: data/raw/10features_for_ML.xlsx).
2) Provide a clean comparison point against EquiformerV2 and our new Fusion branch.
3) Save a metrics table and publication-quality bar charts under
   experiments/model_comparison/.

This script intentionally does NOT load deep model results. Those are produced
elsewhere and later stitched together for tables/figures.

Run:
  python experiments/model_comparison/run_tree_baselines.py
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# External gradient boosting libs
# Optional external libs (graceful fallback if missing)
try:
    from xgboost import XGBRegressor  # type: ignore
    _HAS_XGB = True
except Exception:
    XGBRegressor = None  # type: ignore
    _HAS_XGB = False

try:
    from lightgbm import LGBMRegressor  # type: ignore
    _HAS_LGBM = True
except Exception:
    LGBMRegressor = None  # type: ignore
    _HAS_LGBM = False


# ------------------------------- IO paths -----------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_XLSX = PROJECT_ROOT / 'data' / 'raw' / '10features_for_ML.xlsx'
OUT_DIR = PROJECT_ROOT / 'experiments' / 'model_comparison'


# ------------------------------ Data loading --------------------------------
TARGET_CANDIDATES = [
    'delta_g_h', 'delta gh', 'deltagh', 'dg_h', 'dg h', 'dgh',
    'Δg_h', 'Δg h', 'Δgh', 'Δgh',  # Unicode Delta variants
    'target', 'label', 'y'
]


def _norm_key(s: str) -> str:
    # Robust normalization: lower, map Unicode Delta to 'delta', remove spaces/underscores/hyphens
    s = s.replace('Δ', 'delta').replace('δ', 'delta')
    s = s.lower().replace(' ', '').replace('_', '').replace('-', '')
    return s


def detect_target_column(df: pd.DataFrame) -> str:
    # Prepare normalized candidate keys
    cand_norm = {_norm_key(c): c for c in TARGET_CANDIDATES}
    # First pass: normalized direct match
    for c in df.columns:
        if _norm_key(c) in cand_norm:
            return c
    # Second pass: substring heuristics (e.g., contains 'deltagh' or 'dgh')
    for c in df.columns:
        nk = _norm_key(c)
        if ('deltagh' in nk) or (nk == 'dgh'):
            return c
    # Final: explicit case-insensitive search
    lowered = [c.lower() for c in df.columns]
    for cand in TARGET_CANDIDATES:
        if cand.lower() in lowered:
            return df.columns[lowered.index(cand.lower())]
    raise ValueError(
        'Could not detect ΔG_H target column. Please ensure one of the names '
        f'({TARGET_CANDIDATES}) exists in {DATA_XLSX}. Columns: {list(df.columns)}'
    )


def load_data(path: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    if not path.exists():
        raise FileNotFoundError(f'Dataset not found: {path}')
    df = pd.read_excel(path)
    # auto-detect target
    target_col = detect_target_column(df)
    # keep only numeric columns for features
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if target_col not in numeric_cols:
        # target may be numeric but confirm
        if not pd.api.types.is_numeric_dtype(df[target_col]):
            raise ValueError(f'Target column {target_col} must be numeric.')
        numeric_cols.append(target_col)
    # features = numeric except target
    feature_cols = [c for c in numeric_cols if c != target_col]
    # drop rows with NaN in either features or target
    df_clean = df[feature_cols + [target_col]].dropna(axis=0, how='any')
    X = df_clean[feature_cols].to_numpy(dtype=np.float64)
    y = df_clean[target_col].to_numpy(dtype=np.float64)
    return X, y, feature_cols


# ------------------------------- Models -------------------------------------
def build_models() -> Dict[str, Tuple[object, bool]]:
    """
    Returns a dict: name -> (estimator, needs_scaler)
    Only MLP and SVR use a StandardScaler; tree-based models do not.
    """
    models: Dict[str, Tuple[object, bool]] = {
        'DTR': (DecisionTreeRegressor(max_depth=20, random_state=42), False),
        'RFR': (RandomForestRegressor(n_estimators=200, random_state=42), False),
        'ETR': (ExtraTreesRegressor(n_estimators=200, random_state=42), False),
        'GBR': (GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=42), False),
        'MLP': (MLPRegressor(hidden_layer_sizes=(128, 64), activation='relu', max_iter=2000, random_state=42), True),
        'SVR': (SVR(kernel='rbf', C=10.0, gamma='scale'), True),
    }
    # Add optional models if libraries are available
    if _HAS_XGB:
        models['XGBR'] = (XGBRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.9, colsample_bytree=0.9,
            random_state=42, objective='reg:squarederror', n_jobs=0
        ), False)
    if _HAS_LGBM:
        models['LGBMR'] = (LGBMRegressor(
            n_estimators=500, learning_rate=0.05, num_leaves=64, subsample=0.9, colsample_bytree=0.9,
            random_state=42
        ), False)
    return models


def fit_predict(model, X_train, y_train, X_test, needs_scaler: bool) -> np.ndarray:
    if needs_scaler:
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('reg', model),
        ])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        return y_pred
    else:
        model.fit(X_train, y_train)
        return model.predict(X_test)


def train_and_eval(X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    models = build_models()
    rows = []
    for name, (est, needs_scaler) in models.items():
        try:
            y_pred = fit_predict(est, X_train, y_train, X_test, needs_scaler)
            mae = mean_absolute_error(y_test, y_pred)
            # sklearn compatibility: some versions lack 'squared' arg
            try:
                rmse = mean_squared_error(y_test, y_pred, squared=False)  # type: ignore[arg-type]
            except TypeError:
                rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            r2 = r2_score(y_test, y_pred)
            rows.append({'model': name, 'mae': mae, 'rmse': rmse, 'r2': r2})
            print(f"[{name}] MAE={mae:.6f} RMSE={rmse:.6f} R2={r2:.6f}")
        except Exception as e:
            print(f"[{name}] Failed: {e}")
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values('mae', ascending=True).reset_index(drop=True)
    return df


# ------------------------------- Outputs ------------------------------------
def plot_bar(values: List[float], labels: List[str], ylabel: str, title: str, out_path: Path) -> None:
    plt.figure(figsize=(10, 6))
    x = np.arange(len(labels))
    bars = plt.bar(x, values, color='#4C72B0', alpha=0.9)
    for xi, bi, v in zip(x, bars, values):
        plt.text(bi.get_x() + bi.get_width()/2, v, f"{v:.3f}", ha='center', va='bottom', fontsize=10)
    plt.xticks(x, labels, rotation=0)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
    except PermissionError:
        # Fallback with suffix if file is open/locked
        alt = out_path.with_name(out_path.stem + '_new' + out_path.suffix)
        plt.savefig(alt, dpi=300, bbox_inches='tight')
        print(f"Warning: {out_path} locked, saved to {alt}")
    plt.close()


def save_outputs(df: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / 'model_comparison.csv'
    try:
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f'Saved metrics table to: {csv_path}')
    except PermissionError:
        csv_path = OUT_DIR / 'model_comparison_baselines.csv'
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"Warning: default CSV locked. Saved to: {csv_path}")

    # MAE bar plot
    plot_bar(
        values=df['mae'].tolist(),
        labels=df['model'].tolist(),
        ylabel='Test MAE (eV)',
        title='Test MAE on ΔG_H Prediction (10 engineered features)',
        out_path=OUT_DIR / 'model_comparison.png'
    )

    # RMSE bar plot (optional)
    plot_bar(
        values=df['rmse'].tolist(),
        labels=df['model'].tolist(),
        ylabel='Test RMSE (eV)',
        title='Test RMSE on ΔG_H Prediction (10 engineered features)',
        out_path=OUT_DIR / 'model_comparison_rmse.png'
    )


# ------------------------------- Main ---------------------------------------
def main():
    print('=== Classic ML baselines for ΔG_H prediction ===')
    print('Project root:', PROJECT_ROOT)
    print('Data file:', DATA_XLSX)
    if not _HAS_XGB:
        print('Note: xgboost not installed, skipping XGBRegressor. Install with: pip install xgboost')
    if not _HAS_LGBM:
        print('Note: lightgbm not installed, skipping LGBMRegressor. Install with: pip install lightgbm')
    X, y, feat_cols = load_data(DATA_XLSX)
    print(f'Dataset: X={X.shape}, y={y.shape}, features={len(feat_cols)}')

    df = train_and_eval(X, y)
    if df.empty:
        print('No models succeeded. Exiting.')
        return
    print('\nSorted metrics (by MAE):')
    print(df)
    save_outputs(df)

    # Note:
    # Deep-model results (EquiformerV2 baseline and Fusion Gate) are produced elsewhere,
    # e.g., experiments/all/public_cleaned.csv and the fusion run metrics under
    # experiments/20251021_165820_tabfusion_run_gate_fusion_100epochs/metrics.json.
    # We will join those in a higher-level paper table generator, not here.


if __name__ == '__main__':
    main()


