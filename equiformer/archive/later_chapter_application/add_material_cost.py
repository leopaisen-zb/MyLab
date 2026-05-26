#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为 HER 商用材料筛选表添加材料成本估算。
价格基于 2025 年全球市场行情（LME / Kitco / USGS MCS 2025 / 供应商报价），
统一单位: USD/kg。
"""

import re
import pandas as pd
import numpy as np
from collections import Counter
from pathlib import Path

# ====================================================================
# 元素价格字典  (USD / kg，2025 年参考价)
# 来源标注:
#   LME   = 伦敦金属交易所 3 月期货均价
#   KITCO = Kitco 贵金属现货年均价
#   USGS  = USGS Mineral Commodity Summaries 2025
#   IND   = 工业气体/化学品供应商大宗报价
#   AMES  = Ames Laboratory / 小批量稀有金属报价
# ====================================================================
ELEMENT_PRICE_USD_KG = {
    # --- 非金属 / 气体 ---
    "H":  1.5,        # IND: 工业高纯氢 ~$1-3/kg
    "N":  0.50,       # IND: 液氮大宗 ~$0.3-0.8/kg
    "O":  0.30,       # IND: 工业氧 ~$0.2-0.5/kg
    "C":  0.50,       # IND: 工业碳黑/石墨 ~$0.5/kg
    "Si": 2.50,       # USGS: 冶金级硅 ~$2.5/kg

    # --- 碱金属 / 碱土金属 ---
    "Ca": 3.00,       # USGS: 金属钙 ~$3/kg
    "Sr": 6.50,       # USGS: 金属锶 ~$6-7/kg

    # --- 过渡金属 第一周期 (3d) ---
    "Sc": 3500,       # AMES: 金属钪 极稀有 ~$3000-4000/kg
    "Ti": 11.0,       # USGS: 海绵钛 ~$10-12/kg
    "V":  29.0,       # USGS: 五氧化二钒折算 ~$28-32/kg
    "Cr": 10.0,       # LME: 铬铁折算纯铬 ~$9-12/kg
    "Mn": 2.20,       # USGS: 电解锰 ~$2.0-2.5/kg
    "Fe": 0.50,       # LME: 生铁/废钢 ~$0.4-0.6/kg
    "Co": 28.0,       # LME: 金属钴 ~$25-33/kg
    "Ni": 16.0,       # LME: 镍 ~$15-18/kg
    "Cu": 9.20,       # LME: 铜 ~$8.5-10/kg
    "Zn": 2.80,       # LME: 锌 ~$2.5-3.0/kg

    # --- 过渡金属 第二周期 (4d) ---
    "Y":  35.0,       # USGS: 氧化钇折算 ~$30-40/kg
    "Zr": 38.0,       # USGS: 海绵锆 ~$35-45/kg
    "Nb": 75.0,       # USGS: 铌铁折算 ~$70-80/kg
    "Mo": 46.0,       # LME: 钼铁折算 ~$40-55/kg
    "Tc": 100000,     # 放射性人造元素, 极少商业来源, 名义估价
    "Ru": 14500,      # KITCO: 钌 ~$13000-16000/kg (2025)
    "Rh": 145000,     # KITCO: 铑 ~$130000-160000/kg (2025)
    "Pd": 32000,      # KITCO: 钯 ~$28000-35000/kg (2025)
    "Ag": 950,        # KITCO: 银 ~$30/oz → ~$950/kg
    "Cd": 3.00,       # USGS: 镉 ~$2-4/kg
    "In": 250,        # USGS: 铟 ~$200-300/kg

    # --- 过渡金属 第三周期 (5d) ---
    "La": 5.00,       # USGS: 稀土-镧 ~$4-6/kg
    "Hf": 900,        # USGS/AMES: 铪 ~$800-1000/kg
    "Ta": 300,        # USGS: 钽粉 ~$250-350/kg
    "W":  35.0,       # USGS: APT折算 ~$30-40/kg
    "Re": 3000,       # USGS: 铼 ~$2500-3500/kg
    "Os": 12000,      # AMES: 锇 ~$10000-15000/kg (极稀有)
    "Ir": 52000,      # KITCO: 铱 ~$45000-58000/kg (2025)
    "Pt": 31000,      # KITCO: 铂 ~$30000-32000/kg (2025)
    "Au": 88000,      # KITCO: 金 ~$2700/oz → ~$88000/kg (2025 高位)

    # --- 后过渡金属 / 其他 ---
    "Ga": 300,        # USGS: 镓 ~$250-350/kg
    "Sn": 26.0,       # LME: 锡 ~$24-28/kg
    "Pb": 2.10,       # LME: 铅 ~$2.0-2.2/kg
    "Bi": 10.0,       # USGS: 铋 ~$8-12/kg
    "Hg": 30.0,       # 受限有毒，名义价 ~$30/kg
    "Tl": 50.0,       # AMES: 铊 有毒，小量供应 ~$40-60/kg
    "Al": 2.50,       # LME: 铝 ~$2.3-2.7/kg
}

# 原子量 (g/mol)
ATOMIC_MASS = {
    "H": 1.008, "He": 4.003, "Li": 6.941, "Be": 9.012, "B": 10.81,
    "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998, "Ne": 20.180,
    "Na": 22.990, "Mg": 24.305, "Al": 26.982, "Si": 28.086, "P": 30.974,
    "S": 32.065, "Cl": 35.453, "Ar": 39.948, "K": 39.098, "Ca": 40.078,
    "Sc": 44.956, "Ti": 47.867, "V": 50.942, "Cr": 51.996, "Mn": 54.938,
    "Fe": 55.845, "Co": 58.933, "Ni": 58.693, "Cu": 63.546, "Zn": 65.38,
    "Ga": 69.723, "Ge": 72.630, "As": 74.922, "Se": 78.971, "Br": 79.904,
    "Kr": 83.798, "Rb": 85.468, "Sr": 87.62, "Y": 88.906, "Zr": 91.224,
    "Nb": 92.906, "Mo": 95.95, "Tc": 98.0, "Ru": 101.07, "Rh": 102.906,
    "Pd": 106.42, "Ag": 107.868, "Cd": 112.414, "In": 114.818, "Sn": 118.710,
    "Sb": 121.760, "Te": 127.60, "I": 126.904, "Xe": 131.293, "Cs": 132.905,
    "Ba": 137.327, "La": 138.905, "Ce": 140.116, "Pr": 140.908, "Nd": 144.242,
    "Pm": 145.0, "Sm": 150.36, "Eu": 151.964, "Gd": 157.25, "Tb": 158.925,
    "Dy": 162.500, "Ho": 164.930, "Er": 167.259, "Tm": 168.934, "Yb": 173.045,
    "Lu": 174.967, "Hf": 178.49, "Ta": 180.948, "W": 183.84, "Re": 186.207,
    "Os": 190.23, "Ir": 192.217, "Pt": 195.084, "Au": 196.967, "Hg": 200.592,
    "Tl": 204.383, "Pb": 207.2, "Bi": 208.980,
}

def parse_formula(formula: str) -> dict:
    """
    解析化学式，如 'HCu6Os6' -> {'H': 1, 'Cu': 6, 'Os': 6}
    """
    pattern = r'([A-Z][a-z]?)(\d*)'
    matches = re.findall(pattern, formula)
    result = {}
    for elem, count in matches:
        if elem:
            result[elem] = result.get(elem, 0) + (int(count) if count else 1)
    return result


def compute_material_cost(formula: str) -> dict:
    """
    根据化学式计算材料成本。
    返回: {
      'formula_mass_g': 分子式质量(g/mol),
      'cost_per_mol_usd': 每mol的成本(USD),
      'cost_per_kg_usd': 每千克材料的成本(USD/kg),
      'element_costs': {元素: (质量占比, 每kg中该元素的成本USD)},
      'missing_prices': [缺少价格的元素],
    }
    """
    elements = parse_formula(formula)
    if not elements:
        return None

    total_mass = 0.0
    element_mass = {}
    missing = []

    for elem, count in elements.items():
        mass = ATOMIC_MASS.get(elem)
        if mass is None:
            missing.append(elem)
            continue
        m = mass * count
        element_mass[elem] = m
        total_mass += m

    if total_mass == 0:
        return None

    # 每 mol 成本
    cost_per_mol = 0.0
    element_costs = {}
    for elem, count in elements.items():
        price_kg = ELEMENT_PRICE_USD_KG.get(elem)
        mass = ATOMIC_MASS.get(elem)
        if price_kg is None or mass is None:
            if elem not in missing:
                missing.append(elem)
            continue
        # 该元素每 mol 的质量 (kg) × 价格
        elem_mass_kg = mass * count / 1000.0
        elem_cost = elem_mass_kg * price_kg
        cost_per_mol += elem_cost
        frac = element_mass[elem] / total_mass
        element_costs[elem] = (round(frac * 100, 1), round(elem_cost, 4))

    # 每 kg 材料成本
    total_mass_kg = total_mass / 1000.0
    cost_per_kg = cost_per_mol / total_mass_kg if total_mass_kg > 0 else 0

    return {
        'formula_mass_g': round(total_mass, 2),
        'cost_per_mol_usd': round(cost_per_mol, 4),
        'cost_per_kg_usd': round(cost_per_kg, 2),
        'element_costs': element_costs,
        'missing_prices': missing,
    }


def classify_cost(cost_kg):
    """成本等级分类"""
    if cost_kg < 50:
        return "💰 低成本"
    elif cost_kg < 500:
        return "💰💰 中等成本"
    elif cost_kg < 5000:
        return "💰💰💰 较高成本"
    else:
        return "💰💰💰💰 高成本"


def main():
    input_csv = Path(r"d:\mylab\Jiang\HER_commercial_materials_screening.csv")
    df = pd.read_csv(input_csv)

    print(f"读取 {len(df)} 条商用材料记录")

    # 计算每种材料的成本
    costs_kg = []
    costs_mol = []
    formula_masses = []
    cost_grades = []
    dominant_cost_elems = []
    all_missing = set()

    for _, row in df.iterrows():
        composition = row['composition']
        result = compute_material_cost(composition)
        if result is None:
            costs_kg.append(np.nan)
            costs_mol.append(np.nan)
            formula_masses.append(np.nan)
            cost_grades.append("")
            dominant_cost_elems.append("")
            continue

        costs_kg.append(result['cost_per_kg_usd'])
        costs_mol.append(result['cost_per_mol_usd'])
        formula_masses.append(result['formula_mass_g'])
        cost_grades.append(classify_cost(result['cost_per_kg_usd']))

        # 找出成本占比最高的元素
        if result['element_costs']:
            top_elem = max(result['element_costs'].items(),
                           key=lambda x: x[1][1])
            dominant_cost_elems.append(
                f"{top_elem[0]}({top_elem[1][0]}%)")
        else:
            dominant_cost_elems.append("")

        all_missing.update(result['missing_prices'])

    df['formula_mass_g_mol'] = formula_masses
    df['cost_USD_per_kg'] = costs_kg
    df['cost_USD_per_mol'] = costs_mol
    df['cost_grade'] = cost_grades
    df['dominant_cost_element'] = dominant_cost_elems

    if all_missing:
        print(f"[WARN] 以下元素缺少价格数据: {sorted(all_missing)}")

    # 按性能/成本比排序 (|ΔGH|越小越好，成本越低越好)
    df['abs_deltaGH'] = df['true_deltaGH_eV'].abs()
    # 性能成本指数: 越小越好 = |ΔGH| × cost_per_kg (归一化)
    max_cost = df['cost_USD_per_kg'].max()
    df['value_index'] = df['abs_deltaGH'] * (df['cost_USD_per_kg'] / max_cost)

    # 保存完整结果
    output_csv = Path(r"d:\mylab\Jiang\HER_commercial_materials_with_cost.csv")
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"\n完整结果已保存: {output_csv}")

    # 生成Excel多Sheet报告
    xlsx_path = Path(r"d:\mylab\Jiang\HER_commercial_materials_with_cost.xlsx")
    try:
        with pd.ExcelWriter(str(xlsx_path), engine='openpyxl') as writer:
            # Sheet 1: 按原排名
            cols_out = ['dataset', 'sample_id', 'composition', 'elements',
                        'num_atoms', 'true_deltaGH_eV', 'pred_deltaGH_eV',
                        'abs_error_eV', 'grade',
                        'formula_mass_g_mol', 'cost_USD_per_kg',
                        'cost_USD_per_mol', 'cost_grade', 'dominant_cost_element']
            df[cols_out].to_excel(writer, sheet_name='商用材料_含成本', index=True,
                                  index_label='排名')

            # Sheet 2: 性价比排名 (低成本 + 高性能)
            df_value = df.sort_values('value_index').reset_index(drop=True)
            df_value.index = df_value.index + 1
            df_value[cols_out].to_excel(writer, sheet_name='性价比排名',
                                         index=True, index_label='排名')

            # Sheet 3: 低成本优选 (cost < 500 USD/kg)
            df_cheap = df[df['cost_USD_per_kg'] < 500].sort_values(
                'abs_deltaGH').reset_index(drop=True)
            df_cheap.index = df_cheap.index + 1
            df_cheap[cols_out].to_excel(writer, sheet_name='低成本优选',
                                         index=True, index_label='排名')

            # Sheet 4: 价格参考表
            price_rows = []
            for elem, price in sorted(ELEMENT_PRICE_USD_KG.items()):
                mass = ATOMIC_MASS.get(elem, 0)
                price_rows.append({
                    '元素符号': elem,
                    '原子量(g/mol)': mass,
                    '价格(USD/kg)': price,
                    '数据来源': '2025市场行情参考(LME/KITCO/USGS MCS)',
                })
            pd.DataFrame(price_rows).to_excel(writer, sheet_name='元素价格参考表',
                                               index=False)

        print(f"Excel报告已保存: {xlsx_path}")
    except ImportError:
        print("[INFO] openpyxl未安装，跳过Excel输出。")

    # 统计输出
    print(f"\n{'='*80}")
    print("                   材料成本统计概览")
    print(f"{'='*80}")
    print(f"  总材料数: {len(df)}")
    print(f"  成本范围: ${df['cost_USD_per_kg'].min():.1f} ~ ${df['cost_USD_per_kg'].max():.1f} /kg")
    print(f"  中位成本: ${df['cost_USD_per_kg'].median():.1f} /kg")
    for grade in df['cost_grade'].unique():
        cnt = (df['cost_grade'] == grade).sum()
        print(f"  {grade}: {cnt} 种")

    # Top-10 性价比最优
    print(f"\n{'='*80}")
    print("          Top-15 性价比最优材料 (低成本 + 高活性)")
    print(f"{'='*80}")
    df_best = df.sort_values('value_index').head(15).reset_index(drop=True)
    df_best.index = df_best.index + 1
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 220)
    print(df_best[['composition', 'elements', 'true_deltaGH_eV',
                    'pred_deltaGH_eV', 'cost_USD_per_kg', 'cost_grade',
                    'dominant_cost_element', 'grade']].to_string())
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
