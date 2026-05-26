#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
综合分析报告
生成所有分析的综合摘要
"""

import pandas as pd
import argparse
from pathlib import Path
import sys

# 添加analysis目录到路径
analysis_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(analysis_dir))

from plot_utils import create_summary_plot

def generate_summary(results_csv: str, output_dir: str):
    """生成综合分析摘要"""
    df = pd.read_csv(results_csv)
    
    # 过滤有效数据
    valid_df = df[(df['test_mae'] > 0) & (df['test_rmse'] > 0)]
    
    if valid_df.empty:
        print("没有有效数据")
        return
    
    print(f"生成综合分析摘要: {len(valid_df)} 个有效实验")
    
    # 创建综合摘要图
    output_path = Path(output_dir) / "summary_analysis.png"
    create_summary_plot(valid_df, str(output_path))
    
    # 生成统计摘要
    print("\n" + "="*80)
    print("实验统计摘要")
    print("="*80)
    
    print(f"总实验数: {len(df)}")
    print(f"有效实验数: {len(valid_df)}")
    print(f"成功率: {len(valid_df)/len(df)*100:.1f}%")
    
    print(f"\n性能统计:")
    print(f"MAE - 均值: {valid_df['test_mae'].mean():.6f} eV, 标准差: {valid_df['test_mae'].std():.6f} eV")
    print(f"MAE - 最小值: {valid_df['test_mae'].min():.6f} eV, 最大值: {valid_df['test_mae'].max():.6f} eV")
    print(f"RMSE - 均值: {valid_df['test_rmse'].mean():.6f} eV, 标准差: {valid_df['test_rmse'].std():.6f} eV")
    print(f"RMSE - 最小值: {valid_df['test_rmse'].min():.6f} eV, 最大值: {valid_df['test_rmse'].max():.6f} eV")
    
    if 'params' in valid_df.columns and valid_df['params'].max() > 0:
        print(f"\n模型统计:")
        print(f"参数量 - 均值: {valid_df['params'].mean():,.0f}, 标准差: {valid_df['params'].std():,.0f}")
        print(f"参数量 - 最小值: {valid_df['params'].min():,.0f}, 最大值: {valid_df['params'].max():,.0f}")
    
    # 最佳配置
    best_config = valid_df.loc[valid_df['test_mae'].idxmin()]
    print(f"\n最佳配置:")
    print(f"MAE: {best_config['test_mae']:.6f} eV")
    print(f"RMSE: {best_config['test_rmse']:.6f} eV")
    
    # 保存最佳配置
    best_config_path = Path(output_dir) / "best_config.csv"
    best_config.to_frame().T.to_csv(best_config_path, index=False)
    print(f"最佳配置已保存到: {best_config_path}")
    
    # 参数重要性分析
    print(f"\n参数重要性分析:")
    numeric_cols = valid_df.select_dtypes(include=[int, float]).columns
    numeric_cols = [col for col in numeric_cols if col not in ['test_mae', 'test_rmse', 'test_loss', 'seed']]
    
    correlations = []
    for col in numeric_cols:
        corr_mae = valid_df[col].corr(valid_df['test_mae'])
        corr_rmse = valid_df[col].corr(valid_df['test_rmse'])
        correlations.append({
            'parameter': col,
            'correlation_mae': corr_mae,
            'correlation_rmse': corr_rmse,
            'abs_correlation_mae': abs(corr_mae)
        })
    
    corr_df = pd.DataFrame(correlations)
    corr_df = corr_df.sort_values('abs_correlation_mae', ascending=False)
    
    print("参数与MAE的相关性 (按重要性排序):")
    for _, row in corr_df.head(10).iterrows():
        print(f"  {row['parameter']}: {row['correlation_mae']:.4f}")
    
    # 保存相关性分析
    corr_path = Path(output_dir) / "parameter_correlations.csv"
    corr_df.to_csv(corr_path, index=False)
    print(f"参数相关性分析已保存到: {corr_path}")
    
    # 生成HTML报告
    html_path = Path(output_dir) / "summary_report.html"
    generate_html_report(valid_df, best_config, corr_df, html_path)
    print(f"HTML报告已保存到: {html_path}")

def generate_html_report(df, best_config, corr_df, html_path):
    """生成HTML格式的报告"""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>EquiformerV2 消融实验报告</title>
        <meta charset="utf-8">
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1, h2 {{ color: #333; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .metric {{ background-color: #e7f3ff; }}
            .best {{ background-color: #d4edda; }}
        </style>
    </head>
    <body>
        <h1>EquiformerV2 消融实验报告</h1>
        
        <h2>实验概览</h2>
        <table>
            <tr><th>指标</th><th>值</th></tr>
            <tr><td>总实验数</td><td>{len(df)}</td></tr>
            <tr><td>有效实验数</td><td>{len(df)}</td></tr>
            <tr><td>成功率</td><td>{len(df)/len(df)*100:.1f}%</td></tr>
        </table>
        
        <h2>性能统计</h2>
        <table>
            <tr><th>指标</th><th>均值</th><th>标准差</th><th>最小值</th><th>最大值</th></tr>
            <tr class="metric"><td>MAE (eV)</td><td>{df['test_mae'].mean():.6f}</td><td>{df['test_mae'].std():.6f}</td><td>{df['test_mae'].min():.6f}</td><td>{df['test_mae'].max():.6f}</td></tr>
            <tr class="metric"><td>RMSE (eV)</td><td>{df['test_rmse'].mean():.6f}</td><td>{df['test_rmse'].std():.6f}</td><td>{df['test_rmse'].min():.6f}</td><td>{df['test_rmse'].max():.6f}</td></tr>
        </table>
        
        <h2>最佳配置</h2>
        <table class="best">
            <tr><th>参数</th><th>值</th></tr>
    """
    
    for param in ['test_mae', 'test_rmse', 'num_layers', 'sphere_channels', 'num_heads', 'grid_resolution']:
        if param in best_config:
            html_content += f"<tr><td>{param}</td><td>{best_config[param]}</td></tr>\n"
    
    html_content += f"""
        </table>
        
        <h2>参数重要性 (与MAE的相关性)</h2>
        <table>
            <tr><th>参数</th><th>与MAE相关性</th><th>与RMSE相关性</th></tr>
    """
    
    for _, row in corr_df.head(10).iterrows():
        html_content += f"<tr><td>{row['parameter']}</td><td>{row['correlation_mae']:.4f}</td><td>{row['correlation_rmse']:.4f}</td></tr>\n"
    
    html_content += """
        </table>
        
        <p><em>报告生成时间: """ + pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S') + """</em></p>
    </body>
    </html>
    """
    
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

def main():
    parser = argparse.ArgumentParser(description='综合分析报告')
    parser.add_argument('--results_csv', type=str, required=True, help='实验结果CSV文件')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    
    args = parser.parse_args()
    
    generate_summary(args.results_csv, args.output_dir)

if __name__ == "__main__":
    main()
