#!/usr/bin/env python3
"""
收集 num_layers ∈ {1,4} 且 grid_resolution=12 的已完成 seed=1、2 测试结果，写入 CSV。
输出: experiments/ablation/results_layers_missing.csv
"""
import json
import csv
from pathlib import Path

def main():
    base = Path('experiments/2025-09-10_run1/ablation')
    rows = []
    for num_layers in (1, 4):
        exp_dir = base / f'num_layers={num_layers}__sphere_channels=64__num_heads=4__grid_resolution=12__edge_channels=64'
        if not exp_dir.exists():
            continue
        for seed in (1, 2):
            fp = exp_dir / f'seed={seed}' / 'logs' / 'enhanced_equiformer_v2_test_results.json'
            if fp.exists():
                data = json.loads(fp.read_text(encoding='utf-8'))
                rows.append([
                    num_layers, 64, 4, 12, 64, 0, 0, 0, 0, 1.0, 0,
                    data.get('test_mae', -1.0),
                    data.get('test_rmse', -1.0),
                    data.get('test_loss', -1.0),
                    -1, -1.0, -1.0, seed, ""
                ])

    outp = Path('experiments/ablation/results_layers_missing.csv')
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow([
            'num_layers','sphere_channels','num_heads','grid_resolution','edge_channels',
            'eSCN','attn_renorm','sep_s2','sep_ln','data_ratio','pretrained',
            'test_mae','test_rmse','test_loss','params','latency_ms','throughput','seed','error'
        ])
        w.writerows(rows)
    print(f'Collected rows: {len(rows)}')

if __name__ == '__main__':
    main()


