#!/usr/bin/env python3
"""
推理速度测试脚本
测量 EquiformerV2 原版和 Eqv2-Lite 在不同 batch_size 下的推理速度
"""

import json
import sys
import torch
import torch.nn as nn
import numpy as np
import time
from pathlib import Path
from torch.utils.data import DataLoader
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

# 添加项目路径
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from enhanced_equiformer_v2 import EnhancedEquiformerV2, HydrogenDataset, custom_collate_fn
from standalone_equiformer_v2 import StandaloneEquiformerV2, SimpleGaussianSmearing


def create_standalone_model_with_custom_gaussians(config):
    """创建 StandaloneEquiformerV2 模型，支持自定义 num_gaussians"""
    model = StandaloneEquiformerV2(
        max_radius=config['max_radius'],
        max_neighbors=config['max_neighbors'],
        max_num_elements=config['max_num_elements'],
        num_layers=config['num_layers'],
        sphere_channels=config['sphere_channels'],
        attn_hidden_channels=config['attn_hidden_channels'],
        num_heads=config['num_heads'],
        attn_alpha_channels=config['attn_alpha_channels'],
        attn_value_channels=config['attn_value_channels'],
        ffn_hidden_channels=config['ffn_hidden_channels'],
        lmax_list=config['lmax_list'],
        mmax_list=config['mmax_list'],
        grid_resolution=config['grid_resolution'],
        edge_channels=config['edge_channels'],
        use_atom_edge_embedding=config['use_atom_edge_embedding'],
        share_atom_edge_embedding=config['share_atom_edge_embedding'],
        alpha_drop=config['alpha_drop'],
        drop_path_rate=config['drop_path_rate'],
        proj_drop=config['proj_drop'],
        norm_type=config['norm_type']
    )
    
    # 替换 distance_expansion 以支持自定义 num_gaussians
    if 'num_gaussians' in config:
        num_gaussians = config['num_gaussians']
        old_num_output = model.distance_expansion.num_output
        model.distance_expansion = SimpleGaussianSmearing(
            0.0, model.max_radius, num_gaussians, 2.0
        )
        
        # 更新 edge_channels_list
        model.edge_channels_list = [model.distance_expansion.num_output] + [model.edge_channels] * 2
        
        # 如果通道数改变了，需要重新创建 edge_degree_embedding 和 blocks
        if old_num_output != num_gaussians:
            from equiformer_v2.nets.equiformer_v2.input_block import EdgeDegreeEmbedding
            from equiformer_v2.nets.equiformer_v2.transformer_block import TransBlockV2
            
            # 如果使用了 atom_edge_embedding，需要更新
            if model.share_atom_edge_embedding and model.use_atom_edge_embedding:
                model.edge_channels_list[0] = model.edge_channels_list[0] + 2 * model.edge_channels_list[-1]
            
            # 重新创建 edge_degree_embedding
            model.edge_degree_embedding = EdgeDegreeEmbedding(
                model.sphere_channels,
                model.lmax_list,
                model.mmax_list,
                model.SO3_rotation,
                model.mappingReduced,
                model.max_num_elements,
                model.edge_channels_list,
                model.block_use_atom_edge_embedding,
                rescale_factor=model._avg_degree
            )
            
            # 重新创建所有 Transformer blocks
            model.blocks = nn.ModuleList()
            for i in range(model.num_layers):
                block = TransBlockV2(
                    model.sphere_channels,
                    config['attn_hidden_channels'],
                    config['num_heads'],
                    config['attn_alpha_channels'],
                    config['attn_value_channels'],
                    config['ffn_hidden_channels'],
                    model.sphere_channels,
                    model.lmax_list,
                    model.mmax_list,
                    model.SO3_rotation,
                    model.mappingReduced,
                    model.SO3_grid,
                    model.max_num_elements,
                    model.edge_channels_list,
                    model.block_use_atom_edge_embedding,
                    use_m_share_rad=False,
                    attn_activation='scaled_silu',
                    use_s2_act_attn=False,
                    use_attn_renorm=True,
                    ffn_activation='scaled_silu',
                    use_gate_act=False,
                    use_grid_mlp=config.get('use_grid_mlp', False),
                    use_sep_s2_act=config.get('use_sep_s2_act', True),
                    norm_type=config['norm_type'],
                    alpha_drop=config['alpha_drop'],
                    drop_path_rate=config['drop_path_rate'],
                    proj_drop=config['proj_drop']
                )
                model.blocks.append(block)
    
    return model


def create_model(config_path):
    """根据配置文件创建模型"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    model_config = config['model_config']
    model_type = config['model_type']
    
    if model_type == 'enhanced':
        model = EnhancedEquiformerV2(
            max_radius=model_config['max_radius'],
            max_neighbors=model_config['max_neighbors'],
            max_num_elements=model_config['max_num_elements'],
            num_layers=model_config['num_layers'],
            sphere_channels=model_config['sphere_channels'],
            attn_hidden_channels=model_config['attn_hidden_channels'],
            num_heads=model_config['num_heads'],
            attn_alpha_channels=model_config['attn_alpha_channels'],
            attn_value_channels=model_config['attn_value_channels'],
            ffn_hidden_channels=model_config['ffn_hidden_channels'],
            lmax_list=model_config['lmax_list'],
            mmax_list=model_config['mmax_list'],
            grid_resolution=model_config['grid_resolution'],
            edge_channels=model_config['edge_channels'],
            use_atom_edge_embedding=model_config['use_atom_edge_embedding'],
            share_atom_edge_embedding=model_config['share_atom_edge_embedding'],
            alpha_drop=model_config['alpha_drop'],
            drop_path_rate=model_config['drop_path_rate'],
            proj_drop=model_config['proj_drop'],
            norm_type=model_config['norm_type']
        )
    elif model_type == 'standalone':
        model = create_standalone_model_with_custom_gaussians(model_config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model, config


def count_parameters(model):
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters())


def _move_batch_to_device(batch, device):
    """将batch数据移动到指定设备"""
    if hasattr(batch, 'pos'):
        batch.pos = batch.pos.to(device, non_blocking=True)
    if hasattr(batch, 'atomic_numbers'):
        batch.atomic_numbers = batch.atomic_numbers.to(device, non_blocking=True)
    if hasattr(batch, 'edge_index'):
        batch.edge_index = batch.edge_index.to(device, non_blocking=True)
    if hasattr(batch, 'edge_distance'):
        batch.edge_distance = batch.edge_distance.to(device, non_blocking=True)
    if hasattr(batch, 'batch'):
        batch.batch = batch.batch.to(device, non_blocking=True)
    if hasattr(batch, 'natoms'):
        batch.natoms = batch.natoms.to(device, non_blocking=True)
    return batch

def measure_inference_speed(model, data_loader, device, num_warmup=10, num_runs=50):
    """测量推理速度"""
    model.eval()
    
    # 预热
    print(f"  🔥 预热中 ({num_warmup} batches)...", end=' ', flush=True)
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if i >= num_warmup:
                break
            batch = _move_batch_to_device(batch, device)
            _ = model(batch)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    
    print("✅ 完成")
    print(f"  🚀 开始测试 ({num_runs} batches)...")
    
    # 测试
    inference_times = []
    total_samples = 0
    
    with torch.no_grad():
        iterator = tqdm(enumerate(data_loader), total=min(num_runs, len(data_loader)), 
                       desc="  📊 推理测试", leave=True, ncols=120, 
                       bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        for i, batch in iterator:
            if i >= num_runs:
                break
            
            batch = _move_batch_to_device(batch, device)
            
            # 测量推理时间
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start_time = time.time()
            _ = model(batch)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            # 计算 batch 中的样本数
            if hasattr(batch, 'natoms'):
                total_samples += len(batch.natoms)
            else:
                total_samples += 1
    
    avg_time = np.mean(inference_times)
    std_time = np.std(inference_times)
    avg_time_per_sample = avg_time / (total_samples / len(inference_times))
    
    return {
        'avg_time': avg_time,
        'std_time': std_time,
        'avg_time_per_sample': avg_time_per_sample,
        'samples_per_second': 1.0 / avg_time_per_sample if avg_time_per_sample > 0 else 0,
        'total_samples': total_samples,
        'num_runs': len(inference_times)
    }


def test_model_inference(config_path, batch_size=64):
    """测试模型在指定 batch_size 下的推理速度"""
    print(f"\n{'='*60}")
    print(f"🔬 测试模型: {config_path.name}")
    print(f"{'='*60}")
    
    # 创建模型
    print("  📦 创建模型...", end=' ', flush=True)
    model, config = create_model(config_path)
    print("✅")
    
    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  🖥️  设备: {device}")
    model = model.to(device)
    
    # 计算参数量
    num_params = count_parameters(model)
    print(f"  📊 模型参数量: {num_params:,} ({num_params/1e6:.2f}M)")
    
    # 数据集
    print("  📂 加载数据集...", end=' ', flush=True)
    val_lmdb_path = project_root / "datasets" / "custom_hydrogen" / "val.lmdb"
    val_dataset = HydrogenDataset(str(val_lmdb_path))
    print(f"✅ ({len(val_dataset)} 个样本)")
    
    # 创建 DataLoader
    print(f"  🔄 测试 batch_size={batch_size}...")
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Windows 兼容性
        pin_memory=True if torch.cuda.is_available() else False,
        collate_fn=custom_collate_fn,
    )
    
    try:
        result = measure_inference_speed(model, val_loader, device)
        
        print(f"    ✅ 平均推理时间: {result['avg_time']:.4f} ± {result['std_time']:.4f} 秒/batch")
        print(f"    ✅ 平均每样本时间: {result['avg_time_per_sample']*1000:.2f} ms/sample")
        print(f"    ✅ 推理速度: {result['samples_per_second']:.2f} samples/秒")
    except Exception as e:
        print(f"    ❌ [ERROR] batch_size={batch_size} 测试失败: {e}")
        import traceback
        traceback.print_exc()
        val_dataset.close()
        return None
    
    val_dataset.close()
    
    return {
        'model_name': config['model_name'],
        'num_parameters': num_params,
        'batch_size': batch_size,
        'avg_time': result['avg_time'],
        'std_time': result['std_time'],
        'avg_time_per_sample': result['avg_time_per_sample'],
        'samples_per_second': result['samples_per_second'],
        'total_samples': result['total_samples'],
        'num_runs': result['num_runs']
    }


def main():
    """主函数"""
    print("="*60)
    print("🚀 EquiformerV2 推理速度测试")
    print("="*60)
    
    benchmark_dir = Path(__file__).parent
    
    configs = [
        benchmark_dir / "eqv2_best.json",
        benchmark_dir / "eq_lite.json"
    ]
    
    batch_size = 64  # 只测试一个 batch_size
    
    all_results = []
    
    # 使用tqdm显示模型测试进度
    configs_pbar = tqdm(configs, desc="📋 模型测试进度", leave=True, ncols=100)
    for config_path in configs_pbar:
        if not config_path.exists():
            print(f"  ❌ [ERROR] 配置文件不存在: {config_path}")
            continue
        
        configs_pbar.set_description(f"📋 测试: {config_path.stem}")
        try:
            result = test_model_inference(config_path, batch_size)
            if result is not None:
                all_results.append(result)
                configs_pbar.set_postfix({'状态': '✅ 完成'})
            else:
                configs_pbar.set_postfix({'状态': '❌ 失败'})
        except Exception as e:
            print(f"  ❌ [ERROR] 测试失败: {e}")
            import traceback
            traceback.print_exc()
            configs_pbar.set_postfix({'状态': '❌ 失败'})
            continue
    
    # 保存结果
    if all_results:
        output_file = benchmark_dir / "infer_speed_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\n✅ 推理速度测试结果已保存到: {output_file}")
        
        # 打印对比
        print("\n" + "="*60)
        print("📊 推理速度对比汇总 (batch_size=64)")
        print("="*60)
        for r in all_results:
            print(f"\n🔹 {r['model_name']}:")
            print(f"   参数量: {r['num_parameters']/1e6:.2f}M")
            print(f"   推理速度: {r['samples_per_second']:.2f} samples/秒")
            print(f"   每样本时间: {r['avg_time_per_sample']*1000:.2f} ms")
        
        # 计算加速比
        if len(all_results) == 2:
            original = all_results[0] if 'Original' in all_results[0]['model_name'] or '原版' in all_results[0]['model_name'] else all_results[1]
            lite = all_results[1] if all_results[0] == original else all_results[0]
            
            speedup = original['avg_time_per_sample'] / lite['avg_time_per_sample']
            print(f"\n🚀 加速比: {speedup:.2f}x (Eqv2-Lite 相对于原版)")
    else:
        print("\n⚠️  没有成功完成任何测试")


if __name__ == "__main__":
    main()

