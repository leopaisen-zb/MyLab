#!/usr/bin/env python3
"""
Eqv2-Lite radial (num_gaussians) 消融实验脚本
测试不同高斯基函数数量对模型性能的影响
"""

import json
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tqdm import tqdm
import warnings
from pathlib import Path
import time
import threading
warnings.filterwarnings('ignore')

# 添加项目路径（脚本在src目录下，所以parent.parent是项目根目录）
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

# 导入必要组件
from standalone_equiformer_v2 import (
    StandaloneEquiformerV2,
    HydrogenDataset,
    custom_collate_fn,
    SimpleGaussianSmearing
)


class EquiformerV2Trainer:
    """EquiformerV2 训练器（用于 radial 消融）"""

    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        # 设备与 AMP
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.backends.cudnn.benchmark = True
            self.use_amp = hasattr(torch.cuda.amp, 'autocast')
            self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        else:
            self.device = torch.device('cpu')
            self.use_amp = False
            self.scaler = None

        self.model.to(self.device)

        # 优化器与调度器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['lr'],
            weight_decay=config.get('weight_decay', 1e-4),
        )
        self.criterion = nn.MSELoss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.8, patience=10, min_lr=1e-6
        )

        # 归一化参数
        self.target_mean = config.get('target_mean', 0.0)
        self.target_std = config.get('target_std', 1.0)

        # 记录
        self.train_losses = []
        self.val_losses = []
        self.val_r2s = []

    def _move_to_device(self, batch):
        if hasattr(batch, 'pos'):
            batch.pos = batch.pos.to(self.device, non_blocking=True)
        if hasattr(batch, 'atomic_numbers'):
            batch.atomic_numbers = batch.atomic_numbers.to(self.device, non_blocking=True)
        if hasattr(batch, 'edge_index'):
            batch.edge_index = batch.edge_index.to(self.device, non_blocking=True)
        if hasattr(batch, 'edge_distance'):
            batch.edge_distance = batch.edge_distance.to(self.device, non_blocking=True)
        if hasattr(batch, 'batch'):
            batch.batch = batch.batch.to(self.device, non_blocking=True)
        if hasattr(batch, 'natoms'):
            batch.natoms = batch.natoms.to(self.device, non_blocking=True)
        if hasattr(batch, 'y'):
            batch.y = batch.y.to(self.device, non_blocking=True)
        return batch

    def normalize_target(self, y):
        return (y - self.target_mean) / self.target_std

    def denormalize_target(self, y):
        return y * self.target_std + self.target_mean

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 添加调试信息（只在第一个epoch的第一个batch打印）
        is_first_batch = (len(self.train_losses) == 0)
        if is_first_batch:
            print(f"  🔍 [DEBUG] 开始训练epoch，训练集batch数: {len(self.train_loader)}")

        for batch_idx, batch in enumerate(tqdm(self.train_loader, desc="训练", leave=False)):
            if is_first_batch and batch_idx == 0:
                print(f"  🔍 [DEBUG] 处理第一个训练batch，batch大小: {len(batch.y) if hasattr(batch, 'y') else 'N/A'}")
                if torch.cuda.is_available():
                    print(f"  🔍 [DEBUG] GPU内存 (训练前): {torch.cuda.memory_allocated()/1024**3:.2f}GB")
            
            # 每100个batch打印一次调试信息
            if batch_idx > 0 and batch_idx % 100 == 0:
                if torch.cuda.is_available():
                    print(f"  🔍 [DEBUG] Batch {batch_idx}: GPU内存={torch.cuda.memory_allocated()/1024**3:.2f}GB, "
                          f"平均loss={total_loss/max(n_batches,1):.4f}")
            
            # 监控每个batch的处理时间（如果超过5秒则警告）
            batch_start_time = time.time()
            
            try:
                t0 = time.time()
                batch = self._move_to_device(batch)
                target = self.normalize_target(batch.y)
                t1 = time.time()

                self.optimizer.zero_grad()

                t2 = time.time()
                if self.use_amp and self.scaler is not None:
                    with torch.cuda.amp.autocast():
                        pred = self.model(batch)
                        if pred.dim() > 1:
                            pred = pred.squeeze(-1)
                        if target.dim() > 1:
                            target = target.squeeze(-1)
                        loss = self.criterion(pred, target)
                    t3 = time.time()
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    t4 = time.time()
                else:
                    pred = self.model(batch)
                    if pred.dim() > 1:
                        pred = pred.squeeze(-1)
                    if target.dim() > 1:
                        target = target.squeeze(-1)
                    loss = self.criterion(pred, target)
                    t3 = time.time()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    t4 = time.time()

                total_loss += loss.item()
                n_batches += 1
                
                batch_time = time.time() - batch_start_time
                # 如果单个batch处理时间超过5秒，打印警告
                if batch_time > 5.0:
                    print(f"  ⚠️ [DEBUG] Batch {batch_idx} 处理时间较长: {batch_time:.2f}秒 "
                          f"(数据移动={t1-t0:.2f}s, 前向={t3-t2:.2f}s, 反向={t4-t3:.2f}s)")
                # 如果超过30秒，认为卡住了，跳过这个batch
                if batch_time > 30.0:
                    print(f"  ⚠️ [DEBUG] Batch {batch_idx} 处理时间过长({batch_time:.2f}秒)，跳过此batch")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"  ⚠️ [DEBUG] Batch {batch_idx}: GPU内存不足，跳过此batch")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                else:
                    print(f"  ❌ [DEBUG] Batch {batch_idx} 错误: {e}")
                    import traceback
                    traceback.print_exc()
                    # 继续处理下一个batch，不中断训练
                    continue
            except Exception as e:
                print(f"  ❌ [DEBUG] Batch {batch_idx} 未知错误: {e}")
                import traceback
                traceback.print_exc()
                continue

        return total_loss / max(n_batches, 1)

    def evaluate(self):
        self.model.eval()
        preds, targs = [], []
        total_loss = 0.0
        valid_batches = 0

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 添加调试信息（只在第一次评估时打印）
        is_first_eval = (len(self.val_losses) == 0)
        if is_first_eval:
            print(f"  🔍 [DEBUG] 开始评估，验证集batch数: {len(self.val_loader)}")

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="评估", leave=False)):
                if is_first_eval and batch_idx == 0:
                    print(f"  🔍 [DEBUG] 处理第一个验证batch，batch大小: {len(batch.y) if hasattr(batch, 'y') else 'N/A'}")
                    if torch.cuda.is_available():
                        print(f"  🔍 [DEBUG] GPU内存 (评估前): {torch.cuda.memory_allocated()/1024**3:.2f}GB")
                batch = self._move_to_device(batch)
                target = self.normalize_target(batch.y)

                if self.use_amp and self.scaler is not None:
                    with torch.cuda.amp.autocast():
                        pred = self.model(batch)
                else:
                    pred = self.model(batch)

                if pred.dim() > 1:
                    pred = pred.squeeze(-1)
                if target.dim() > 1:
                    target = target.squeeze(-1)

                loss = self.criterion(pred, target)

                pred_denorm = self.denormalize_target(pred).cpu().numpy()
                target_denorm = batch.y.cpu().numpy()

                preds.extend(pred_denorm.tolist())
                targs.extend(target_denorm.tolist())
                total_loss += loss.item()
                valid_batches += 1

        if valid_batches == 0:
            return None

        preds = np.array(preds)
        targs = np.array(targs)

        mae = mean_absolute_error(targs, preds)
        mse = mean_squared_error(targs, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(targs, preds)

        return {
            'loss': total_loss / valid_batches,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
        }

    def train(self, num_epochs: int):
        best_r2 = -float('inf')
        best_state = None

        for epoch in range(num_epochs):
            print(f"\n🔍 [DEBUG] 开始 Epoch {epoch+1}/{num_epochs}")
            if torch.cuda.is_available():
                print(f"  - GPU内存 (epoch开始): {torch.cuda.memory_allocated()/1024**3:.2f}GB")
            
            try:
                train_loss = self.train_epoch()
                print(f"  ✅ [DEBUG] Epoch {epoch+1} 训练完成，平均loss: {train_loss:.4f}")
            except Exception as e:
                print(f"  ❌ [DEBUG] Epoch {epoch+1} 训练失败: {e}")
                import traceback
                traceback.print_exc()
                continue
            
            print(f"  🔍 [DEBUG] 开始评估 Epoch {epoch+1}...")
            try:
                val_metrics = self.evaluate()
                if val_metrics is None:
                    print(f"  ⚠️ [DEBUG] Epoch {epoch+1} 评估失败，跳过")
                    continue
                print(f"  ✅ [DEBUG] Epoch {epoch+1} 评估完成，R²: {val_metrics['r2']:.4f}")
            except Exception as e:
                print(f"  ❌ [DEBUG] Epoch {epoch+1} 评估失败: {e}")
                import traceback
                traceback.print_exc()
                continue

            self.train_losses.append(train_loss)
            self.val_losses.append(val_metrics['loss'])
            self.val_r2s.append(val_metrics['r2'])

            self.scheduler.step(val_metrics['loss'])

            if val_metrics['r2'] > best_r2:
                best_r2 = val_metrics['r2']
                best_state = self.model.state_dict().copy()
                print(f"  🎯 [DEBUG] 新的最佳R²: {best_r2:.4f}")

            if (epoch + 1) % 5 == 0 or epoch == 0:
                lr = self.optimizer.param_groups[0]['lr']
                print(
                    f"Epoch {epoch+1:3d}/{num_epochs}: "
                    f"Train Loss={train_loss:.4f}, "
                    f"Val Loss={val_metrics['loss']:.4f}, "
                    f"Val R²={val_metrics['r2']:.4f}, "
                    f"Val MAE={val_metrics['mae']:.4f}, "
                    f"LR={lr:.6f}"
                )
            
            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if best_state is not None:
            print(f"🔍 [DEBUG] 加载最佳模型 (R²={best_r2:.4f})")
            self.model.load_state_dict(best_state)

        print(f"🔍 [DEBUG] 进行最终评估...")
        return self.evaluate()


# 创建一个支持自定义 num_gaussians 的 StandaloneEquiformerV2 子类
class StandaloneEquiformerV2WithGaussians(StandaloneEquiformerV2):
    """支持自定义 num_gaussians 的 StandaloneEquiformerV2"""
    
    def __init__(self, num_gaussians=128, **kwargs):
        # 先调用父类初始化（会创建默认的 distance_expansion）
        super().__init__(**kwargs)
        
        # 替换 distance_expansion
        self.distance_expansion = SimpleGaussianSmearing(0.0, self.max_radius, num_gaussians, 2.0)
        
        # 更新 edge_channels_list
        self.edge_channels_list = [self.distance_expansion.num_output] + [self.edge_channels] * 2
        
        # 如果使用了 atom_edge_embedding，需要更新
        if self.share_atom_edge_embedding and self.use_atom_edge_embedding:
            self.edge_channels_list[0] = self.edge_channels_list[0] + 2 * self.edge_channels_list[-1]
        
        # 重新创建 edge_degree_embedding（因为 edge_channels_list 改变了）
        from equiformer_v2.nets.equiformer_v2.input_block import EdgeDegreeEmbedding
        self.edge_degree_embedding = EdgeDegreeEmbedding(
            self.sphere_channels,
            self.lmax_list,
            self.mmax_list,
            self.SO3_rotation,
            self.mappingReduced,
            self.max_num_elements,
            self.edge_channels_list,
            self.block_use_atom_edge_embedding,
            rescale_factor=self._avg_degree
        )
        
        # 重新创建所有 Transformer blocks（因为 edge_channels_list 改变了）
        from equiformer_v2.nets.equiformer_v2.transformer_block import TransBlockV2
        self.blocks = nn.ModuleList()
        for i in range(self.num_layers):
            block = TransBlockV2(
                self.sphere_channels,
                64,  # attn_hidden_channels
                8,   # num_heads
                32,  # attn_alpha_channels
                16,  # attn_value_channels
                256, # ffn_hidden_channels
                self.sphere_channels,
                self.lmax_list,
                self.mmax_list,
                self.SO3_rotation,
                self.mappingReduced,
                self.SO3_grid,
                self.max_num_elements,
                self.edge_channels_list,
                self.block_use_atom_edge_embedding,
                use_m_share_rad=False,
                attn_activation='scaled_silu',
                use_s2_act_attn=False,
                use_attn_renorm=True,
                ffn_activation='scaled_silu',
                use_gate_act=False,
                use_grid_mlp=False,
                use_sep_s2_act=True,
                norm_type='layer_norm_sh',
                alpha_drop=0.1,
                drop_path_rate=0.05,
                proj_drop=0.0
            )
            self.blocks.append(block)

def run_single_experiment(num_gaussians: int, output_root: Path, config, train_loader, val_loader):
    """运行单个 num_gaussians 实验，并保存 metrics.json"""
    print("\n" + "=" * 60)
    print(f"🔬 开始实验: num_gaussians = {num_gaussians}")
    print("=" * 60)

    # 使用支持自定义 num_gaussians 的模型类
    print(f"🔍 [DEBUG] 开始创建模型 (num_gaussians={num_gaussians})...")
    model = StandaloneEquiformerV2WithGaussians(
        num_gaussians=num_gaussians,  # 消融变量
        max_radius=12.0,
        max_neighbors=20,
        max_num_elements=90,
        num_layers=6,
        sphere_channels=128,
        attn_hidden_channels=64,
        num_heads=8,
        attn_alpha_channels=32,
        attn_value_channels=16,
        ffn_hidden_channels=256,
        lmax_list=[4],
        mmax_list=[2],
        grid_resolution=18,
        edge_channels=128,
        use_atom_edge_embedding=True,
        share_atom_edge_embedding=False,
        alpha_drop=0.1,
        drop_path_rate=0.05,
        proj_drop=0.0,
    )

    print(f"✅ 模型已创建，参数数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   距离扩展基函数数量: {num_gaussians}")
    
    # 将模型移动到GPU（如果可用）
    if torch.cuda.is_available():
        print("🔍 [DEBUG] 移动模型到GPU...")
        model = model.cuda()
        print(f"  - GPU内存: {torch.cuda.memory_allocated()/1024**3:.2f}GB / {torch.cuda.memory_reserved()/1024**3:.2f}GB")
    
    # 测试模型是否能正常前向传播
    print("🔍 [DEBUG] 测试模型前向传播...")
    try:
        print("  - 设置模型为评估模式...")
        model.eval()
        
        print("  - 获取测试batch...")
        test_batch = next(iter(val_loader))
        print(f"  - Batch信息: pos.shape={test_batch.pos.shape if hasattr(test_batch, 'pos') else 'N/A'}, "
              f"atomic_numbers.shape={test_batch.atomic_numbers.shape if hasattr(test_batch, 'atomic_numbers') else 'N/A'}")
        
        print("  - 移动数据到GPU..." if torch.cuda.is_available() else "  - 使用CPU...")
        if torch.cuda.is_available():
            test_batch.pos = test_batch.pos.cuda()
            test_batch.atomic_numbers = test_batch.atomic_numbers.cuda()
            test_batch.edge_index = test_batch.edge_index.cuda()
            test_batch.edge_distance = test_batch.edge_distance.cuda()
            test_batch.batch = test_batch.batch.cuda()
            test_batch.natoms = test_batch.natoms.cuda()
            print(f"  - GPU内存: {torch.cuda.memory_allocated()/1024**3:.2f}GB / {torch.cuda.memory_reserved()/1024**3:.2f}GB")
        
        print("  - 执行前向传播...")
        with torch.no_grad():
            output = model(test_batch)
        print(f"  - 输出形状: {output.shape}")
        print("✅ [DEBUG] 模型前向传播测试通过")
    except Exception as e:
        print(f"❌ [DEBUG] 模型前向传播测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None

    print("🔍 [DEBUG] 创建训练器...")
    trainer = EquiformerV2Trainer(model, train_loader, val_loader, config)
    print("✅ [DEBUG] 训练器创建完成")
    
    print("🚀 开始训练...")
    print(f"🔍 [DEBUG] 训练配置: epochs={config['num_epochs']}, batch_size={config['batch_size']}, lr={config['lr']}")
    final_metrics = trainer.train(config["num_epochs"])

    if final_metrics is None:
        print(f"❌ 实验失败: num_gaussians = {num_gaussians}")
        return None

    exp_dir = output_root / f"gaussians_{num_gaussians}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "num_gaussians": num_gaussians,
        "mae": float(final_metrics["mae"]),
        "rmse": float(final_metrics["rmse"]),
        "r2": float(final_metrics["r2"]),
        "loss": float(final_metrics["loss"]),
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "config": config,
    }

    with open(exp_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"✅ 实验完成: num_gaussians = {num_gaussians}")
    print(f"   MAE = {final_metrics['mae']:.4f}")
    print(f"   RMSE = {final_metrics['rmse']:.4f}")
    print(f"   R² = {final_metrics['r2']:.4f}")
    print(f"   结果已保存到: {exp_dir}")

    return metrics


def main():
    print("🚀 Eqv2-Lite radial (num_gaussians) 消融实验")
    print("=" * 60)

    # 输出目录 - 使用用户指定的路径
    output_root = Path(r"D:\mylab\equiformer\result\eq_lite_ablation\radial")
    output_root.mkdir(parents=True, exist_ok=True)
    print(f"📁 结果将保存到: {output_root}")

    # 实验配置
    num_gaussians_list = [32, 64, 128, 256]
    config = {
        'lr': 0.0003,
        'weight_decay': 1e-4,
        'num_epochs': 50,
        'batch_size': 6,
        'target_mean': -0.2042156457901001,
        'target_std': 0.6712170839309692,
    }

    # 数据集
    print("\n📂 加载数据集...")
    train_lmdb_path = project_root / "datasets" / "custom_hydrogen" / "train.lmdb"
    val_lmdb_path = project_root / "datasets" / "custom_hydrogen" / "val.lmdb"

    print(f"训练集路径: {train_lmdb_path}")
    print(f"验证集路径: {val_lmdb_path}")
    print(f"路径是否存在: train={train_lmdb_path.exists()}, val={val_lmdb_path.exists()}")

    print("🔍 [DEBUG] 开始创建训练数据集...")
    train_dataset = HydrogenDataset(str(train_lmdb_path))
    print(f"✅ [DEBUG] 训练数据集创建完成，样本数: {len(train_dataset)}")
    
    print("🔍 [DEBUG] 开始创建验证数据集...")
    val_dataset = HydrogenDataset(str(val_lmdb_path))
    print(f"✅ [DEBUG] 验证数据集创建完成，样本数: {len(val_dataset)}")

    print("🔍 [DEBUG] 开始创建训练数据加载器...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False,
        collate_fn=custom_collate_fn,
    )
    print(f"✅ [DEBUG] 训练数据加载器创建完成，batch数: {len(train_loader)}")
    
    print("🔍 [DEBUG] 开始创建验证数据加载器...")
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False,
        collate_fn=custom_collate_fn,
    )
    print(f"✅ [DEBUG] 验证数据加载器创建完成，batch数: {len(val_loader)}")
    
    # 测试数据加载器是否能正常工作
    print("🔍 [DEBUG] 测试数据加载器...")
    try:
        print("  - 尝试获取第一个训练batch...")
        test_train_batch = next(iter(train_loader))
        print(f"  ✅ 成功获取训练batch，batch大小: {len(test_train_batch.y) if hasattr(test_train_batch, 'y') else 'N/A'}")
        
        print("  - 尝试获取第一个验证batch...")
        test_val_batch = next(iter(val_loader))
        print(f"  ✅ 成功获取验证batch，batch大小: {len(test_val_batch.y) if hasattr(test_val_batch, 'y') else 'N/A'}")
    except Exception as e:
        print(f"  ❌ 数据加载器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return

    all_results = []
    print(f"\n🔍 [DEBUG] 准备运行 {len(num_gaussians_list)} 个实验: {num_gaussians_list}")
    
    for idx, ng in enumerate(num_gaussians_list):
        print(f"\n{'='*60}")
        print(f"📊 实验进度: {idx+1}/{len(num_gaussians_list)}")
        print(f"{'='*60}")
        try:
            print(f"🔍 [DEBUG] 开始实验 {idx+1}: num_gaussians={ng}")
            res = run_single_experiment(ng, output_root, config, train_loader, val_loader)
            if res is not None:
                all_results.append(res)
                print(f"✅ [DEBUG] 实验 {idx+1} 完成")
            else:
                print(f"⚠️ [DEBUG] 实验 {idx+1} 返回None")
        except Exception as e:
            print(f"❌ [DEBUG] 实验失败 (num_gaussians={ng}): {e}")
            import traceback
            traceback.print_exc()
            print(f"⚠️ [DEBUG] 继续下一个实验...")
            continue

    if all_results:
        summary_df = pd.DataFrame(all_results)[
            ['num_gaussians', 'mae', 'rmse', 'r2', 'loss', 'num_parameters']
        ]
        summary_df.to_csv(output_root / "summary.csv", index=False, encoding="utf-8-sig")
        print("\n📊 消融实验汇总:")
        print(summary_df.to_string(index=False))
        print(f"\n✅ 汇总结果已保存到: {output_root / 'summary.csv'}")

    train_dataset.close()
    val_dataset.close()

    print("\n🎉 radial (num_gaussians) 消融实验完成！")


if __name__ == "__main__":
    main()

