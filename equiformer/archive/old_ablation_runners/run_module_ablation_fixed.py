#!/usr/bin/env python3
"""
重新运行EquiformerV2模块开关消融实验
使用修复后的批跑脚本，确保实验目录命名正确
增加可视化功能和实时训练监控
"""
import subprocess
import sys
import os
import time
import json
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from pathlib import Path
import threading
import queue

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class TrainingMonitor:
    """训练过程监控器"""
    
    def __init__(self):
        self.process = None
        self.output_queue = queue.Queue()
        self.training_logs = []
        self.is_running = False
        
    def start_monitoring(self, cmd, cwd):
        """开始监控训练过程"""
        self.is_running = True
        self.process = subprocess.Popen(
            cmd, 
            cwd=cwd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # 启动输出读取线程
        self.reader_thread = threading.Thread(target=self._read_output)
        self.reader_thread.daemon = True
        self.reader_thread.start()
        
    def _read_output(self):
        """读取训练输出"""
        for line in iter(self.process.stdout.readline, ''):
            if not self.is_running:
                break
            self.output_queue.put(line.strip())
            
    def get_latest_output(self):
        """获取最新的输出"""
        outputs = []
        while not self.output_queue.empty():
            try:
                outputs.append(self.output_queue.get_nowait())
            except queue.Empty:
                break
        return outputs
    
    def stop_monitoring(self):
        """停止监控"""
        self.is_running = False
        if self.process:
            self.process.terminate()
            self.process.wait()

def visualize_training_progress(config, exp_tag, seed):
    """可视化单个实验的训练进度"""
    print(f"\n🎯 开始可视化训练: {exp_tag}")
    print("=" * 60)
    
    # 构建训练命令
    cmd = [
        sys.executable, "src/train_enhanced_equiformer_v2.py",
        "--num_layers", str(config['num_layers']),
        "--sphere_channels", str(config['sphere_channels']),
        "--num_heads", str(config['num_heads']),
        "--grid_resolution", str(config['grid_resolution']),
        "--edge_channels", str(config['edge_channels']),
        "--eSCN", str(config['eSCN']),
        "--attn_renorm", str(config['attn_renorm']),
        "--sep_s2", str(config['sep_s2']),
        "--sep_ln", str(config['sep_ln']),
        "--seed", str(seed),
        "--experiment_tag", exp_tag,
        "--batch_size", "16"
    ]
    
    print(f"🚀 运行命令: {' '.join(cmd)}")
    
    # 创建监控器
    monitor = TrainingMonitor()
    
    try:
        print(f"\n⏳ 开始训练...")
        monitor.start_monitoring(cmd, ".")
        
        # 实时显示训练输出
        start_time = time.time()
        epoch_count = 0
        batch_count = 0
        training_losses = []
        validation_losses = []
        validation_maes = []
        
        while monitor.is_running and monitor.process.poll() is None:
            outputs = monitor.get_latest_output()
            
            for output in outputs:
                if output:
                    # 检测epoch进度
                    if "Epoch" in output and "/" in output and "completed" in output:
                        epoch_count += 1
                        print(f"✅ Epoch {epoch_count} 完成")
                    
                    # 检测batch进度
                    elif "batch=" in output and "loss=" in output:
                        batch_count += 1
                        if batch_count % 100 == 0:  # 每100个batch显示一次
                            print(f"   📊 Batch {batch_count}: {output}")
                    
                    # 检测验证结果
                    elif "验证MAE:" in output:
                        print(f"   📈 {output}")
                        # 提取MAE值
                        try:
                            mae_value = float(output.split("验证MAE:")[1].split("eV")[0].strip())
                            validation_maes.append(mae_value)
                        except:
                            pass
                    
                    # 检测训练损失
                    elif "训练损失:" in output:
                        try:
                            train_loss = float(output.split("训练损失:")[1].strip())
                            training_losses.append(train_loss)
                        except:
                            pass
                    
                    # 检测验证损失
                    elif "验证损失:" in output:
                        try:
                            val_loss = float(output.split("验证损失:")[1].strip())
                            validation_losses.append(val_loss)
                        except:
                            pass
                    
                    # 检测警告
                    elif "Warning:" in output:
                        print(f"   ⚠️  {output}")
                    
                    # 检测错误
                    elif "Error" in output or "error" in output:
                        print(f"   ❌ {output}")
            
            time.sleep(0.1)  # 短暂休眠避免过度占用CPU
        
        # 等待进程完成
        if monitor.process:
            return_code = monitor.process.wait()
            
            if return_code == 0:
                print(f"\n🎉 训练成功完成!")
                elapsed_time = time.time() - start_time
                print(f"⏱️  总耗时: {elapsed_time:.2f} 秒")
                
                # 创建训练过程可视化
                create_training_visualization(config, exp_tag, seed, training_losses, validation_losses, validation_maes)
                
                return True
            else:
                print(f"\n❌ 训练失败! 返回码: {return_code}")
                return False
                
    except KeyboardInterrupt:
        print(f"\n⏹️  用户中断训练")
        return False
    except Exception as e:
        print(f"\n💥 训练异常: {e}")
        return False
    finally:
        monitor.stop_monitoring()

def create_training_visualization(config, exp_tag, seed, train_losses, val_losses, val_maes):
    """创建训练过程可视化"""
    try:
        # 构建实验目录路径
        kv_parts = [
            f"num_layers={config['num_layers']}",
            f"sphere_channels={config['sphere_channels']}",
            f"num_heads={config['num_heads']}",
            f"grid_resolution={config['grid_resolution']}",
            f"edge_channels={config['edge_channels']}",
        ]
        kv_parts.append(exp_tag)
        ablation_name = "__".join(kv_parts)
        
        experiment_date = "2025-09-10"
        experiment_dir = Path("experiments") / f"{experiment_date}_run1" / "ablation" / ablation_name / f"seed={seed}"
        
        # 创建可视化图表
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 训练和验证损失
        if train_losses and val_losses:
            epochs = range(1, len(train_losses) + 1)
            ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
            ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title(f'Training Progress - {exp_tag}')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 2. 验证MAE
        if val_maes:
            epochs = range(1, len(val_maes) + 1)
            ax2.plot(epochs, val_maes, 'g-', label='Validation MAE', linewidth=2)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('MAE (eV)')
            ax2.set_title('Validation MAE Progress')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. 配置信息
        config_text = f"""Configuration:
• Layers: {config['num_layers']}
• Sphere Channels: {config['sphere_channels']}
• Heads: {config['num_heads']}
• Grid Resolution: {config['grid_resolution']}
• Edge Channels: {config['edge_channels']}
• eSCN: {config['eSCN']}
• Attn Renorm: {config['attn_renorm']}
• Sep S2: {config['sep_s2']}
• Sep LN: {config['sep_ln']}
• Seed: {seed}"""
        
        ax3.text(0.1, 0.9, config_text, transform=ax3.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace')
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
        ax3.axis('off')
        ax3.set_title('Experiment Configuration')
        
        # 4. 性能总结
        if val_maes:
            final_mae = val_maes[-1]
            best_mae = min(val_maes)
            improvement = ((val_maes[0] - final_mae) / val_maes[0]) * 100 if val_maes[0] > 0 else 0
            
            summary_text = f"""Performance Summary:
• Final MAE: {final_mae:.6f} eV
• Best MAE: {best_mae:.6f} eV
• Improvement: {improvement:.2f}%
• Total Epochs: {len(val_maes)}"""
            
            ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=10,
                    verticalalignment='top', fontfamily='monospace')
            ax4.set_xlim(0, 1)
            ax4.set_ylim(0, 1)
            ax4.axis('off')
            ax4.set_title('Performance Summary')
        
        plt.tight_layout()
        
        # 保存图片到实验目录
        os.makedirs(experiment_dir, exist_ok=True)
        output_path = experiment_dir / "training_visualization.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"📊 训练可视化已保存: {output_path}")
        
        plt.close()
        
    except Exception as e:
        print(f"⚠️  创建训练可视化失败: {e}")

def create_experiment_summary():
    """创建实验总结可视化"""
    try:
        experiment_date = "2025-09-10"
        ablation_dir = Path("experiments") / f"{experiment_date}_run1" / "ablation"
        
        if not ablation_dir.exists():
            print("❌ ablation目录不存在")
            return
        
        # 统计实验信息
        subdirs = [d for d in ablation_dir.iterdir() if d.is_dir()]
        module_dirs = [d for d in subdirs if "MODULE" in d.name]
        
        print(f"\n📊 实验总结:")
        print(f"   总实验数: {len(subdirs)}")
        print(f"   模块消融实验: {len(module_dirs)}")
        
        # 创建总结图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 实验类型分布
        lmax_count = len([d for d in subdirs if "grid_resolution" in d.name and "MODULE" not in d.name])
        layer_count = len([d for d in subdirs if "num_layers" in d.name and "grid_resolution" not in d.name])
        
        labels = ['Module Ablation', 'Lmax Ablation', 'Layer Ablation']
        sizes = [len(module_dirs), lmax_count, layer_count]
        colors = ['#ff9999', '#66b3ff', '#99ff99']
        
        ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Experiment Type Distribution')
        
        # 实验时间线
        if subdirs:
            times = [d.stat().st_mtime for d in subdirs]
            times.sort()
            
            ax2.plot(range(len(times)), times, 'b-', marker='o', markersize=4)
            ax2.set_title('Experiment Timeline')
            ax2.set_xlabel('Experiment Index')
            ax2.set_ylabel('Timestamp')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存总结图片
        summary_path = ablation_dir / "experiment_summary.png"
        plt.savefig(summary_path, dpi=150, bbox_inches='tight')
        print(f"📊 实验总结已保存: {summary_path}")
        
        plt.close()
        
    except Exception as e:
        print(f"⚠️  创建实验总结失败: {e}")

def run_module_ablation_fixed():
    """运行修复后的模块消融实验（带可视化）"""
    print("=== 重新运行模块消融实验（修复版 + 可视化） ===")
    
    # 检查环境
    try:
        import pandas as pd
        print("✓ pandas可用")
    except ImportError:
        print("❌ pandas不可用，请激活正确的conda环境")
        return False
    
    # 检查GPU
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ GPU可用: {torch.cuda.get_device_name()}")
        else:
            print("❌ GPU不可用")
            return False
    except Exception as e:
        print(f"❌ GPU检查失败: {e}")
        return False
    
    # 模块消融实验配置
    module_configs = [
        {
            'num_layers': 2, 'sphere_channels': 64, 'num_heads': 4, 'grid_resolution': 16,
            'edge_channels': 64, 'eSCN': 1, 'attn_renorm': 1, 'sep_s2': 1, 'sep_ln': 1,
            'name': 'All Modules ON'
        },
        {
            'num_layers': 2, 'sphere_channels': 64, 'num_heads': 4, 'grid_resolution': 16,
            'edge_channels': 64, 'eSCN': 1, 'attn_renorm': 0, 'sep_s2': 1, 'sep_ln': 1,
            'name': 'No Attn Renorm'
        },
        {
            'num_layers': 2, 'sphere_channels': 64, 'num_heads': 4, 'grid_resolution': 16,
            'edge_channels': 64, 'eSCN': 1, 'attn_renorm': 1, 'sep_s2': 0, 'sep_ln': 1,
            'name': 'No Sep S2'
        },
        {
            'num_layers': 2, 'sphere_channels': 64, 'num_heads': 4, 'grid_resolution': 16,
            'edge_channels': 64, 'eSCN': 1, 'attn_renorm': 1, 'sep_s2': 1, 'sep_ln': 0,
            'name': 'No Sep LN'
        },
        {
            'num_layers': 2, 'sphere_channels': 64, 'num_heads': 4, 'grid_resolution': 16,
            'edge_channels': 64, 'eSCN': 1, 'attn_renorm': 0, 'sep_s2': 0, 'sep_ln': 0,
            'name': 'Only eSCN'
        }
    ]
    
    print(f"\n📋 实验配置:")
    print(f"   - 实验数量: {len(module_configs)}")
    print(f"   - 每个实验重复种子: 1 (仅seed=0)")
    print(f"   - 实验标签: MODULE_VIS")
    print(f"   - 预计总时间: {len(module_configs)*1*1.5:.1f}小时")
    
    print("\n配置详情:")
    for i, config in enumerate(module_configs):
        print(f"  {i+1}. {config['name']}: AR={config['attn_renorm']}, S2={config['sep_s2']}, LN={config['sep_ln']}")
    
    # 运行每个配置的实验
    successful_experiments = 0
    total_experiments = len(module_configs) * 1  # 1个种子
    
    for i, config in enumerate(module_configs):
        print(f"\n{'='*80}")
        print(f"🧪 实验 {i+1}/{len(module_configs)}: {config['name']}")
        print(f"{'='*80}")
        
        # 只运行seed=0
        seed = 0
        exp_tag = f"MODULE_VIS_{config['name'].replace(' ', '_')}"
        
        print(f"\n🌱 种子 {seed}:")
        success = visualize_training_progress(config, exp_tag, seed)
        
        if success:
            successful_experiments += 1
            print(f"✅ 实验成功完成")
        else:
            print(f"❌ 实验失败")
        
        # 短暂休息避免GPU过热
        time.sleep(2)
    
    # 创建实验总结
    print(f"\n{'='*80}")
    print(f"📊 实验总结")
    print(f"{'='*80}")
    print(f"✅ 成功实验: {successful_experiments}/{total_experiments}")
    print(f"❌ 失败实验: {total_experiments - successful_experiments}/{total_experiments}")
    
    # 创建总结可视化
    create_experiment_summary()
    
    return successful_experiments == total_experiments

def run_analysis():
    """运行结果分析"""
    print("\n=== 运行结果分析 ===")
    
    results_file = "experiments/ablation/results_modules_fixed.csv"
    if not os.path.exists(results_file):
        print(f"❌ 结果文件不存在: {results_file}")
        return False
    
    cmd = [
        "python", "experiments/ablation/analysis/module_report.py",
        "--input", results_file
    ]
    
    print(f"运行分析命令: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd)
        if result.returncode == 0:
            print("✅ 结果分析完成！")
            return True
        else:
            print("❌ 分析运行出现问题")
            return False
    except Exception as e:
        print(f"❌ 分析运行异常: {e}")
        return False

if __name__ == "__main__":
    print("EquiformerV2模块开关消融实验（修复版 + 可视化）")
    print("=" * 60)
    
    # 确认运行
    print("是否开始运行带可视化的模块消融实验？")
    print("输入 'y' 开始运行，其他键退出")
    choice = input("选择: ").strip().lower()
    
    if choice != 'y':
        print("退出实验")
        sys.exit(0)
    
    # 运行实验
    if run_module_ablation_fixed():
        print("\n🎉 模块消融实验完成！")
        print("📊 训练可视化: experiments/2025-09-10_run1/ablation/*/training_visualization.png")
        print("📈 实验总结: experiments/2025-09-10_run1/ablation/experiment_summary.png")
        print("📁 训练日志: experiments/2025-09-10_run1/ablation/")
    else:
        print("\n❌ 实验失败或中断")
