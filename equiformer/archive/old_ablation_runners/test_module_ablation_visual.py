#!/usr/bin/env python3
"""
可视化模块消融实验测试脚本
实时显示训练过程和结果分析
"""

import os
import sys
import json
import subprocess
import time
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from pathlib import Path
import threading
import queue

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

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

def visualize_training_progress():
    """可视化训练进度"""
    print("🎯 开始可视化模块消融实验测试")
    print("=" * 80)
    
    # 测试配置
    test_config = {
        'num_layers': 2,
        'sphere_channels': 64,
        'num_heads': 4,
        'grid_resolution': 16,
        'edge_channels': 64,
        'eSCN': 1,
        'attn_renorm': 1,
        'sep_s2': 1,
        'sep_ln': 1
    }
    
    exp_tag = "TEST_MODULE_VIS"
    seed = 0
    
    print(f"📋 测试配置:")
    for key, value in test_config.items():
        print(f"   {key}: {value}")
    print(f"   实验标签: {exp_tag}")
    print(f"   种子: {seed}")
    
    # 构建训练命令
    cmd = [
        sys.executable, str(project_root / "src" / "train_enhanced_equiformer_v2.py"),
        "--num_layers", str(test_config['num_layers']),
        "--sphere_channels", str(test_config['sphere_channels']),
        "--num_heads", str(test_config['num_heads']),
        "--grid_resolution", str(test_config['grid_resolution']),
        "--edge_channels", str(test_config['edge_channels']),
        "--eSCN", str(test_config['eSCN']),
        "--attn_renorm", str(test_config['attn_renorm']),
        "--sep_s2", str(test_config['sep_s2']),
        "--sep_ln", str(test_config['sep_ln']),
        "--seed", str(seed),
        "--experiment_tag", exp_tag,
        "--num_epochs", "3",  # 运行3个epoch进行测试
        "--batch_size", "16"
    ]
    
    print(f"\n🚀 运行命令:")
    print(f"   {' '.join(cmd)}")
    
    # 创建监控器
    monitor = TrainingMonitor()
    
    try:
        print(f"\n⏳ 开始训练...")
        monitor.start_monitoring(cmd, str(project_root))
        
        # 实时显示训练输出
        start_time = time.time()
        epoch_count = 0
        batch_count = 0
        
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
                        if batch_count % 50 == 0:  # 每50个batch显示一次
                            print(f"   📊 Batch {batch_count}: {output}")
                    
                    # 检测验证结果
                    elif "验证MAE:" in output:
                        print(f"   📈 {output}")
                    
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
                
                # 检查实验结果
                check_experiment_results(test_config, exp_tag, seed)
                
            else:
                print(f"\n❌ 训练失败! 返回码: {return_code}")
                
    except KeyboardInterrupt:
        print(f"\n⏹️  用户中断训练")
    except Exception as e:
        print(f"\n💥 训练异常: {e}")
    finally:
        monitor.stop_monitoring()

def check_experiment_results(config, exp_tag, seed):
    """检查实验结果"""
    print(f"\n" + "=" * 60)
    print(f"🔍 检查实验结果")
    print(f"=" * 60)
    
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
    experiment_dir = project_root / "experiments" / f"{experiment_date}_run1" / "ablation" / ablation_name / f"seed={seed}"
    logs_dir = experiment_dir / "logs"
    
    print(f"📁 实验目录: {experiment_dir}")
    print(f"📁 日志目录: {logs_dir}")
    
    # 检查目录是否存在
    if not experiment_dir.exists():
        print(f"❌ 实验目录不存在!")
        return
    
    print(f"✅ 实验目录存在")
    
    # 检查关键文件
    files_to_check = {
        "enhanced_equiformer_v2_test_results.json": "测试结果",
        "enhanced_equiformer_v2_training_history.json": "训练历史",
        "best_enhanced_equiformer_v2.pt": "最佳模型",
        "enhanced_equiformer_v2_predictions.csv": "预测结果"
    }
    
    print(f"\n📄 检查结果文件:")
    for filename, description in files_to_check.items():
        file_path = logs_dir / filename if filename.endswith('.json') else experiment_dir / filename
        exists = file_path.exists()
        status = "✅" if exists else "❌"
        print(f"   {status} {description}: {filename}")
        
        if exists and filename.endswith('.json'):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                if filename == "enhanced_equiformer_v2_test_results.json":
                    print(f"      📊 测试MAE: {data.get('test_mae', 'N/A'):.6f}")
                    print(f"      📊 测试RMSE: {data.get('test_rmse', 'N/A'):.6f}")
                elif filename == "enhanced_equiformer_v2_training_history.json":
                    epochs = len(data.get('train_loss', []))
                    print(f"      📈 训练轮数: {epochs}")
            except Exception as e:
                print(f"      ⚠️  读取失败: {e}")

def visualize_directory_structure():
    """可视化目录结构"""
    print(f"\n" + "=" * 60)
    print(f"📂 当前实验目录结构")
    print(f"=" * 60)
    
    experiment_date = "2025-09-10"
    ablation_dir = project_root / "experiments" / f"{experiment_date}_run1" / "ablation"
    
    if not ablation_dir.exists():
        print(f"❌ ablation目录不存在: {ablation_dir}")
        return
    
    print(f"📁 ablation目录: {ablation_dir}")
    
    # 统计目录信息
    subdirs = [d for d in ablation_dir.iterdir() if d.is_dir()]
    print(f"📊 子目录数量: {len(subdirs)}")
    
    # 按类型分组
    module_dirs = [d for d in subdirs if "TEST_MODULE" in d.name]
    lmax_dirs = [d for d in subdirs if "grid_resolution" in d.name and "TEST_MODULE" not in d.name]
    layer_dirs = [d for d in subdirs if "num_layers" in d.name and "grid_resolution" not in d.name]
    
    print(f"\n📋 目录分类:")
    print(f"   🧪 模块消融实验: {len(module_dirs)}")
    print(f"   📏 Lmax消融实验: {len(lmax_dirs)}")
    print(f"   🔢 层数消融实验: {len(layer_dirs)}")
    
    # 显示最近的几个目录
    print(f"\n📝 最近的实验目录:")
    recent_dirs = sorted(subdirs, key=lambda x: x.stat().st_mtime, reverse=True)[:5]
    for i, dir_path in enumerate(recent_dirs, 1):
        mtime = time.ctime(dir_path.stat().st_mtime)
        print(f"   {i}. {dir_path.name}")
        print(f"      修改时间: {mtime}")

def create_summary_visualization():
    """创建总结可视化"""
    print(f"\n" + "=" * 60)
    print(f"📊 创建总结可视化")
    print(f"=" * 60)
    
    try:
        # 创建简单的状态图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 实验状态饼图
        experiment_date = "2025-09-10"
        ablation_dir = project_root / "experiments" / f"{experiment_date}_run1" / "ablation"
        
        if ablation_dir.exists():
            subdirs = [d for d in ablation_dir.iterdir() if d.is_dir()]
            
            # 统计不同类型的实验
            module_count = len([d for d in subdirs if "TEST_MODULE" in d.name])
            lmax_count = len([d for d in subdirs if "grid_resolution" in d.name and "TEST_MODULE" not in d.name])
            layer_count = len([d for d in subdirs if "num_layers" in d.name and "grid_resolution" not in d.name])
            
            labels = ['模块消融', 'Lmax消融', '层数消融']
            sizes = [module_count, lmax_count, layer_count]
            colors = ['#ff9999', '#66b3ff', '#99ff99']
            
            ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax1.set_title('实验类型分布')
            
            # 实验时间线
            if subdirs:
                times = [d.stat().st_mtime for d in subdirs]
                times.sort()
                
                ax2.plot(range(len(times)), times, 'b-', marker='o', markersize=4)
                ax2.set_title('实验时间线')
                ax2.set_xlabel('实验序号')
                ax2.set_ylabel('时间戳')
                ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        output_path = project_root / "experiments" / "ablation" / "test_visualization.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"📊 可视化图片已保存: {output_path}")
        
        plt.show()
        
    except Exception as e:
        print(f"⚠️  创建可视化失败: {e}")

def main():
    """主函数"""
    print("🎨 可视化模块消融实验测试脚本")
    print("=" * 80)
    
    try:
        # 1. 可视化训练过程
        visualize_training_progress()
        
        # 2. 可视化目录结构
        visualize_directory_structure()
        
        # 3. 创建总结可视化
        create_summary_visualization()
        
        print(f"\n🎉 测试完成!")
        
    except KeyboardInterrupt:
        print(f"\n⏹️  用户中断测试")
    except Exception as e:
        print(f"\n💥 测试异常: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
