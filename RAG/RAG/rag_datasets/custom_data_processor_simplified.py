#!/usr/bin/env python3
"""
简化版数据预处理器：避免torch_geometric依赖问题
将VASP文件和Excel特征数据转换为可用格式
"""

import os
import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
from ase.io import read as ase_read
from ase import Atoms
import lmdb


class SimpleData:
    """简单的数据容器，替代torch_geometric.data.Data"""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def keys(self):
        return [k for k in self.__dict__.keys() if not k.startswith('_')]


def compute_distance_graph(pos, max_radius=12.0, max_neighbors=50):
    """手动计算距离图"""
    n_atoms = pos.shape[0]
    edge_index_list = []
    
    for i in range(n_atoms):
        distances = torch.norm(pos - pos[i], dim=1)
        neighbors = torch.where((distances < max_radius) & (distances > 0))[0]
        
        # 限制邻居数量
        if len(neighbors) > max_neighbors:
            _, indices = torch.topk(distances[neighbors], max_neighbors, largest=False)
            neighbors = neighbors[indices]
        
        for j in neighbors:
            edge_index_list.append([i, j.item()])
    
    if edge_index_list:
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    
    return edge_index


class VASPDataProcessor:
    """处理VASP文件和特征数据的类"""
    
    def __init__(self, 
                 vasp_dir: str, 
                 excel_path: str, 
                 output_dir: str,
                 max_radius: float = 12.0,
                 max_neighbors: int = 50):
        """
        Args:
            vasp_dir: VASP文件目录路径
            excel_path: Excel特征文件路径
            output_dir: 输出目录
            max_radius: 原子间最大距离
            max_neighbors: 最大邻居数
        """
        self.vasp_dir = Path(vasp_dir)
        self.excel_path = Path(excel_path)
        self.output_dir = Path(output_dir)
        self.max_radius = max_radius
        self.max_neighbors = max_neighbors
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 读取Excel数据
        print(f"Reading Excel file: {excel_path}")
        self.df = pd.read_excel(excel_path)
        print(f"Loaded Excel data with shape: {self.df.shape}")
        
        # 特征列（排除非特征列）
        self.feature_cols = [col for col in self.df.columns 
                           if col not in ['structures', 'Equation', 'reactants', 'products', 'Unnamed: 0', 'Structure']]
        self.target_col = 'ΔGH'  # 目标列
        
        print(f"Feature columns ({len(self.feature_cols)}): {self.feature_cols}")
        print(f"Target column: {self.target_col}")

    def parse_vasp_file(self, vasp_path: str) -> Optional[Atoms]:
        """解析单个VASP文件"""
        try:
            atoms = ase_read(vasp_path, format='vasp')
            return atoms
        except Exception as e:
            print(f"Error reading {vasp_path}: {e}")
            return None

    def atoms_to_data(self, atoms: Atoms, features: np.ndarray, target: float, structure_id: int) -> SimpleData:
        """将ASE Atoms对象转换为SimpleData对象"""
        # 原子坐标
        pos = torch.tensor(atoms.positions, dtype=torch.float32)
        
        # 原子类型（原子序数）
        atomic_numbers = torch.tensor(atoms.numbers, dtype=torch.long)
        
        # 计算边和距离
        edge_index = compute_distance_graph(pos, self.max_radius, self.max_neighbors)
        
        # 计算边的距离
        if edge_index.shape[1] > 0:
            edge_distance = torch.norm(pos[edge_index[0]] - pos[edge_index[1]], dim=1)
        else:
            edge_distance = torch.zeros(0, dtype=torch.float32)
        
        # 创建Data对象
        data = SimpleData(
            pos=pos,
            atomic_numbers=atomic_numbers,
            edge_index=edge_index,
            edge_distance=edge_distance,
            natoms=torch.tensor([len(atoms)], dtype=torch.long),
            y=torch.tensor([target], dtype=torch.float32),
            features=torch.tensor(features, dtype=torch.float32),
            sid=torch.tensor([structure_id], dtype=torch.long),
            tags=torch.ones(len(atoms), dtype=torch.long),  # 默认所有原子为自由原子
        )
        
        return data

    def process_all_data(self) -> List[SimpleData]:
        """处理所有数据"""
        processed_data = []
        
        print("Processing VASP files and features...")
        
        for idx, row in self.df.iterrows():
            structure_id = row['structures']
            vasp_file = self.vasp_dir / f"{structure_id}.vasp"
            
            if not vasp_file.exists():
                if idx < 10:  # 只对前10个打印警告，避免输出过多
                    print(f"Warning: VASP file {vasp_file} not found, skipping...")
                continue
            
            # 解析VASP文件
            atoms = self.parse_vasp_file(str(vasp_file))
            if atoms is None:
                continue
            
            # 提取特征和目标
            features = row[self.feature_cols].values.astype(np.float32)
            target = row[self.target_col]
            
            # 检查数据有效性
            if np.isnan(target):
                print(f"Warning: NaN target for structure {structure_id}, skipping...")
                continue
                
            if np.any(np.isnan(features)):
                print(f"Warning: NaN features for structure {structure_id}, skipping...")
                continue
            
            # 转换为Data
            data = self.atoms_to_data(atoms, features, target, structure_id)
            processed_data.append(data)
            
            if (idx + 1) % 500 == 0:
                print(f"Processed {idx + 1} structures, successful: {len(processed_data)}")
        
        print(f"Successfully processed {len(processed_data)} out of {len(self.df)} structures")
        return processed_data

    def create_train_val_test_split(self, data_list: List[SimpleData], 
                                  train_ratio: float = 0.8, 
                                  val_ratio: float = 0.1) -> Tuple[List[SimpleData], List[SimpleData], List[SimpleData]]:
        """划分训练、验证、测试集"""
        np.random.seed(42)  # 固定随机种子
        indices = np.random.permutation(len(data_list))
        
        n_train = int(len(data_list) * train_ratio)
        n_val = int(len(data_list) * val_ratio)
        
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]
        
        train_data = [data_list[i] for i in train_indices]
        val_data = [data_list[i] for i in val_indices]
        test_data = [data_list[i] for i in test_indices]
        
        print(f"Data split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
        
        return train_data, val_data, test_data

    def save_to_lmdb(self, data_list: List[SimpleData], lmdb_path: str):
        """保存数据到LMDB格式"""
        lmdb_path = Path(lmdb_path)
        lmdb_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 创建LMDB环境
        env = lmdb.open(str(lmdb_path), map_size=1099511627776)  # 1TB
        
        with env.begin(write=True) as txn:
            # 保存数据长度
            txn.put("length".encode("ascii"), pickle.dumps(len(data_list)))
            
            # 保存每个数据点
            for idx, data in enumerate(data_list):
                txn.put(f"{idx}".encode("ascii"), pickle.dumps(data))
                
                if (idx + 1) % 500 == 0:
                    print(f"Saved {idx + 1}/{len(data_list)} data points to LMDB")
        
        env.close()
        print(f"Saved {len(data_list)} data points to {lmdb_path}")

    def compute_normalization_stats(self, train_data: List[SimpleData]) -> Dict:
        """计算归一化统计信息"""
        targets = torch.cat([data.y for data in train_data])
        
        stats = {
            "target_mean": float(targets.mean()),
            "target_std": float(targets.std()),
            "num_samples": len(train_data),
            "target_min": float(targets.min()),
            "target_max": float(targets.max())
        }
        
        print(f"Normalization stats:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        return stats

    def save_data_summary(self, train_data: List[SimpleData], val_data: List[SimpleData], test_data: List[SimpleData]):
        """保存数据摘要信息"""
        summary = {
            "total_structures": len(train_data) + len(val_data) + len(test_data),
            "train_samples": len(train_data),
            "val_samples": len(val_data),
            "test_samples": len(test_data),
            "feature_columns": self.feature_cols,
            "target_column": self.target_col,
            "max_radius": self.max_radius,
            "max_neighbors": self.max_neighbors
        }
        
        # 计算一些统计信息
        all_data = train_data + val_data + test_data
        atom_counts = [int(data.natoms.item()) for data in all_data]
        edge_counts = [data.edge_index.shape[1] for data in all_data]
        
        summary.update({
            "atom_count_stats": {
                "min": int(np.min(atom_counts)),
                "max": int(np.max(atom_counts)),
                "mean": float(np.mean(atom_counts)),
                "std": float(np.std(atom_counts))
            },
            "edge_count_stats": {
                "min": int(np.min(edge_counts)),
                "max": int(np.max(edge_counts)),
                "mean": float(np.mean(edge_counts)),
                "std": float(np.std(edge_counts))
            }
        })
        
        # 保存摘要
        summary_file = self.output_dir / "data_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Data summary saved to {summary_file}")

    def run_full_pipeline(self):
        """运行完整的数据处理流程"""
        print("=" * 60)
        print("Starting full data processing pipeline...")
        print("=" * 60)
        
        # 1. 处理所有数据
        print("\n1. Processing all data...")
        all_data = self.process_all_data()
        
        if len(all_data) == 0:
            raise ValueError("No data was processed successfully!")
        
        # 2. 划分数据集
        print("\n2. Splitting data...")
        train_data, val_data, test_data = self.create_train_val_test_split(all_data)
        
        # 3. 计算归一化统计信息
        print("\n3. Computing normalization stats...")
        norm_stats = self.compute_normalization_stats(train_data)
        
        # 4. 保存到LMDB
        print("\n4. Saving to LMDB format...")
        self.save_to_lmdb(train_data, self.output_dir / "train.lmdb")
        self.save_to_lmdb(val_data, self.output_dir / "val.lmdb")
        self.save_to_lmdb(test_data, self.output_dir / "test.lmdb")
        
        # 5. 保存归一化统计信息
        print("\n5. Saving statistics...")
        stats_file = self.output_dir / "normalization_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(norm_stats, f, indent=2)
        
        # 6. 保存数据摘要
        self.save_data_summary(train_data, val_data, test_data)
        
        print("\n" + "=" * 60)
        print("Data processing pipeline completed successfully!")
        print("=" * 60)
        print(f"📁 Output directory: {self.output_dir}")
        print(f"📊 Train samples: {len(train_data)}")
        print(f"📊 Val samples: {len(val_data)}")
        print(f"📊 Test samples: {len(test_data)}")
        print(f"🎯 Target mean: {norm_stats['target_mean']:.4f}")
        print(f"🎯 Target std: {norm_stats['target_std']:.4f}")
        print("=" * 60)


if __name__ == "__main__":
    # 使用示例
    print("🚀 Starting VASP data processing...")
    
    processor = VASPDataProcessor(
        vasp_dir="data/raw/the_atomic_structure_for_ML_model",
        excel_path="data/raw/25features_for_ML.xlsx",
        output_dir="datasets/custom_hydrogen"
    )
    
    try:
        processor.run_full_pipeline()
        print("✅ Processing completed successfully!")
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        raise 