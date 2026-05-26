#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版EquiformerV2实现，直接使用OC20的完整模块
不简化任何功能，包括完整的高斯距离扩展、SO3操作等
"""

import os
import sys
import json
import pickle
import lmdb
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
import warnings
import math
warnings.filterwarnings('ignore')

# 添加项目路径以便导入EquiformerV2组件
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

# 直接导入OC20的完整EquiformerV2实现
from equiformer_v2.nets.equiformer_v2.equiformer_v2_oc20 import EquiformerV2_OC20

# 数据类
class SimpleData:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def keys(self):
        return [k for k in self.__dict__.keys() if not k.startswith('_')]

# 自定义 Unpickler 来处理 SimpleData 类的反序列化
class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # 如果找不到 SimpleData，尝试从当前模块导入
        if name == 'SimpleData':
            try:
                return super().find_class(module, name)
            except (AttributeError, ModuleNotFoundError):
                return SimpleData
        return super().find_class(module, name)

class HydrogenDataset(Dataset):
    """氢吸附数据集"""
    
    def __init__(self, lmdb_path, transform=None):
        self.lmdb_path = Path(lmdb_path)
        self.transform = transform
        
        # 连接LMDB数据库
        self.env = lmdb.open(str(self.lmdb_path), readonly=True, lock=False)
        
        # 获取数据长度
        with self.env.begin() as txn:
            self.length = pickle.loads(txn.get("length".encode("ascii")))
        
        print("Loaded dataset from {} with {} samples".format(lmdb_path, self.length))
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        with self.env.begin() as txn:
            data_bytes = txn.get("{}".format(idx).encode("ascii"))
            if data_bytes is None:
                raise KeyError(f"Index {idx} not found in database")
            
            # 使用自定义 Unpickler 来处理 SimpleData 类
            import io
            try:
                # 先尝试普通 pickle
                data = pickle.loads(data_bytes)
            except (AttributeError, ModuleNotFoundError):
                # 如果失败，使用自定义 Unpickler
                unpickler = CustomUnpickler(io.BytesIO(data_bytes))
                data = unpickler.load()
        
        if self.transform:
            data = self.transform(data)
        
        return data
    
    def close(self):
        self.env.close()

# 增强版EquiformerV2模型 - 直接使用OC20的完整实现
class EnhancedEquiformerV2(EquiformerV2_OC20):
    """
    增强版EquiformerV2实现
    直接继承OC20的完整EquiformerV2_OC20，不简化任何功能
    包括完整的高斯距离扩展、SO3操作、S²激活等
    """
    
    def __init__(
        self,
        max_radius=12.0,
        max_neighbors=20,
        max_num_elements=90,
        num_layers=6,
        sphere_channels=128,
        attn_hidden_channels=64,
        num_heads=8,
        attn_alpha_channels=64,
        attn_value_channels=16,
        ffn_hidden_channels=128,
        lmax_list=[6],
        mmax_list=[2],
        grid_resolution=18,
        edge_channels=128,
        use_atom_edge_embedding=True,
        share_atom_edge_embedding=False,
        alpha_drop=0.1,
        drop_path_rate=0.05,
        proj_drop=0.0,
        norm_type='layer_norm_sh'
    ):
        # 直接调用父类构造函数，使用OC20的完整配置
        super().__init__(
            num_atoms=100,  # 最大原子数，用于初始化
            bond_feat_dim=0,  # 不使用键特征
            num_targets=1,  # 只预测一个目标（能量）
            use_pbc=False,  # 氢吸附数据不需要周期性边界条件
            regress_forces=False,  # 只预测能量，不预测力
            otf_graph=True,  # 动态计算图
            max_neighbors=max_neighbors,
            max_radius=max_radius,
            max_num_elements=max_num_elements,
            num_layers=num_layers,
            sphere_channels=sphere_channels,
            attn_hidden_channels=attn_hidden_channels,
            num_heads=num_heads,
            attn_alpha_channels=attn_alpha_channels,
            attn_value_channels=attn_value_channels,
            ffn_hidden_channels=ffn_hidden_channels,
            norm_type=norm_type,
            lmax_list=lmax_list,
            mmax_list=mmax_list,
            grid_resolution=grid_resolution,
            num_sphere_samples=128,
            edge_channels=edge_channels,
            use_atom_edge_embedding=use_atom_edge_embedding,
            share_atom_edge_embedding=share_atom_edge_embedding,
            distance_function='gaussian',  # 使用完整的高斯距离函数
            num_distance_basis=512,  # 使用完整的高斯基函数数量
            attn_activation='silu',
            use_s2_act_attn=False,
            use_attn_renorm=True,
            ffn_activation='silu',
            use_gate_act=False,
            use_grid_mlp=True,  # 使用完整的网格MLP
            use_sep_s2_act=True,  # 使用完整的可分离S²激活
            alpha_drop=alpha_drop,
            drop_path_rate=drop_path_rate,
            proj_drop=proj_drop,
            weight_init='uniform'
        )
        
        # 保存配置参数
        self.max_radius = max_radius
        self.max_neighbors = max_neighbors
        self.max_num_elements = max_num_elements
        self.num_layers = num_layers
        self.sphere_channels = sphere_channels
        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.grid_resolution = grid_resolution
        self.edge_channels = edge_channels
        self.use_atom_edge_embedding = use_atom_edge_embedding
        self.share_atom_edge_embedding = share_atom_edge_embedding
    
    def generate_graph(self, data):
        """生成图结构 - 使用OC20的完整实现"""
        # 直接使用父类的图生成方法
        return super().generate_graph(data)
    
    def forward(self, data):
        """前向传播 - 使用OC20的完整实现"""
        # 直接使用父类的前向传播方法，包含所有完整功能
        return super().forward(data)

def create_batch_from_data_list(data_list):
    """从数据列表创建批次"""
    batch_data = SimpleData()
    
    pos_list = []
    atomic_numbers_list = []
    y_list = []
    natoms_list = []
    batch_indices = []
    edge_index_list = []
    edge_distance_list = []
    
    total_atoms = 0
    
    for i, data in enumerate(data_list):
        pos_list.append(data.pos)
        atomic_numbers_list.append(data.atomic_numbers)
        y_list.append(data.y)
        natoms_list.append(data.natoms)
        
        # 为每个原子分配批次索引
        batch_indices.extend([i] * data.natoms.item())
        
        # 调整边索引
        edge_index_adjusted = data.edge_index + total_atoms
        edge_index_list.append(edge_index_adjusted)
        edge_distance_list.append(data.edge_distance)
        
        total_atoms += data.natoms.item()
    
    # 合并数据
    batch_data.pos = torch.cat(pos_list, dim=0)
    batch_data.atomic_numbers = torch.cat(atomic_numbers_list, dim=0)
    batch_data.y = torch.cat(y_list, dim=0)
    batch_data.natoms = torch.tensor(natoms_list)
    batch_data.batch = torch.tensor(batch_indices, dtype=torch.long)
    
    if edge_index_list:
        batch_data.edge_index = torch.cat(edge_index_list, dim=1)
        batch_data.edge_distance = torch.cat(edge_distance_list, dim=0)
    else:
        batch_data.edge_index = torch.empty((2, 0), dtype=torch.long)
        batch_data.edge_distance = torch.empty((0,), dtype=torch.float)
    
    return batch_data

def custom_collate_fn(batch):
    """自定义collate函数"""
    return create_batch_from_data_list(batch)