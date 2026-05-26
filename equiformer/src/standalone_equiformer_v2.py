#!/usr/bin/env python3
"""
独立的EquiformerV2实现，基于项目核心组件但避免OCP框架依赖
使用真正的EquiformerV2架构进行氢吸附能量预测
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

# 导入EquiformerV2的核心组件
from equiformer_v2.nets.equiformer_v2.gaussian_rbf import GaussianRadialBasisLayer
from equiformer_v2.nets.equiformer_v2.so3 import SO3_Embedding, SO3_Grid, SO3_Rotation, SO3_LinearV2, CoefficientMappingModule
from equiformer_v2.nets.equiformer_v2.so2_ops import SO2_Convolution
from equiformer_v2.nets.equiformer_v2.layer_norm import get_normalization_layer
from equiformer_v2.nets.equiformer_v2.transformer_block import TransBlockV2, FeedForwardNetwork, SO2EquivariantGraphAttention
from equiformer_v2.nets.equiformer_v2.input_block import EdgeDegreeEmbedding
from equiformer_v2.nets.equiformer_v2.module_list import ModuleListInfo
from equiformer_v2.nets.equiformer_v2.edge_rot_mat import init_edge_rot_mat
from equiformer_v2.nets.equiformer_v2.radial_function import RadialFunction

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
            length_bytes = txn.get("length".encode("ascii"))
            if length_bytes is not None:
                self.length = pickle.loads(length_bytes)
            else:
                raise ValueError("Database does not contain 'length' key")
        
        print(f"Loaded dataset from {lmdb_path} with {self.length} samples")
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        with self.env.begin() as txn:
            data_bytes = txn.get(f"{idx}".encode("ascii"))
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

# 简化的高斯距离扩展（替代OCP的实现）
class SimpleGaussianSmearing(nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50, width=0.5):
        super().__init__()
        self.start = start
        self.stop = stop
        self.num_gaussians = num_gaussians
        self.width = width
        
        # 创建高斯中心
        offset = torch.linspace(start, stop, num_gaussians)
        self.register_buffer('offset', offset)
        self.num_output = num_gaussians
    
    def forward(self, distances):
        # 计算高斯基函数
        distances = distances.unsqueeze(-1) - self.offset
        return torch.exp(-self.width * distances ** 2)

# 独立的EquiformerV2模型
class StandaloneEquiformerV2(nn.Module):
    """
    独立的EquiformerV2实现
    基于原始EquiformerV2架构但避免OCP框架依赖
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
        norm_type='layer_norm_sh'
    ):
        super().__init__()
        
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
        
        if self.share_atom_edge_embedding:
            assert self.use_atom_edge_embedding
            self.block_use_atom_edge_embedding = False
        else:
            self.block_use_atom_edge_embedding = self.use_atom_edge_embedding
        
        # 基本统计
        self._avg_num_nodes = 77.81317
        self._avg_degree = 23.395238876342773
        
        self.num_resolutions = len(self.lmax_list)
        self.sphere_channels_all = self.num_resolutions * self.sphere_channels
        
        # 原子嵌入
        self.sphere_embedding = nn.Embedding(self.max_num_elements, self.sphere_channels_all)
        
        # 距离扩展
        self.distance_expansion = SimpleGaussianSmearing(0.0, self.max_radius, 128, 2.0)
        
        # 边特征通道
        self.edge_channels_list = [self.distance_expansion.num_output] + [self.edge_channels] * 2
        
        # 原子边嵌入
        if self.share_atom_edge_embedding and self.use_atom_edge_embedding:
            self.source_embedding = nn.Embedding(self.max_num_elements, self.edge_channels_list[-1])
            self.target_embedding = nn.Embedding(self.max_num_elements, self.edge_channels_list[-1])
            self.edge_channels_list[0] = self.edge_channels_list[0] + 2 * self.edge_channels_list[-1]
        else:
            self.source_embedding, self.target_embedding = None, None
        
        # SO3旋转模块
        self.SO3_rotation = nn.ModuleList()
        for i in range(self.num_resolutions):
            self.SO3_rotation.append(SO3_Rotation(self.lmax_list[i]))
        
        # 系数映射
        self.mappingReduced = CoefficientMappingModule(self.lmax_list, self.mmax_list)
        
        # SO3网格
        self.SO3_grid = ModuleListInfo('({}, {})'.format(max(self.lmax_list), max(self.lmax_list)))
        for l in range(max(self.lmax_list) + 1):
            SO3_m_grid = nn.ModuleList()
            for m in range(max(self.lmax_list) + 1):
                SO3_m_grid.append(
                    SO3_Grid(l, m, resolution=self.grid_resolution, normalization='component')
                )
            self.SO3_grid.append(SO3_m_grid)
        
        # 边度嵌入
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
        
        # Transformer块
        self.blocks = nn.ModuleList()
        for i in range(self.num_layers):
            block = TransBlockV2(
                self.sphere_channels,
                attn_hidden_channels,
                num_heads,
                attn_alpha_channels,
                attn_value_channels,
                ffn_hidden_channels,
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
                norm_type=norm_type,
                alpha_drop=alpha_drop,
                drop_path_rate=drop_path_rate,
                proj_drop=proj_drop
            )
            self.blocks.append(block)
        
        # 输出块
        self.norm = get_normalization_layer(norm_type, lmax=max(self.lmax_list), num_channels=self.sphere_channels)
        self.energy_block = FeedForwardNetwork(
            self.sphere_channels,
            ffn_hidden_channels,
            1,
            self.lmax_list,
            self.mmax_list,
            self.SO3_grid,
            activation='scaled_silu',
            use_gate_act=False,
            use_grid_mlp=False,
            use_sep_s2_act=True
        )
        
        # 初始化权重
        self.apply(self._init_weights)
        self.apply(self._uniform_init_rad_func_linear_weights)
    
    def _init_weights(self, m):
        if isinstance(m, (torch.nn.Linear, SO3_LinearV2)):
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
            std = 1 / math.sqrt(m.in_features)
            torch.nn.init.normal_(m.weight, 0, std)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)
    
    def _uniform_init_rad_func_linear_weights(self, m):
        if isinstance(m, RadialFunction):
            m.apply(self._uniform_init_linear_weights)
    
    def _uniform_init_linear_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.uniform_(m.weight, -0.1, 0.1)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
    
    def generate_graph(self, data):
        """生成图结构"""
        pos = data.pos
        batch = data.batch
        
        # 简单的距离图生成
        edge_index = data.edge_index
        edge_distance_vec = pos[edge_index[1]] - pos[edge_index[0]]
        edge_distance = torch.norm(edge_distance_vec, dim=1)
        
        return edge_index, edge_distance, edge_distance_vec, None, None, None
    
    def forward(self, data):
        """前向传播"""
        device = data.pos.device
        dtype = data.pos.dtype
        
        atomic_numbers = data.atomic_numbers.long()
        num_atoms = len(atomic_numbers)
        
        # 生成图
        edge_index, edge_distance, edge_distance_vec, _, _, _ = self.generate_graph(data)
        
        # 计算边旋转矩阵
        edge_rot_mat = init_edge_rot_mat(edge_distance_vec)
        
        # 初始化Wigner矩阵
        for i in range(self.num_resolutions):
            self.SO3_rotation[i].set_wigner(edge_rot_mat)
        
        # 初始化节点嵌入
        x = SO3_Embedding(
            num_atoms,
            self.lmax_list,
            self.sphere_channels,
            device,
            dtype,
        )
        
        # 设置初始嵌入
        offset_res = 0
        offset = 0
        for i in range(self.num_resolutions):
            if self.num_resolutions == 1:
                x.embedding[:, offset_res, :] = self.sphere_embedding(atomic_numbers)
            else:
                x.embedding[:, offset_res, :] = self.sphere_embedding(atomic_numbers)[:, offset : offset + self.sphere_channels]
            offset = offset + self.sphere_channels
            offset_res = offset_res + int((self.lmax_list[i] + 1) ** 2)
        
        # 边编码
        edge_distance_expanded = self.distance_expansion(edge_distance)
        if self.share_atom_edge_embedding and self.use_atom_edge_embedding:
            source_element = atomic_numbers[edge_index[0]]
            target_element = atomic_numbers[edge_index[1]]
            source_embedding = self.source_embedding(source_element)
            target_embedding = self.target_embedding(target_element)
            edge_distance_expanded = torch.cat((edge_distance_expanded, source_embedding, target_embedding), dim=1)
        
        # 边度嵌入
        edge_degree = self.edge_degree_embedding(atomic_numbers, edge_distance_expanded, edge_index)
        x.embedding = x.embedding + edge_degree.embedding
        
        # Transformer层
        for i in range(self.num_layers):
            x = self.blocks[i](x, atomic_numbers, edge_distance_expanded, edge_index, batch=data.batch)
        
        # 最终归一化
        x.embedding = self.norm(x.embedding)
        
        # 能量预测
        node_energy = self.energy_block(x)
        node_energy = node_energy.embedding.narrow(1, 0, 1)
        
        # 聚合到图级别 - 确保数据类型一致性
        energy = torch.zeros(len(data.natoms), device=device, dtype=node_energy.dtype)
        energy.index_add_(0, data.batch, node_energy.view(-1))
        energy = energy / self._avg_num_nodes
        
        return energy

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