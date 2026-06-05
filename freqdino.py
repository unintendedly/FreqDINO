import os, sys
import math
import json

from torch.distributed import destroy_process_group

sys.setrecursionlimit(15000)
import torch
import numpy as np
import random
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Sampler
import torch.nn.functional as F
import torchvision
import time
import albumentations as albu
from albumentations.pytorch import ToTensorV2
from sklearn import metrics
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import logging
from tqdm import tqdm
from get_args import list_args
import timm
from util import cal_metrics
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler
from hf_datasets import HFDataset
from datetime import datetime
import torch
# import pywt
from pytorch_wavelets import DWTForward, DWTInverse
import torch.distributed as dist
from datetime import timedelta
import matplotlib.pyplot as plt


def get_albu_transforms(type_='train', output_size=(1024, 1024),
                        mean=[0.48145466, 0.4578275, 0.40821073],
                        std=[0.26862954, 0.26130258, 0.27577711]):
    assert type_ in ['train', 'test', 'pad', 'resize'], "type_ must be 'train' or 'test' of 'pad' "
    trans = None
    if type_ == 'train':
        trans = albu.Compose([
            albu.RandomScale(scale_limit=0.2, p=1),
            albu.HorizontalFlip(p=0.5),
            albu.VerticalFlip(p=0.5),
            albu.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=0.1, p=1),
            albu.ImageCompression(quality_lower=70, quality_upper=100, p=0.2),
            albu.RandomRotate90(p=0.5),
            albu.GaussianBlur(blur_limit=(3, 7), p=0.2),
        ])

    if type_ == 'test':
        trans = albu.Compose([
        ])

    if type_ == 'pad':
        trans = albu.Compose([
            albu.PadIfNeeded(min_height=output_size[0], min_width=output_size[1], border_mode=0, value=0,
                             position='top_left', mask_value=0),
            albu.Normalize(mean=mean, std=std),
            albu.Crop(0, 0, output_size[0], output_size[1]),
            ToTensorV2(transpose_mask=True)
        ])
    if type_ == 'resize':
        trans = albu.Compose([
            albu.Resize(output_size[0], output_size[1]),
            albu.Normalize(mean=mean, std=std),
            ToTensorV2(transpose_mask=True)
        ])

    return trans


def post_transforms(type_='train', output_size=(1024, 1024),
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711]):
    assert type_ in ['train', 'test', 'pad', 'resize'], "type_ must be 'train' or 'test' of 'pad' "
    import albumentations as albu
    from albumentations.pytorch import ToTensorV2
    trans = None
    if type_ == 'pad':
        trans = albu.Compose([
            albu.PadIfNeeded(min_height=output_size[0], min_width=output_size[1], border_mode=0, value=0,
                             position='top_left', mask_value=0),
            albu.Normalize(mean=mean, std=std),
            albu.Crop(0, 0, output_size[0], output_size[1]),
            ToTensorV2(transpose_mask=True)
        ])
    if type_ == 'resize':
        trans = albu.Compose([
            albu.Resize(output_size[0], output_size[1]),
            albu.Normalize(mean=mean, std=std),
            albu.Crop(0, 0, output_size[0], output_size[1]),
            ToTensorV2(transpose_mask=True)
        ])
    return trans


def find_squares_numpy(n):
    """
    使用NumPy找到完全平方数
    """
    if n <= 0:
        return 0, 1, 1

    sqrt_n = np.sqrt(n)
    floor_sqrt = np.floor(sqrt_n).astype(int)
    ceil_sqrt = np.ceil(sqrt_n).astype(int)

    lower = floor_sqrt ** 2
    upper = ceil_sqrt ** 2

    # 如果n本身就是完全平方数
    if lower == n:
        upper = lower

    # 找到最近的
    if n - lower <= upper - n:
        nearest = lower
    else:
        nearest = upper

    return lower, upper, nearest


class RandomSubsetSampler(DistributedSampler):
    def __init__(self, data_source, num_samples, num_replicas=None, rank=None, shuffle=True, seed=0):
        super().__init__(data_source, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed)
        self.data_source = data_source
        self.num_samples = min(num_samples, len(data_source))
        self.epoch = 0
        self.seed = seed

        # 初始抽样
        self._generate_indices()

    def _generate_indices(self):
        """生成随机子集索引"""
        # 使用确定的随机种子，确保所有进程得到相同的结果
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        # 生成全局随机排列并取前num_samples个
        global_indices = torch.randperm(len(self.data_source), generator=g)[:self.num_samples].tolist()

        # 将全局索引分配给各个进程
        indices_per_process = len(global_indices) // self.num_replicas
        self.indices = global_indices[self.rank * indices_per_process: (self.rank + 1) * indices_per_process]

        # 如果无法均匀分配，将剩余样本分配给前几个进程
        if self.rank < len(global_indices) % self.num_replicas:
            extra_index = len(global_indices) - (self.rank + 1)
            if extra_index < len(global_indices):
                self.indices.append(global_indices[extra_index])

    def set_epoch(self, epoch):
        """设置epoch，用于在每个epoch开始时重新抽样"""
        super().set_epoch(epoch)
        self.epoch = epoch
        self._generate_indices()

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class DWTAdapter(nn.Module):
    def __init__(self, in_features, hidden_dim_ratio=4, wavelet='sym4', adapter_id=0):
        super().__init__()
        self.hidden_dim_ratio = hidden_dim_ratio
        self.upsampled_dim = int(in_features * hidden_dim_ratio)
        self.adapter_id = adapter_id

        # 上采样和下采样/降维和升维
        self.linear1 = nn.Linear(in_features, self.upsampled_dim)
        self.linear2 = nn.Linear(self.upsampled_dim, in_features)

        # 为每个频段定义完全独立的注意力机制
        # LL频段处理流程
        self.ll_attention = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(8, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )

        # LH频段处理流程
        self.lh_attention = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(8, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )

        # HL频段处理流程
        self.hl_attention = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(8, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )

        # HH频段处理流程
        self.hh_attention = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(8, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )

        self.activation = nn.GELU()

        # 初始化小波变换对象
        self.dwt = DWTForward(J=1, wave=wavelet, mode='zero')
        self.idwt = DWTInverse(wave=wavelet, mode='zero')

    def apply_attention_to_band(self, band, attention_module):
        """
        对单个频段应用独立的注意力机制：
        1. 3x3 conv -> GELU -> 3x3 conv -> Sigmoid 生成注意力图
        2. 注意力图与原始频段逐元素相乘
        3. 结果与原始频段相加（残差连接）
        """
        # 生成注意力图 [batch, 1, h, w]
        attention_map = attention_module(band)

        # 注意力加权 [batch, 1, h, w]
        weighted_band = band * attention_map

        # 残差连接 [batch, 1, h, w]
        enhanced_band = weighted_band + band

        return enhanced_band

    def forward(self, x):
        batch_size, seq_len, features = x.shape
        total_elements = batch_size * seq_len

        # 1. 上采样
        upsampled = self.linear1(x)  # [batch_size, seq_len, upsampled_dim]

        # 2. 重塑为2D
        side_len = int(np.sqrt(self.upsampled_dim))
        upsampled_2d = upsampled.reshape(total_elements, side_len, side_len)

        # 3. 添加通道维度并执行DWT
        upsampled_2d_channel = upsampled_2d.unsqueeze(1)  # [total_elements, 1, side_len, side_len]
        LL, YH = self.dwt(upsampled_2d_channel)
        highs = YH[0]
        LH, HL, HH = highs[:, :, 0], highs[:, :, 1], highs[:, :, 2]

        # 4. 对每个频段应用独立的注意力机制
        LL_enhanced = self.apply_attention_to_band(LL, self.ll_attention)
        LH_enhanced = self.apply_attention_to_band(LH, self.lh_attention)
        HL_enhanced = self.apply_attention_to_band(HL, self.hl_attention)
        HH_enhanced = self.apply_attention_to_band(HH, self.hh_attention)

        # 5. 逆DWT重构
        processed_highs = torch.stack([LH_enhanced, HL_enhanced, HH_enhanced], dim=2)
        reconstructed = self.idwt((LL_enhanced, [processed_highs]))  # [total_elements, 1, side_len, side_len]

        # 6. 展平、激活和下采样
        reconstructed_flat = reconstructed.squeeze(1).reshape(total_elements, -1)
        activated = self.activation(reconstructed_flat)
        output = self.linear2(activated)

        # 7. 恢复原始形状
        output_reshaped = output.reshape(batch_size, seq_len, features)

        return output_reshaped


class DWTLoRAAdapterLate(nn.Module):
    def __init__(self, in_features, out_features, adapter_dim_ratio=8, wavelet='sym4', mode='symmetric'):
        super().__init__()
        # Adapter网络结构
        f_length = {
            'haar': 2,
            'db1': 2,
            'db2': 4,
            'db4': 8,
            'sym4': 8,
            'sym5': 10,
            'coif3': 18,
            'coif4': 24,
        }
        self.in_features = in_features
        self.out_features = out_features
        self.filter_length = f_length[wavelet]
        flatten_length_in = ((int(math.sqrt(in_features)) + self.filter_length - 1) // 2) ** 2
        self.adapter_dim = max(1, int(flatten_length_in / adapter_dim_ratio))
        lower, upper, nearest = find_squares_numpy(self.adapter_dim)
        self.adapter_dim = lower
        self.down_proj_ll = nn.Linear(flatten_length_in, self.adapter_dim, bias=False)
        self.down_proj_lh = nn.Linear(flatten_length_in, self.adapter_dim, bias=False)
        self.down_proj_hl = nn.Linear(flatten_length_in, self.adapter_dim, bias=False)
        self.down_proj_hh = nn.Linear(flatten_length_in, self.adapter_dim, bias=False)

        self.activation_ll = nn.GELU()
        self.activation_lh = nn.LeakyReLU(negative_slope=0.05)
        self.activation_hl = nn.LeakyReLU(negative_slope=0.05)
        self.activation_hh = nn.ReLU()

        flatten_length_out = (int(math.sqrt(self.adapter_dim)) * 2 - self.filter_length + 2) ** 2
        self.up_proj = nn.Linear(flatten_length_out, out_features, bias=False)

        self.activation_all = nn.SiLU()

        # 小波变换对象 [citation:7]
        self.dwt = DWTForward(J=1, wave=wavelet, mode=mode)
        self.idwt = DWTInverse(wave=wavelet, mode=mode)
        self.freq_weights = nn.Parameter(torch.tensor([0.1, 4, 4, 4]))

        nn.init.zeros_(self.up_proj.weight)

    def dwt_process(self, x):
        """高效的高频特征提取"""
        batch_size, channels, h, w = x.shape

        # 调整到偶数尺寸
        h_even = h if h % 2 == 0 else h - 1
        w_even = w if w % 2 == 0 else w - 1

        if h_even != h or w_even != w:
            x = F.interpolate(x, size=(h_even, w_even), mode='bilinear', align_corners=False)

        # DWT分解 [citation:2]
        LL, YH = self.dwt(x)
        highs = YH[0]
        LH, HL, HH = highs[:, :, 0], highs[:, :, 1], highs[:, :, 2]
        return LL, LH, HL, HH

    def idwt_process(self, LL, LH, HL, HH):
        # 逆DWT重构
        processed_highs = torch.stack([LH, HL, HH], dim=2)
        reconstructed = self.idwt((LL, [processed_highs]))
        return reconstructed

    def forward(self, x):
        batch_size, seq_len, features = x.shape
        # print(f'x: {x.shape}')

        # 重塑为2D - 自动处理维度
        total_elements = batch_size * seq_len
        side_len = int(np.sqrt(features))

        if side_len * side_len == features:
            # 完美平方数
            x_2d = x.reshape(total_elements, 1, side_len, side_len)
        else:
            # 非完美平方数，使用最近的可能形状
            possible_size = int(np.sqrt(features))
            target_features = possible_size * possible_size
            x_flat = x.reshape(total_elements, -1)

            if target_features > features:
                padding = torch.zeros(total_elements, target_features - features,
                                      device=x.device, dtype=x.dtype)
                x_padded = torch.cat([x_flat, padding], dim=1)
            else:
                x_padded = x_flat[:, :target_features]

            x_2d = x_padded.reshape(total_elements, 1, possible_size, possible_size)

        # DWT处理
        LL, LH, HL, HH = self.dwt_process(x_2d)
        # print(f'LL: {LL.shape}')
        _, _, fre_size, _ = LL.shape
        LL = LL.reshape(total_elements, 1, -1)
        LH = LH.reshape(total_elements, 1, -1)
        HL = HL.reshape(total_elements, 1, -1)
        HH = HH.reshape(total_elements, 1, -1)

        # HFEF 高频增强
        LL = LL * self.freq_weights[0]  # 抑制低频
        LH = LH * self.freq_weights[1]
        HL = HL * self.freq_weights[2]
        HH = HH * self.freq_weights[3]
        # 降维
        LL = self.activation_ll(self.down_proj_ll(LL))
        LH = self.activation_lh(self.down_proj_lh(LH))
        HL = self.activation_hl(self.down_proj_hl(HL))
        HH = self.activation_hh(self.down_proj_hh(HH))
        # print(f'LL_down: {LL.shape}')
        _, _, feature_len = LL.shape
        s_feature_len = int(np.sqrt(feature_len))
        LL = LL.reshape(total_elements, 1, s_feature_len, s_feature_len)
        LH = LH.reshape(total_elements, 1, s_feature_len, s_feature_len)
        HL = HL.reshape(total_elements, 1, s_feature_len, s_feature_len)
        HH = HH.reshape(total_elements, 1, s_feature_len, s_feature_len)

        x_dwt_2d = self.idwt_process(LL, LH, HL, HH)

        x_dwt = x_dwt_2d.reshape(batch_size, seq_len, -1)
        return self.activation_all(self.up_proj(x_dwt))


class AdapterLinear(nn.Module):
    """
    带有Adapter适配器的线性层
    """

    def __init__(self, linear_layer, adapter_dim_ratio=8, wavelet='db4', use_residual=True):
        super(AdapterLinear, self).__init__()
        self.linear = linear_layer
        self.adapter = DWTLoRAAdapterLate(
            linear_layer.in_features, linear_layer.out_features,
            adapter_dim_ratio=adapter_dim_ratio, wavelet=wavelet
        )
        self.use_residual = use_residual

    def forward(self, x):
        # 4. 残差连接
        if self.use_residual:
            # 原始线性层输出 + DWT适配
            return self.linear(x) + self.adapter(x)
        else:
            # DWT适配
            return self.adapter(x)

    def __getattr__(self, name):
        # 对于其他属性，从原始线性层获取
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.linear, name)


def apply_adapter_to_dinov3(model, adapter_dim_ratio=8, wavelet='db4', use_residual=True):
    """
    手动为DINOv3模型应用DWT适配器
    """
    print("应用DWT适配器实现到DINOv3模型...")

    # 记录被替换的模块数量
    replaced_modules = 0

    # 遍历所有模块，找到需要添加Adapter的线性层
    for name, module in model.named_modules():
        # 针对注意力机制的关键层添加Adapter
        if isinstance(module, nn.Linear):
            # 检查是否是注意力机制或MLP中的关键层
            target_modules = ['qkv', 'proj', 'fc1', 'fc2']
            if any(target in name.lower() for target in target_modules):
                # 获取父模块和属性名
                parent = model
                path = name.split('.')
                for p in path[:-1]:
                    parent = getattr(parent, p)

                attr_name = path[-1]
                original_linear = getattr(parent, attr_name)

                # 替换为AdapterLinear
                adapter_linear = AdapterLinear(original_linear, adapter_dim_ratio=adapter_dim_ratio,
                                               wavelet=wavelet, use_residual=use_residual)
                setattr(parent, attr_name, adapter_linear)
                replaced_modules += 1
                # print(f"为层 {name} 添加Adapter适配器")

    print(f"总共为 {replaced_modules} 个线性层添加了Adapter适配器")

    # 打印可训练参数信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print(f"总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    print(f"冻结参数: {frozen_params:,}")
    print(f"可训练参数占比: {trainable_params / total_params * 100:.2f}%")

    return model


class EnhancedDINOv3Detector(nn.Module):
    def __init__(self, backbone, num_classes=2, feature_dim=1024, adapter_interval=3, hidden_dim_ratio=4):
        super(EnhancedDINOv3Detector, self).__init__()
        self.backbone = backbone
        self.feature_dim = feature_dim
        self.adapter_interval = adapter_interval

        # 获取backbone的blocks
        self.blocks = backbone.blocks

        # 添加adapters - 每adapter_interval个block添加一个，从第adapter_interval个开始
        self.adapters = nn.ModuleList()
        num_blocks = len(self.blocks)
        for i in range(adapter_interval - 1, num_blocks, adapter_interval):
            if i < num_blocks:
                adapter = DWTAdapter(feature_dim, hidden_dim_ratio=hidden_dim_ratio, wavelet=args.wavelet)
                self.adapters.append(adapter)

        # 自注意力层
        self.self_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=8,
            batch_first=True
        )

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim * 2, feature_dim)
        )

        # 新的分类头 - 只需要处理global token的特征
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim // 2, num_classes)
        )

        # 总的token (可学习参数)
        self.global_token = nn.Parameter(torch.randn(1, 1, feature_dim))

    def forward(self, x, masks=None):
        if x.dim() == 3:
            x = x.unsqueeze(0)

        batch_size = x.shape[0]

        # 使用backbone的prepare_tokens_with_masks准备tokens
        x, (H, W) = self.backbone.prepare_tokens_with_masks(x, masks)

        # 存储adapter处理过的CLS tokens
        adapter_cls_tokens = []

        # 通过blocks并应用adapters
        adapter_idx = 0
        for i, block in enumerate(self.blocks):
            # 计算rope_sincos
            if hasattr(self.backbone, 'rope_embed') and self.backbone.rope_embed is not None:
                rope_sincos = self.backbone.rope_embed(H=H, W=W)
            else:
                rope_sincos = None

            # 通过block的前半部分（norm1 + attention）
            x_attn = self._forward_block_attention(block, x, rope_sincos)

            # 每adapter_interval个block后应用adapter
            if i >= (self.adapter_interval - 1) and (
                    i - (self.adapter_interval - 1)) % self.adapter_interval == 0 and adapter_idx < len(self.adapters):
                # 并行执行：MLP + Adapter
                x = self._forward_mlp_adapter_parallel(block, x_attn, adapter_idx)

                # 提取CLS token (第一个token)
                cls_token = x[:, 0, :]  # [batch_size, feature_dim]
                adapter_cls_tokens.append(cls_token)
                adapter_idx += 1
            else:
                # 正常执行block的MLP部分
                x = self._forward_block_mlp(block, x_attn)

        # 应用最终的norm层
        x = self.backbone.norm(x)

        # 处理norm后的特征
        if self.backbone.untie_cls_and_patch_norms or self.backbone.untie_global_and_local_cls_norm:
            x_norm_cls_reg = x[:, :self.backbone.n_storage_tokens + 1]

            if self.backbone.untie_global_and_local_cls_norm and self.training:
                x_norm_cls_reg = self.backbone.local_cls_norm(x_norm_cls_reg)
            elif self.backbone.untie_cls_and_patch_norms:
                x_norm_cls_reg = self.backbone.cls_norm(x_norm_cls_reg)
            else:
                x_norm_cls_reg = self.backbone.norm(x_norm_cls_reg)
        else:
            x_norm = x
            x_norm_cls_reg = x_norm[:, :self.backbone.n_storage_tokens + 1]

        # 获取最终的CLS token
        final_cls_token = x_norm_cls_reg[:, 0]  # [batch_size, feature_dim]

        # 如果没有adapters被应用，使用最终CLS token
        if len(adapter_cls_tokens) == 0:
            adapter_cls_tokens = [final_cls_token]

        # 重塑为序列形式 [batch_size, num_tokens, feature_dim]
        num_adapters = len(adapter_cls_tokens)
        if num_adapters > 0:
            # 将每个CLS token转换为序列形式
            cls_sequence = torch.stack(adapter_cls_tokens, dim=1)  # [batch_size, num_adapters, feature_dim]
        else:
            cls_sequence = final_cls_token.unsqueeze(1)  # [batch_size, 1, feature_dim]

        # 添加全局token
        global_tokens = self.global_token.expand(batch_size, -1, -1)  # [batch_size, 1, feature_dim]

        # 拼接CLS tokens和global token
        cls_sequence_with_global = torch.cat([cls_sequence, global_tokens],
                                             dim=1)  # [batch_size, num_adapters+1, feature_dim]

        # 自注意力 - 处理拼接后的序列
        attended, _ = self.self_attention(
            cls_sequence_with_global,
            cls_sequence_with_global,
            cls_sequence_with_global
        )  # [batch_size, num_adapters+1, feature_dim]

        # MLP - 处理自注意力后的整个序列
        mlp_output = self.mlp(attended)  # [batch_size, num_adapters+1, feature_dim]

        # 提取global token (序列的最后一个token)
        global_token_output = mlp_output[:, -1, :]  # [batch_size, feature_dim]

        # 分类头 - 只使用global token的输出
        output = self.classifier(global_token_output)

        return output

    def _forward_block_attention(self, block, x, rope_sincos):
        """执行block的attention部分"""
        # norm1 + attention
        x_norm1 = block.norm1(x)
        x_attn = block.attn(x_norm1, rope=rope_sincos)

        # layerscale1
        if hasattr(block, 'ls1') and not isinstance(block.ls1, nn.Identity):
            x_attn = block.ls1(x_attn)

        # 残差连接
        x_attn_out = x + x_attn
        return x_attn_out

    def _forward_block_mlp(self, block, x_attn):
        """正常执行block的MLP部分"""
        # norm2 + mlp
        x_norm2 = block.norm2(x_attn)
        x_mlp = block.mlp(x_norm2)

        # layerscale2
        if hasattr(block, 'ls2') and not isinstance(block.ls2, nn.Identity):
            x_mlp = block.ls2(x_mlp)

        # 残差连接
        x_mlp_out = x_attn + x_mlp
        return x_mlp_out

    def _forward_mlp_adapter_parallel(self, block, x_attn, adapter_idx):
        """并行执行MLP和Adapter，然后合并结果"""
        # norm2
        x_norm2 = block.norm2(x_attn)

        # 并行执行MLP和Adapter
        x_mlp = block.mlp(x_norm2)
        x_adapter = self.adapters[adapter_idx](x_norm2)

        # 合并MLP和Adapter的输出
        x_combined = x_mlp + x_adapter

        # layerscale2
        if hasattr(block, 'ls2') and not isinstance(block.ls2, nn.Identity):
            x_combined = block.ls2(x_combined)

        # 残差连接
        x_out = x_attn + x_combined
        return x_out


def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def train_standard(args, model, device, optimizer, train_loader, valid_loader, save_dir, writer):
    """标准训练（保持原有训练逻辑）"""
    # 根据是否分布式设置当前设备
    if args.distributed:
        # 分布式训练中，每个进程使用自己的设备
        current_device = torch.device(f"cuda:{args.local_rank}")
    else:
        # 单机训练使用主设备
        current_device = device

    global_step = 0
    criterion = nn.CrossEntropyLoss()

    # 混合精度设置
    if current_device.type == 'cuda':
        scaler = torch.amp.GradScaler('cuda')
        use_amp = True
    else:
        use_amp = False

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # checkpoint
    if args.resume > -1:
        checkpoint = torch.load(os.path.join(save_dir, 'models_params_{}.tar'.format(args.resume)),
                                map_location='cuda:{}'.format(torch.cuda.current_device()))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict((checkpoint['optimizer_state_dict']))

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5, last_epoch=args.resume)

    if args.rank == 0:  # 只在主进程打印
        print(f'train_loader: {len(train_loader)}')

    for epoch in range(args.resume + 1, args.epochs):
        # train part
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)

        if args.rank == 0:  # 只在主进程打印
            print('start train mode...')

        epoch_loss = 0.0
        total_num = 0
        correct_num = 0
        model.train()

        with torch.enable_grad():
            st_time = time.time()
            for i, data in enumerate(train_loader):
                inputs, labels = data['image'], data['label']
                optimizer.zero_grad()
                inputs = inputs.to(current_device, non_blocking=True)
                labels = labels.to(current_device, non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                loss.backward()

                optimizer.step()
                epoch_loss += loss.item()
                total_num += inputs.size(0)
                correct_num += torch.sum(torch.argmax(outputs, 1) == labels).item()
                global_step += 1

                # record train stat into tensorboardX
                if global_step % args.record_step == 0:
                    if args.rank == 0:  # 只在主进程计算和记录
                        period = time.time() - st_time
                        train_acc = torch.mean((torch.argmax(outputs, 1) == labels).float()).item()
                        log.info(
                            'Training state: Epoch [{:0>3}/{:0>3}], Iteration [{:0>3}/{:0>3}], Loss: {:.4f} Acc:{:.4} time:{}m {}s'
                            .format(epoch + 1, args.epochs, i + 1, len(train_loader), epoch_loss / (i + 1),
                                    correct_num / total_num, int(period // 60), int(period % 60)))
                        if args.writer:
                            writer.add_scalar('loss', loss.item(), global_step)
                            writer.add_scalar('train_acc', train_acc, global_step)
                    st_time = time.time()
                    total_num = 0
                    correct_num = 0

        # eval part - 原有验证集测试
        if args.rank == 0:  # 只在主进程打印
            print('start eval mode...')
        model.eval()

        # 使用张量收集所有进程的预测结果
        all_frame_predictions = []
        all_frame_labels = []
        all_video_predictions = []
        all_video_labels = []

        with torch.no_grad():
            for data in tqdm(valid_loader, total=len(valid_loader), ncols=70, leave=False, unit='step'):
                inputs, labels = data['image'], data['label']
                inputs = inputs.to(current_device)
                labels = labels.to(current_device)

                batch_size = inputs.shape[0]

                # 统一处理逻辑，避免频繁的CPU-GPU数据传输
                if inputs.dim() == 5:
                    # [batch_size, frames, C, H, W] 格式
                    batch_size, num_frames = inputs.shape[0], inputs.shape[1]

                    # 重塑输入以便批量处理
                    inputs_reshaped = inputs.view(-1, *inputs.shape[2:])
                    outputs = model(inputs_reshaped)
                    outputs = F.softmax(outputs, dim=-1)

                    # 重塑输出回 [batch_size, num_frames, num_classes]
                    outputs = outputs.view(batch_size, num_frames, -1)

                    # 获取正类概率 [batch_size, num_frames]
                    frame_probs = outputs[:, :, 1]

                    # 视频级预测 [batch_size]
                    video_probs = frame_probs.mean(dim=1)

                    # 处理每个样本
                    for i in range(batch_size):
                        # 帧级预测
                        current_frame_probs = frame_probs[i]
                        current_label = labels[i]

                        # 收集到列表
                        all_frame_predictions.extend(current_frame_probs.cpu().tolist())
                        all_frame_labels.extend([current_label.item()] * num_frames)
                        all_video_predictions.append(video_probs[i].cpu().item())
                        all_video_labels.append(current_label.item())

                else:
                    # [batch_size, C, H, W] 格式（单帧情况）
                    outputs = model(inputs)
                    outputs = F.softmax(outputs, dim=-1)

                    # 获取正类概率 [batch_size]
                    frame_probs = outputs[:, 1]
                    video_probs = frame_probs  # 单帧情况下视频级预测就是帧级预测

                    for i in range(batch_size):
                        current_prob = frame_probs[i]
                        current_label = labels[i]

                        all_frame_predictions.append(current_prob.cpu().item())
                        all_frame_labels.append(current_label.item())
                        all_video_predictions.append(current_prob.cpu().item())
                        all_video_labels.append(current_label.item())
        # 在分布式训练中，收集所有进程的结果
        if args.distributed:
            # 将所有列表转换为张量以便跨进程通信
            frame_pred_tensor = torch.tensor(all_frame_predictions, device=current_device)
            frame_label_tensor = torch.tensor(all_frame_labels, device=current_device)
            video_pred_tensor = torch.tensor(all_video_predictions, device=current_device)
            video_label_tensor = torch.tensor(all_video_labels, device=current_device)

            # 收集所有进程的张量
            gathered_frame_preds = [torch.zeros_like(frame_pred_tensor) for _ in range(args.world_size)]
            gathered_frame_labels = [torch.zeros_like(frame_label_tensor) for _ in range(args.world_size)]
            gathered_video_preds = [torch.zeros_like(video_pred_tensor) for _ in range(args.world_size)]
            gathered_video_labels = [torch.zeros_like(video_label_tensor) for _ in range(args.world_size)]

            dist.all_gather(gathered_frame_preds, frame_pred_tensor)
            dist.all_gather(gathered_frame_labels, frame_label_tensor)
            dist.all_gather(gathered_video_preds, video_pred_tensor)
            dist.all_gather(gathered_video_labels, video_label_tensor)

            # 只在主进程计算全局指标
            if args.rank == 0:
                # 合并所有进程的结果
                global_frame_preds = torch.cat(gathered_frame_preds).cpu().tolist()
                global_frame_labels = torch.cat(gathered_frame_labels).cpu().tolist()
                global_video_preds = torch.cat(gathered_video_preds).cpu().tolist()
                global_video_labels = torch.cat(gathered_video_labels).cpu().tolist()

                frame_results = cal_metrics(global_frame_labels, global_frame_preds, threshold=0.5)
                video_results = cal_metrics(global_video_labels, global_video_preds, threshold=0.5)
            else:
                # 其他进程跳过指标计算
                frame_results = None
                video_results = None
        else:
            # 单卡训练直接计算
            frame_results = cal_metrics(all_frame_labels, all_frame_predictions, threshold=0.5)
            video_results = cal_metrics(all_video_labels, all_video_predictions, threshold=0.5)
        # 只在主进程记录结果
        if args.rank == 0 and frame_results is not None and video_results is not None:
            log.info(
                'valid result: Epoch [{:0>3}/{:0>3}], Video_Acc: {:.4}, Video_Auc: {:.4} Video_EER:{:.4} Frame_Acc: {:.4}, Frame_Auc: {:.4} Frame_EER:{:.4}'
                .format(epoch + 1, args.epochs, video_results.ACC, video_results.AUC, video_results.EER,
                        frame_results.ACC,
                        frame_results.AUC, frame_results.EER))
            log.info(
                'valid result: Epoch [{:0>3}/{:0>3}], Frame_F1:{:.4}, Frame_Acc: {:.4}, '
                .format(epoch + 1, args.epochs, frame_results.F1, frame_results.ACC))

        # save model
        state = {'model_state_dict': model.state_dict(),
                 'optimizer_state_dict': optimizer.state_dict(),
                 'epoch': epoch}
        torch.save(state, os.path.join(save_dir, 'models_params_{}.tar'.format(epoch)))

        scheduler.step()
    return


def test(args, model, test_loader, device, no_load=False):
    """向后兼容的测试函数，支持batch_size=1和batch_size>1"""
    if not no_load and args.rank == 0:
        checkpoint = torch.load(save_dir, map_location='cuda:{}'.format(torch.cuda.current_device()))
        if args.rank == 0:  # 只在主进程打印
            print('load model from {}'.format(save_dir))
        model.load_state_dict(checkpoint['model_state_dict'])

    # 确保所有进程模型同步（如果加载了模型）
    if args.distributed and not no_load:
        dist.barrier()

    if args.rank == 0:  # 只在主进程打印
        print('start test mode...')
    model.eval()

    # 收集预测结果
    all_video_predictions = []
    all_video_labels = []
    all_frame_predictions = []
    all_frame_labels = []

    with torch.no_grad():
        for data in tqdm(test_loader, total=len(test_loader), ncols=70, leave=False, unit='step'):
            inputs, labels = data['image'], data['label']
            inputs = inputs.to(device)
            labels = labels.to(device)
            # 确保标签是整数类型
            labels = labels.long()

            batch_size = inputs.shape[0]

            # 检查是否是传统的batch_size=1格式
            if batch_size == 1 and inputs.dim() == 5 and inputs.shape[1] > 1:
                # 传统格式: [1, frames, C, H, W]
                inputs = inputs.squeeze(0)  # 变为 [frames, C, H, W]
                outputs = model(inputs)
                outputs = F.softmax(outputs, dim=-1)

                # 帧级预测
                frame = outputs.shape[0]
                all_frame_predictions.extend(outputs[:, 1].cpu().tolist())
                all_frame_labels.extend(labels.expand(frame).cpu().tolist())

                # 视频级预测
                pre = torch.mean(outputs[:, 1])
                all_video_predictions.append(pre.cpu().item())
                all_video_labels.append(labels.cpu().item())

            else:
                # 新的batch_size>1格式
                if inputs.dim() == 5:
                    num_frames = inputs.shape[1]
                    inputs_reshaped = inputs.reshape(-1, *inputs.shape[2:])
                else:
                    num_frames = 1
                    inputs_reshaped = inputs

                outputs = model(inputs_reshaped)
                outputs = F.softmax(outputs, dim=-1)

                if num_frames > 1:
                    outputs = outputs.reshape(batch_size, num_frames, -1)

                for i in range(batch_size):
                    if num_frames > 1:
                        frame_probs = outputs[i, :, 1].cpu().tolist()
                        video_prob = torch.mean(outputs[i, :, 1]).cpu().item()
                    else:
                        frame_probs = [outputs[i, 1].cpu().item()]
                        video_prob = frame_probs[0]

                    all_frame_predictions.extend(frame_probs)
                    all_frame_labels.extend([labels[i].item()] * len(frame_probs))
                    all_video_predictions.append(video_prob)
                    all_video_labels.append(labels[i].item())
    # 在分布式训练中，收集所有进程的结果
    if args.distributed:
        # 转换为张量进行跨进程通信
        frame_pred_tensor = torch.tensor(all_frame_predictions, device=device)
        frame_label_tensor = torch.tensor(all_frame_labels, device=device)
        video_pred_tensor = torch.tensor(all_video_predictions, device=device)
        video_label_tensor = torch.tensor(all_video_labels, device=device)

        # 收集所有进程的结果
        gathered_frame_preds = [torch.zeros_like(frame_pred_tensor) for _ in range(args.world_size)]
        gathered_frame_labels = [torch.zeros_like(frame_label_tensor) for _ in range(args.world_size)]
        gathered_video_preds = [torch.zeros_like(video_pred_tensor) for _ in range(args.world_size)]
        gathered_video_labels = [torch.zeros_like(video_label_tensor) for _ in range(args.world_size)]

        dist.all_gather(gathered_frame_preds, frame_pred_tensor)
        dist.all_gather(gathered_frame_labels, frame_label_tensor)
        dist.all_gather(gathered_video_preds, video_pred_tensor)
        dist.all_gather(gathered_video_labels, video_label_tensor)

        # 只在主进程计算全局指标
        if args.rank == 0:
            global_frame_preds = torch.cat(gathered_frame_preds).cpu().tolist()
            global_frame_labels = torch.cat(gathered_frame_labels).cpu().tolist()
            global_video_preds = torch.cat(gathered_video_preds).cpu().tolist()
            global_video_labels = torch.cat(gathered_video_labels).cpu().tolist()

            # 调试：打印标签的唯一值
            unique_frame_labels = set(global_frame_labels)

            # 确保标签是0或1
            if not all(label in [0, 1] for label in global_frame_labels):
                print(f"Warning: Frame labels contain values other than 0 or 1: {unique_frame_labels[:10]}")
            # 如果标签是-1和1，将其转换为0和1
            if unique_frame_labels == {-1, 1}:
                print("Converting labels from {-1, 1} to {0, 1}")
                global_frame_labels = [1 if x == 1 else 0 for x in global_frame_labels]
                global_video_labels = [1 if x == 1 else 0 for x in global_video_labels]
            frame_results = cal_metrics(global_frame_labels, global_frame_preds, threshold=0.5)
            video_results = cal_metrics(global_video_labels, global_video_preds, threshold=0.5)
        else:
            frame_results, video_results = None, None
    else:
        # 单卡训练直接计算
        unique_frame_labels = set(all_frame_labels)
        # print(f"Debug - Unique frame labels: {unique_frame_labels}")

        # 确保标签是0或1
        if not all(label in [0, 1] for label in all_frame_labels):
            print(f"Warning: Frame labels contain values other than 0 or 1: {unique_frame_labels}")
        # 如果标签是-1和1，将其转换为0和1
        if unique_frame_labels == {-1, 1}:
            print("Converting labels from {-1, 1} to {0, 1}")
            all_frame_labels = [1 if x == 1 else 0 for x in all_frame_labels]
            all_video_labels = [1 if x == 1 else 0 for x in all_video_labels]

        frame_results = cal_metrics(all_frame_labels, all_frame_predictions, threshold=0.5)
        video_results = cal_metrics(all_video_labels, all_video_predictions, threshold=0.5)

    # 只在主进程记录结果
    if args.rank == 0 and frame_results is not None and video_results is not None:
        log.info(
            'Test result: Video_Acc: {:.4f}, Video_Auc: {:.4f}, Video_EER: {:.4f}, '
            'Frame_Acc: {:.4f}, Frame_Auc: {:.4f}, Frame_EER: {:.4f}'
            .format(video_results.ACC, video_results.AUC, video_results.EER,
                    frame_results.ACC, frame_results.AUC, frame_results.EER))
        log.info('Test result: Frame_F1: {:.4f}, Frame_Acc: {:.4f}'.format(frame_results.F1, frame_results.ACC))

    return video_results, frame_results


def setup_distributed_training():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://', timeout=timedelta(seconds=1200))
        args.distributed = True
        print(
            f"Initialized distributed training: rank {args.rank}, world_size {args.world_size}, local_rank {args.local_rank}")
    else:
        args.rank = 0
        args.world_size = 1
        args.distributed = False
        args.local_rank = 0
        print("Running in single GPU mode")


def freeze_backbone_params(model):
    """冻结backbone的所有参数"""
    # 冻结整个backbone
    for param in model.parameters():
        param.requires_grad = False

    # 但保留backbone的CLS token和norm层可训练（如果需要）
    # 如果需要完全冻结，可以注释掉下面几行
    if hasattr(model, 'cls_token'):
        model.cls_token.requires_grad = True
    if hasattr(model, 'norm'):
        for param in model.norm.parameters():
            param.requires_grad = True

    print("=== 参数冻结状态 ===")
    trainable_params = 0
    total_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    print(f"总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    print(f"冻结参数: {total_params - trainable_params:,}")
    print(f"可训练比例: {trainable_params / total_params * 100:.2f}%")


if __name__ == '__main__':
    start_time = time.time()
    setup_seed(2025)
    torch.multiprocessing.set_sharing_strategy('file_system')
    args = list_args()
    setup_distributed_training()

    # 添加DINOv3相关导入
    prefix = '/data1'
    REPO_DIR = os.path.join(prefix, "dinov3-main/")
    if args.rank == 0:  # 只在主进程打印
        print(f'REPO_DIR: {REPO_DIR}')

    if args.backbone_size == 'small':
        filename = 'dinov3_vits16_pretrain_lvd1689m-08c60483.pth'
    elif args.backbone_size == 'base':
        filename = 'dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth'
    elif args.backbone_size == 'large':
        filename = 'dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth'
    else:
        raise Exception("backbone_size<UNK>")
    WEIGHTS_DIR = os.path.join(prefix, 'weights/dinov3/', filename)
    if args.rank == 0:  # 只在主进程打印
        print(f'WEIGHTS_DIR: {WEIGHTS_DIR}')

    data_path = os.path.join(prefix, 'dataset/datasets--nebula--OpenSDI_train/',
                             'snapshots/73aee8053cf3b21eb2507db01c2c56ae749db60c')
    test_data_path = os.path.join(prefix, 'dataset/datasets--nebula--OpenSDI_test/',
                                  'snapshots/7e233eaf98fcfee4c74c788f0e34d06feb7ad0df')
    train_split_name = 'sd15'
    test_split_name = 'sd15'
    image_size = args.image_size

    # 用于日志文件名
    if args.test_mode or args.resume != -1:
        save_dir = os.path.join(prefix, "FreqDINO/checkpoints/", args.model_dir)
    else:
        time_start = datetime.now().strftime('%m%d_%H%M')
        if args.rank == 0:  # 只在主进程打印
            print(f'time:  {time_start}')
        save_dir = os.path.join(prefix, "FreqDINO/checkpoints/", args.model_dir, time_start)
        if args.rank == 0:  # 只在主进程打印
            print(f'save_dir:  {save_dir}')
        os.makedirs(save_dir, exist_ok=True)

    # logging
    if args.resume == -1:
        mode = 'w'
    else:
        mode = 'a'
    if args.test_mode:
        log_name = 'test.log'
    else:
        log_name = 'train.log'
    logging.basicConfig(
        filename=os.path.join(save_dir, log_name),
        filemode=mode,
        format='%(asctime)s: %(levelname)s: [%(filename)s:%(lineno)d]: %(message)s',
        level=logging.INFO)
    log = logging.getLogger()
    log.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    log.addHandler(handler)

    if args.rank == 0:  # 只在主进程打印
        print("Loading DINOv3 model...")
    if args.backbone_size == 'small':
        dinov3_vit = torch.hub.load(REPO_DIR, 'dinov3_vits16', source='local', weights=WEIGHTS_DIR)
        freeze_backbone_params(dinov3_vit)
        dinov3_vit = apply_adapter_to_dinov3(dinov3_vit, adapter_dim_ratio=args.adapter_dim_ratio, wavelet=args.wavelet,
                                             use_residual=True)
        model = EnhancedDINOv3Detector(dinov3_vit, num_classes=2, feature_dim=384,
                                       adapter_interval=args.adapter_interval, hidden_dim_ratio=args.hidden_dim_ratio)
    elif args.backbone_size == 'base':
        dinov3_vit = torch.hub.load(REPO_DIR, 'dinov3_vitb16', source='local', weights=WEIGHTS_DIR)
        freeze_backbone_params(dinov3_vit)
        dinov3_vit = apply_adapter_to_dinov3(dinov3_vit, adapter_dim_ratio=args.adapter_dim_ratio, wavelet=args.wavelet,
                                             use_residual=True)
        model = EnhancedDINOv3Detector(dinov3_vit, num_classes=2, feature_dim=768,
                                       adapter_interval=args.adapter_interval, hidden_dim_ratio=args.hidden_dim_ratio)
    elif args.backbone_size == 'large':
        dinov3_vit = torch.hub.load(REPO_DIR, 'dinov3_vitl16', source='local', weights=WEIGHTS_DIR)
        freeze_backbone_params(dinov3_vit)
        dinov3_vit = apply_adapter_to_dinov3(dinov3_vit, adapter_dim_ratio=args.adapter_dim_ratio, wavelet=args.wavelet,
                                             use_residual=True)
        model = EnhancedDINOv3Detector(dinov3_vit, num_classes=2, feature_dim=1024,
                                       adapter_interval=args.adapter_interval, hidden_dim_ratio=args.hidden_dim_ratio)
    else:
        dinov3_vit = None
        model = None
        raise Exception("backbone_size<UNK>")
    # print(model)
    print("=== 参数冻结状态 ===")
    trainable_params = 0
    total_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    print(f"总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    print(f"冻结参数: {total_params - trainable_params:,}")
    print(f"可训练比例: {trainable_params / total_params * 100:.2f}%")

    # 自动设备设置逻辑
    if args.distributed:
        # 分布式训练 - 使用当前进程的 GPU
        device = torch.device(f"cuda:{args.local_rank}")
        model = model.to(device)
        # 使用 DistributedDataParallel
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True
        )
        print(f"Using DistributedDataParallel on GPU {args.local_rank}")
    else:
        # 单机训练 - 使用所有可见 GPU
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            device = torch.device("cuda:0")
            model = model.to(device)

            if num_gpus > 1:
                model = torch.nn.DataParallel(model)
                print(f"Using DataParallel on {num_gpus} GPUs")
            else:
                print("Using single GPU")
        else:
            device = torch.device("cpu")
            print("Using CPU for training")
    if args.rank == 0:  # 只在主进程打印
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name()}")

    # loss function and optimizer
    criterion = nn.CrossEntropyLoss()

    if args.rank == 0:  # 只在主进程打印
        print('Start train process...')

    if args.writer:
        writer = SummaryWriter(log_dir=os.path.join(save_dir, 'tensorboard'))
    else:
        writer = None

    if args.test_mode:
        if_padding = False
        if_resizing = True

        cross_test_transform = get_albu_transforms('test')
        post_transform = post_transforms(type_="pad" if if_padding else "resize",
                                         output_size=(image_size, image_size))

        prefix = ""
        dataset_name = os.path.join(prefix, "dataset/datasets--nebula--OpenSDI_test/",
                                    "snapshots/7e233eaf98fcfee4c74c788f0e34d06feb7ad0df")
        for split in ["sd15", "sd2", "sdxl", "sd3", "flux"]:
            if args.rank == 0:  # 只在主进程打印
                log.info(f'----------------split: {split} ----------------')
            dataset = HFDataset(
                dataset_name, split, pixel=False,
                is_padding=if_padding,
                is_resizing=if_resizing,
                output_size=(image_size, image_size),
                common_transforms=cross_test_transform,
                post_transform=post_transform,
                edge_width=7,
            )
            cross_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                      num_workers=args.num_workers, pin_memory=True)
            test(args, model, cross_loader, device)
    else:
        if_padding = False
        if_resizing = True
        train_transform = get_albu_transforms('train')
        test_transform = get_albu_transforms('test')
        post_function = get_albu_transforms(type_="pad" if if_padding else "resize",
                                            output_size=(image_size, image_size))

        post_function_sam = get_albu_transforms(type_="pad" if if_padding else "resize",
                                                output_size=(1024, 1024),
                                                mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])

        train_dataset = HFDataset(
            data_path,
            train_split_name,
            is_padding=if_padding,
            is_resizing=if_resizing,
            output_size=(image_size, image_size),
            common_transforms=train_transform,
            edge_width=7,
            post_transform=post_function,
            post_transform_sam=post_function_sam,
        )

        test_dataset = HFDataset(
            test_data_path,
            test_split_name,
            is_padding=if_padding,
            is_resizing=if_resizing,
            output_size=(image_size, image_size),
            common_transforms=test_transform,
            edge_width=7,
            post_transform=post_function,
            post_transform_sam=post_function_sam,
        )
        if args.rank == 0:  # 只在主进程打印
            print(f'train len {len(train_dataset)}')
            print(f'test len {len(test_dataset)}')
            print(f'num_samples: {args.num_samples}')

        if args.distributed:
            # 分布式训练 - 使用 DistributedSampler
            sampler_train = RandomSubsetSampler(
                train_dataset,
                num_samples=args.num_samples,
                num_replicas=args.world_size,
                rank=args.rank,
                shuffle=True,
                seed=2025
            )
        else:
            # 单卡训练
            sampler_train = RandomSubsetSampler(
                train_dataset,
                num_samples=args.num_samples,
                num_replicas=1,
                rank=0,
                shuffle=True,
                seed=2025
            )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=sampler_train,
            num_workers=args.num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=True if args.num_workers > 0 else False
        )
        if args.rank == 0:  # 只在主进程打印
            print('Load train!')
        if args.distributed:
            # 验证集也使用分布式采样器
            valid_sampler = DistributedSampler(test_dataset, shuffle=False)
        else:
            valid_sampler = None
        valid_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            sampler=valid_sampler,  # 使用采样器
            shuffle=(valid_sampler is None),  # 如果没有采样器才shuffle
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=True if args.num_workers > 0 else False
        )
        if args.rank == 0:  # 只在主进程打印
            print('Load test!')
        if args.rank == 0:  # 只在主进程打印
            print("Using Standard Training with Enhanced DWT Adapters")
        # 标准训练优化器设置
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=1e-5
        )

        train_standard(args, model, device, optimizer, train_loader, valid_loader, save_dir,
                             writer)

        duration = time.time() - start_time
        if args.rank == 0:  # 只在主进程打印
            print('The run time is {}h {}m'.format(int(duration // 3600), int(duration % 3600 // 60)))
            print(f'Save dir:{save_dir}')
    if args.writer:
        writer.close()

    destroy_process_group()
