# -*- coding: utf-8 -*-

import torch

from timm.models.layers import DropPath
from torch.nn import Dropout, Conv2d
from torch.nn.modules.utils import _pair

import torch.nn as nn
import numpy as np


class Reconstruct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(Reconstruct, self).__init__()
        # 根据卷积核大小设置填充。
        if kernel_size == 3:
            padding = 1
        else:
            padding = 0

        # 定义一个具有指定参数的卷积层。
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        # 为卷积层的输出定义批量归一化。
        self.norm = nn.BatchNorm2d(out_channels)
        # 定义ReLU激活函数，使用就地操作以节省内存。
        self.activation = nn.ReLU(inplace=True)
        # 存储上采样的比例因子。
        self.scale_factor = scale_factor

    def forward(self, x):
        # 如果输入为None，返回None。这在网络的条件执行中有用。
        if x is None:
            return None

        # 提取批次大小、补丁数和隐藏层维度。
        B, n_patch, hidden = x.size()
        # 假设补丁以方形网格排列，计算高度和宽度。
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        # 变换和重塑输入张量为卷积层预期的格式。
        x = x.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        # 对重塑后的输入张量应用上采样。
        x = nn.functional.upsample(x, scale_factor=self.scale_factor, mode='nearest')

        # 应用卷积操作，接着是批量归一化和ReLU激活。
        out = self.conv(x)
        out = self.norm(out)
        out = self.activation(out)
        return out


class Embeddings(nn.Module):
    # Embeddings类，继承自PyTorch的nn.Module，用于构建图像的补丁和位置嵌入

    def __init__(self, config, patch_size, img_size, in_channels):
        # 初始化函数，接收配置参数、补丁尺寸、图像尺寸和输入通道数
        super().__init__()
        # 调用父类的初始化方法
        img_size = _pair(img_size)
        # 将图像尺寸转换为二元组
        patch_size = _pair(patch_size)
        # 将补丁尺寸转换为二元组
        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        # 计算图像中的补丁总数
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=in_channels,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        # 定义卷积层以创建补丁嵌入，卷积核和步长均为补丁尺寸
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, in_channels))
        # 初始化位置嵌入，并设置为可训练参数
        self.dropout = Dropout(0.1)
        # 定义dropout层以减少过拟合

    def forward(self, x):
        # 前向传播函数
        if x is None:
            return None
        # 如果输入为None，则直接返回None
        x = self.patch_embeddings(x)
        # 通过卷积层得到补丁嵌入
        x = x.flatten(2)
        # 将补丁嵌入的维度压平
        x = x.transpose(-1, -2)
        # 转置补丁嵌入，以匹配位置嵌入的维度
        embeddings = x + self.position_embeddings
        # 将补丁嵌入和位置嵌入相加
        embeddings = self.dropout(embeddings)
        # 应用dropout
        return embeddings
        # 返回嵌入结果


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=None, out_dim=None):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act_layer = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = Dropout(0.1)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)  # [B, num_patches, hidden_dim]
        x = self.act_layer(x)
        x = self.dropout(x)
        x = self.fc2(x)  # [B, num_patches, out_dim]
        x = self.act_layer(x)
        x = self.dropout(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x)  # [B, num_patches, 3*embed_dim]
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)  # [B, num_patches, 3, num_heads, per_HeadDim]
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, num_patches, per_HeadDim]
        q, k, v = qkv[0], qkv[1], qkv[
            2]  # [B, num_heads, num_patches, per_HeadDim] [4, 8, 196, 8/16/32/64] easy to use tensor

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, num_patches, num_patches]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v)  # [B, num_heads, num_patches, per_HeadDim]
        x = x.transpose(1, 2)  # [B, num_patches, num_heads, per_HeadDim]
        x = x.reshape(B, N, C)  # [B, num_patches, embed_dim]
        x = self.proj(x)  # [B, num_patches, embed_dim]
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_dim=dim, hidden_dim=self.mlp_hidden_dim, out_dim=dim)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class ConvTransBN(nn.Module):  # (convolution => [BN] => ReLU)
    def __init__(self, in_channels, out_channels):
        super(ConvTransBN, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)


class VisionTransformer(nn.Module):  # Transformer-branch
    def __init__(self, args, img_size, channel_num, patch_size, embed_dim, depth=1, num_heads=8,
                 mlp_ratio=4., qkv_bias=True, num_classes=1, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super(VisionTransformer, self).__init__()
        self.config = args
        self.vis = args.vis
        self.embeddings = Embeddings(config=args, patch_size=patch_size, img_size=img_size, in_channels=channel_num)
        self.depth = depth
        self.dim = embed_dim
        norm_layer = nn.LayerNorm
        self.norm = norm_layer(embed_dim)
        act_layer = nn.GELU

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.Encoder_blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                  attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(self.depth)])

        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.CTBN = ConvTransBN(in_channels=embed_dim, out_channels=embed_dim // 2)
        self.CTBN2 = ConvTransBN(in_channels=embed_dim * 2, out_channels=embed_dim)

    def forward(self, x, skip_x, reconstruct=False):
        if not reconstruct:  # x(8,64,224,224)
            x = self.embeddings(x)  # x(8,196,64) x(8,196,128)
            x = self.Encoder_blocks(x)  # x(8,196,64) x(8,196,128)
        else:
            x = self.Encoder_blocks(x)
        if (self.dim == 64 and not reconstruct) or (self.dim == 512 and reconstruct):
            return x
        elif not reconstruct:
            x = x.transpose(1, 2)  # [B, embed_dim, num_patches] (8,128,196)
            x = self.CTBN(x)  # [B, embed_dim//2, num_patches] (8,64,196)
            x = x.transpose(1, 2)  # [B, num_patches, embed_dim//2]
            y = torch.cat([x, skip_x], dim=2)  # [B, num_patches, embed_dim] (8,196,128)
            return y
        elif reconstruct:
            skip_x = skip_x.transpose(1, 2)
            skip_x = self.CTBN2(skip_x)
            skip_x = skip_x.transpose(1, 2)

            y = x + skip_x
            return y
