# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.models.layers import trunc_normal_, DropPath
from .utils import LayerNorm   # 确保 from .utils import LayerNorm 可用


# ========== MLCA 模块 (local_size=5, local_weight=0.5) ==========
class MLCA(nn.Module):
    def __init__(self, in_size, local_size=3, gamma=2, b=1, local_weight=0.3):
        super(MLCA, self).__init__()
        self.local_size = local_size
        self.gamma = gamma
        self.b = b
        t = int(abs(math.log(in_size, 2) + self.b) / self.gamma)
        k = t if t % 2 else t + 1
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.conv_local = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.local_weight = local_weight
        self.local_arv_pool = nn.AdaptiveAvgPool2d(local_size)
        self.global_arv_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        local_arv = self.local_arv_pool(x)
        global_arv = self.global_arv_pool(local_arv)
        b, c, m, n = x.shape
        b_local, c_local, m_local, n_local = local_arv.shape

        temp_local = local_arv.view(b, c_local, -1).transpose(-1, -2).reshape(b, 1, -1)
        temp_global = global_arv.view(b, c, -1).transpose(-1, -2)

        y_local = self.conv_local(temp_local)
        y_global = self.conv(temp_global)

        y_local_transpose = y_local.reshape(b, self.local_size * self.local_size, c).transpose(-1, -2).view(b, c, self.local_size, self.local_size)
        y_global_transpose = y_global.view(b, -1).unsqueeze(-1).unsqueeze(-1)

        att_local = y_local_transpose.sigmoid()
        att_global = F.adaptive_avg_pool2d(y_global_transpose.sigmoid(), [self.local_size, self.local_size])
        att_all = F.adaptive_avg_pool2d(att_global * (1 - self.local_weight) + (att_local * self.local_weight), [m, n])
        return x * att_all


# ========== MDConv 辅助函数和模块 ==========
def _SplitChannels(channels, num_groups):
    split_channels = [channels // num_groups for _ in range(num_groups)]
    split_channels[0] += channels - sum(split_channels)
    return split_channels


class MDConv(nn.Module):
    """ Mixed Depthwise Convolution (多尺度深度卷积) """
    def __init__(self, channels, kernel_size, stride):
        super(MDConv, self).__init__()
        self.num_groups = len(kernel_size)
        self.split_channels = _SplitChannels(channels, self.num_groups)

        self.mixed_depthwise_conv = nn.ModuleList()
        for i in range(self.num_groups):
            self.mixed_depthwise_conv.append(
                nn.Conv2d(
                    self.split_channels[i],
                    self.split_channels[i],
                    kernel_size[i],
                    stride=stride,
                    padding=kernel_size[i] // 2,
                    groups=self.split_channels[i],
                    bias=False
                )
            )

    def forward(self, x):
        if self.num_groups == 1:
            return self.mixed_depthwise_conv[0](x)
        x_split = torch.split(x, self.split_channels, dim=1)
        x = [conv(t) for conv, t in zip(self.mixed_depthwise_conv, x_split)]
        x = torch.cat(x, dim=1)
        return x


# ========== 改进的 Block（并行集成 MLCA + MDConv） ==========
class Block(nn.Module):
    """ ConvNeXtV2 Block 双模块增强版
    - 主路径：dwconv (9x9) → MLCA → LayerNorm → Linear(4x) → GELU → Linear(1x)
    - 并行分支：MDConv (多尺度 depthwise conv)
    - 两分支通过可学习权重融合，再残差连接
    """
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        # 主路径（含 MLCA）
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=9, padding=4, groups=dim)
        self.mlca = MLCA(in_size=dim, local_size=5, local_weight=0.5)   # 使用更强的 MLCA
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

        # 并行 MDConv 分支
        self.mdconv = MDConv(channels=dim, kernel_size=[3, 5, 7, 9], stride=1)
        self.norm_md = LayerNorm(dim, eps=1e-6, data_format="channels_first")

        # 可学习的融合权重
        self.fusion_alpha = nn.Parameter(torch.tensor(0.5))
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x  # (N, C, H, W)

        # ----- 主路径（dwconv + MLCA + MLP）-----
        x_main = self.dwconv(x)            # (N, C, H, W)
        x_main = self.mlca(x_main)         # MLCA 增强
        x_main = x_main.permute(0, 2, 3, 1)  # (N, H, W, C)
        x_main = self.norm(x_main)
        x_main = self.pwconv1(x_main)
        x_main = self.act(x_main)
        x_main = self.pwconv2(x_main)      # (N, H, W, C)
        x_main = x_main.permute(0, 3, 1, 2)  # (N, C, H, W)

        # ----- MDConv 分支 -----
        x_md = self.mdconv(x)              # (N, C, H, W)
        x_md = self.norm_md(x_md)          # 稳定训练

        # ----- 融合 -----
        x_out = x_main + self.fusion_alpha * x_md

        # 残差连接
        x = input + self.drop_path(x_out)
        return x


# ========== ConvNeXt V2 主体（与标准结构一致，仅 Block 被替换） ==========
class ConvNeXtV2(nn.Module):
    def __init__(self, in_chans=3, num_classes=1000,
                 depths=[3, 3, 27, 2], dims=[96, 192, 384, 768],
                 drop_path_rate=0., head_init_scale=1.):
        super().__init__()
        self.depths = depths
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1]))

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


# ========== 模型构造函数（所有尺寸，depths 和 dims 按改进版调整） ==========
def convnextv2_atto(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[40, 80, 160, 320], **kwargs)
    return model

def convnextv2_femto(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[48, 96, 192, 384], **kwargs)
    return model

def convnext_pico(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[64, 128, 256, 512], **kwargs)
    return model

def convnextv2_nano(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 8, 2], dims=[80, 160, 320, 640], **kwargs)
    return model

def convnextv2_tiny(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 27, 2], dims=[96, 192, 384, 768], **kwargs)
    return model

def convnextv2_base(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 27, 2], dims=[128, 256, 512, 1024], **kwargs)
    return model

def convnextv2_large(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 27, 2], dims=[192, 384, 768, 1536], **kwargs)
    return model

def convnextv2_huge(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 27, 2], dims=[352, 704, 1408, 2816], **kwargs)
    return model