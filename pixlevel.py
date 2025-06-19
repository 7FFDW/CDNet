# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from SimAM import SimAM

'''pixel-level module'''


class PixLevelModule(nn.Module):
    def __init__(self, in_channels):
        super(PixLevelModule, self).__init__()

        # 设定中间层大小的比例
        self.middle_layer_size_ratio = 2

        # 定义一个用于平均池化的卷积层，核大小为1
        self.conv_avg = nn.Conv2d(in_channels, out_channels=in_channels, kernel_size=1, bias=False)

        # 定义一个用于平均池化层的ReLU激活函数
        self.relu_avg = nn.ReLU(inplace=True)

        # 定义一个用于最大池化的卷积层，核大小为1
        self.conv_max = nn.Conv2d(in_channels, out_channels=in_channels, kernel_size=1, bias=False)

        # 定义一个用于最大池化层的ReLU激活函数
        self.relu_max = nn.ReLU(inplace=True)

        # 定义一个瓶颈层，包含两个线性层和一个ReLU激活函数
        self.bottleneck = nn.Sequential(
            nn.Linear(3, 3 * self.middle_layer_size_ratio),  # 乘以中间层大小比例
            nn.ReLU(inplace=True),
            nn.Linear(3 * self.middle_layer_size_ratio, 1)
        )

        # 定义一个包含卷积层和Sigmoid激活函数的序列
        self.conv_sig = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )

    # 定义前向传播函数
    def forward(self, x):
        # 通过平均池化卷积层处理输入
        x_avg = self.conv_avg(x)
        # 应用ReLU激活函数
        x_avg = self.relu_avg(x_avg)
        # 计算各通道的平均值
        x_avg = torch.mean(x_avg, dim=1)
        # 增加一个维度
        x_avg = x_avg.unsqueeze(dim=1)

        # 通过最大池化卷积层处理输入
        x_max = self.conv_max(x)
        # 应用ReLU激活函数
        x_max = self.relu_max(x_max)
        # 计算各通道的最大值
        x_max = torch.max(x_max, dim=1).values
        # 增加一个维度
        x_max = x_max.unsqueeze(dim=1)

        # 将最大值和平均值相加
        x_out = x_max + x_avg
        # 将平均值，最大值和它们的和拼接在一起
        x_output = torch.cat((x_avg, x_max, x_out), dim=1)
        # 转置维度，为了适配线性层的输入格式
        x_output = x_output.transpose(1, 3)
        # 通过瓶颈层处理
        x_output = self.bottleneck(x_output)
        # 再次转置维度
        x_output = x_output.transpose(1, 3)
        # 将处理后的输出与原始输入相乘
        y = x_output * x
        return y


# 通道注意力模块


# 融合特征的卷积模块
# "Diversified Aggregated Convolution" 翻译成中文是“多样化聚合卷积”
# class DAConv(nn.Module):
#     def __init__(self, in_channels, out_channels, args, kernel_size=3, stride=1, padding=1):
#         super(DAConv, self).__init__()
#         self.args = args
#         # 增加两个权重参数
#         self.weight1 = nn.Parameter(torch.ones(1, in_channels // 2, 1, 1))
#         self.weight2 = nn.Parameter(torch.ones(1, in_channels // 2, 1, 1))
#         # torch.nn.init.kaiming_normal_(self.weight1, mode='fan_out', nonlinearity='relu')
#         # torch.nn.init.kaiming_normal_(self.weight2, mode='fan_out', nonlinearity='relu')
#         #
#         # self.bn1 = nn.BatchNorm2d(in_channels // 2)
#         # self.bn2 = nn.BatchNorm2d(in_channels // 2)
#
#         # 标准卷积用于混合融合后的特征
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
#         # ReLU激活函数
#         # self.relu = nn.ReLU()
#         self.sa = SimAM()
#
#         self.epsilon = 1e-5
#
#     def forward(self, feature_map1, feature_map2):
#         # u = uncertain((feature_map1, feature_map2))
#         #
#         # weights1 = 1 / (u[0] + self.epsilon)  # epsilon是一个很小的数，避免除以0
#         # weights2 = 1 / (u[1] + self.epsilon)
#         # sum_weights = weights1 + weights2
#         # norm_weights1 = weights1 / sum_weights
#         # norm_weights2 = weights2 / sum_weights
#
#         # combined = norm_weights1 * feature_map1+norm_weights2 * feature_map2
#         combined = torch.cat((self.weight1 * feature_map1, self.weight2 * feature_map2), dim=1)
#
#         combined = self.sa(combined)
#         # 应用批量归一化
#         # combined = self.bn(combined)
#
#         # 通过卷积层混合特征
#         combined = self.conv(combined)
#
#         # 应用激活函数
#         # combined = self.relu(combined)
#
#         return combined
class DAConv(nn.Module):
    def __init__(self, in_channels1, in_channels2, out_channels, args, kernel_size=3, stride=1, padding=1):
        super(DAConv, self).__init__()
        self.args = args
        # 增加两个权重参数
        self.weight1 = nn.Parameter(torch.ones(1, in_channels1, 1, 1))
        self.weight2 = nn.Parameter(torch.ones(1, in_channels2, 1, 1))

        # 标准卷积用于混合融合后的特征
        self.conv = nn.Conv2d(in_channels1 + in_channels2, out_channels, kernel_size, stride, padding)
        self.sa = SimAM()
        self.epsilon = 1e-5

    def forward(self, feature_map1, feature_map2):
        # 调整 feature_map1 和 feature_map2 的空间尺寸使其一致
        if feature_map1.size(2) != feature_map2.size(2) or feature_map1.size(3) != feature_map2.size(3):
            if feature_map1.size(2) > feature_map2.size(2):
                feature_map2 = F.interpolate(feature_map2, size=(feature_map1.size(2), feature_map1.size(3)), mode='bilinear', align_corners=True)
            else:
                feature_map1 = F.interpolate(feature_map1, size=(feature_map2.size(2), feature_map2.size(3)), mode='bilinear', align_corners=True)

        # 合并特征图
        combined = torch.cat((self.weight1 * feature_map1, self.weight2 * feature_map2), dim=1)
        combined = self.sa(combined)
        combined = self.conv(combined)

        return combined

def uncertain(alpha1):
    """
    :param alpha1: Dirichlet distribution parameters of view 1
    :param alpha2: Dirichlet distribution parameters of view 2
    :return: uncertainty of every view
    """
    alpha = dict()
    alpha[0], alpha[1] = alpha1[0], alpha1[1]
    b, S, E, u = dict(), dict(), dict(), dict()
    classes = 2
    for v in range(2):
        S[v] = torch.sum(alpha[v], dim=1, keepdim=True)
        # E[v] = alpha[v]-1
        # b[v] = E[v]/(S[v].expand(E[v].shape))
        u[v] = classes / S[v]

    return u

class CCA(nn.Module):
    """
    CCA Block
    """
    def __init__(self, F_g, F_x):
        super().__init__()
        self.mlp_x = nn.Sequential(
            Flatten(),
            nn.Linear(F_x, F_x))
        self.mlp_g = nn.Sequential(
            Flatten(),
            nn.Linear(F_g, F_x))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # channel-wise attention
        avg_pool_x = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        channel_att_x = self.mlp_x(avg_pool_x)
        avg_pool_g = F.avg_pool2d( g, (g.size(2), g.size(3)), stride=(g.size(2), g.size(3)))
        channel_att_g = self.mlp_g(avg_pool_g)
        channel_att_sum = (channel_att_x + channel_att_g)/2.0
        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        x_after_channel = x * scale
        out = self.relu(x_after_channel)
        return out

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAMLayer, self).__init__()

        # channel attention
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # shared MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )

        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x

        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x

class Fusion(nn.Module):
    def __init__(self, channels, reduction):
        super(Fusion, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()
        self.out_conv = nn.Conv2d(channels, reduction, kernel_size=1)

    def forward(self, x):
        avg_pool = self.avg_pool(x)
        max_pool = self.max_pool(x)
        out = self.fc1(avg_pool + max_pool)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        x = self.out_conv(x * out)
        return x

if __name__ == '__main__':
    model = DAConv(2, 1)
    print(sum([l.nelement() for l in model.parameters()]))
