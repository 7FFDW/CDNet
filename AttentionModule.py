import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Softmax


class ChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Conv1d(in_channels, in_channels // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv1d(in_channels // 16, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv1d(2, 1, 7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_channels):
        super(CBAM, self).__init__()
        self.channelAttention = ChannelAttention(in_channels)
        self.spatialAttention = SpatialAttention()

    def forward(self, x):
        x = self.channelAttention(x) * x
        x = self.spatialAttention(x) * x
        return x


class OSAM(nn.Module):
    def __init__(self, channel):
        super(OSAM, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(channel, channel, 3, 1, 1),
            nn.BatchNorm1d(channel),
            nn.ReLU(),
            CBAM(channel),
            nn.Conv1d(channel, channel, 3, 1, 1),
            nn.BatchNorm1d(channel),
            nn.ReLU(),
            nn.Conv1d(channel, channel, 3, 1, 1),
            nn.BatchNorm1d(channel),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.features(x)
        return x


class ACmix(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_att=7, head=4, kernel_conv=3, stride=1, dilation=1):
        super(ACmix, self).__init__()
        # 初始化基本参数
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.head = head
        self.kernel_att = kernel_att
        self.kernel_conv = kernel_conv
        self.stride = stride
        self.dilation = dilation
        # 初始化可学习的比率参数
        self.rate1 = torch.nn.Parameter(torch.Tensor(1))
        self.rate2 = torch.nn.Parameter(torch.Tensor(1))
        self.head_dim = self.out_planes // self.head

        # 初始化卷积层
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv3 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv_p = nn.Conv2d(2, self.head_dim, kernel_size=1)

        # 初始化注意力机制所需的参数
        self.padding_att = (self.dilation * (self.kernel_att - 1) + 1) // 2
        self.pad_att = torch.nn.ReflectionPad2d(self.padding_att)
        self.unfold = nn.Unfold(kernel_size=self.kernel_att, padding=0, stride=self.stride)
        self.softmax = torch.nn.Softmax(dim=1)

        # 初始化用于深度卷积的全连接层
        self.fc = nn.Conv2d(3 * self.head, self.kernel_conv * self.kernel_conv, kernel_size=1, bias=False)
        self.dep_conv = nn.Conv2d(self.kernel_conv * self.kernel_conv * self.head_dim, out_planes,
                                  kernel_size=self.kernel_conv, bias=True, groups=self.head_dim, padding=1,
                                  stride=stride)

        self.reset_parameters()

    def reset_parameters(self):
        # 参数初始化函数
        self.init_rate_half(self.rate1)
        self.init_rate_half(self.rate2)
        # 初始化深度卷积的权重
        kernel = torch.zeros(self.kernel_conv * self.kernel_conv, self.kernel_conv, self.kernel_conv)
        for i in range(self.kernel_conv * self.kernel_conv):
            kernel[i, i // self.kernel_conv, i % self.kernel_conv] = 1.
        kernel = kernel.squeeze(0).repeat(self.out_planes, 1, 1, 1)
        self.dep_conv.weight = nn.Parameter(data=kernel, requires_grad=True)
        self.dep_conv.bias = self.init_rate_0(self.dep_conv.bias)

    def forward(self, x):
        # 前向传播函数
        q, k, v = self.conv1(x), self.conv2(x), self.conv3(x)
        scaling = float(self.head_dim) ** -0.5
        b, c, h, w = q.shape
        h_out, w_out = h // self.stride, w // self.stride

        pe = self.conv_p(self.position(h, w, x.is_cuda))

        q_att = q.reshape(b * self.head, self.head_dim, h, w) * scaling
        k_att = k.reshape(b * self.head, self.head_dim, h, w)
        v_att = v.reshape(b * self.head, self.head_dim, h, w)

        # 降低分辨率以适应步幅
        if self.stride > 1:
            q_att = self.reduce_resolution(q_att, self.stride)
            q_pe = self.reduce_resolution(pe, self.stride)
        else:
            q_pe = pe

        # 展开键向量并计算注意力
        unfold_k = self.unfold(self.pad_att(k_att)).view(b * self.head, self.head_dim,
                                                         self.kernel_att * self.kernel_att, h_out,
                                                         w_out)
        unfold_rpe = self.unfold(self.pad_att(pe)).view(1, self.head_dim, self.kernel_att * self.kernel_att, h_out,
                                                        w_out)

        att = (q_att.unsqueeze(2) * (unfold_k + q_pe.unsqueeze(2) - unfold_rpe)).sum(
            1)
        att = self.softmax(att)

        out_att = self.unfold(self.pad_att(v_att)).view(b * self.head, self.head_dim, self.kernel_att * self.kernel_att,
                                                        h_out, w_out)
        out_att = (att.unsqueeze(1) * out_att).sum(2).view(b, self.out_planes, h_out, w_out)

        # 计算并应用卷积滤波器
        f_all = self.fc(torch.cat(
            [q.view(b, self.head, self.head_dim, h * w), k.view(b, self.head, self.head_dim, h * w),
             v.view(b, self.head, self.head_dim, h * w)], 1))
        f_conv = f_all.permute(0, 2, 1, 3).reshape(x.shape[0], -1, x.shape[-2], x.shape[-1])

        out_conv = self.dep_conv(f_conv)

        # 结合注意力和卷积输出
        return self.rate1 * out_att + self.rate2 * out_conv

    @staticmethod
    def position(H, W, is_cuda=True):
        # 生成位置编码
        if is_cuda:
            loc_w = torch.linspace(-1.0, 1.0, W).cuda().unsqueeze(0).repeat(H, 1)
            loc_h = torch.linspace(-1.0, 1.0, H).cuda().unsqueeze(1).repeat(1, W)
        else:
            loc_w = torch.linspace(-1.0, 1.0, W).unsqueeze(0).repeat(H, 1)
            loc_h = torch.linspace(-1.0, 1.0, H).unsqueeze(1).repeat(1, W)
        loc = torch.cat([loc_w.unsqueeze(0), loc_h.unsqueeze(0)], 0).unsqueeze(0)
        return loc

    @staticmethod
    def reduce_resolution(x, stride):
        # 降低图像分辨率
        return x[:, :, ::stride, ::stride]

    @staticmethod
    def init_rate_half(tensor):
        # 初始化为0.5
        if tensor is not None:
            tensor.data.fill_(0.5)

    @staticmethod
    def init_rate_0(tensor):
        # 初始化为0
        if tensor is not None:
            tensor.data.fill_(0.)



class CrissCrossAttention(nn.Module):
    """ Criss-Cross 注意力模块"""

    def __init__(self, args, in_dim):
        super(CrissCrossAttention, self).__init__()
        # 使用1x1卷积生成查询向量
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        # 使用1x1卷积生成键向量
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        # 使用1x1卷积生成值向量
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        # Softmax层用于归一化注意力得分
        self.softmax = Softmax(dim=3)
        # 可学习的参数，调节注意力的影响
        self.gamma = nn.Parameter(torch.zeros(1))
        self.args = args  # 外部参数

    def forward(self, x):
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)  # 生成查询表示
        # 调整查询向量的维度
        proj_query_H = proj_query.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height).permute(0, 2, 1)
        proj_query_W = proj_query.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width).permute(0, 2, 1)
        proj_key = self.key_conv(x)  # 生成键表示
        # 调整键向量的维度
        proj_key_H = proj_key.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_key_W = proj_key.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        proj_value = self.value_conv(x)  # 生成值表示
        # 调整值向量的维度
        proj_value_H = proj_value.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_value_W = proj_value.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        # 计算注意力得分
        energy_H = (torch.bmm(proj_query_H, proj_key_H) + self.INF(m_batchsize, height, width).to(self.args.device)).view(m_batchsize, width, height, height).permute(0, 2, 1, 3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize, height, width, width)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        # 应用注意力得分
        att_H = concate[:, :, :, 0:height].permute(0, 2, 1, 3).contiguous().view(m_batchsize * width, height, height)
        att_W = concate[:, :, :, height:height + width].contiguous().view(m_batchsize * height, width, width)
        # 计算输出
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize, width, -1, height).permute(0, 2, 3, 1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize, height, -1, width).permute(0, 2, 1, 3)
        return self.gamma * (out_H + out_W) + x  # 输出结果

    @staticmethod
    def INF(B, H, W):
        # 生成一个特定结构的矩阵，用于注意力计算中
        return -torch.diag(torch.tensor(float("inf")).repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1).to()


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi



class AttentionLayer(nn.Module):

    def __init__(self, feature_dim):
        super(AttentionLayer, self).__init__()
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)

    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        attention_scores = torch.bmm(query, key.transpose(1, 2))
        attention_scores = F.softmax(attention_scores, dim=-1)
        weighted_values = torch.bmm(attention_scores, value)

        return weighted_values


class AdaptiveFeatureResizer(nn.Module):
    def __init__(self, input_dim, output_dim, output_size):
        super(AdaptiveFeatureResizer, self).__init__()
        self.attention_layer = AttentionLayer(input_dim)
        self.conv1 = nn.Conv2d(input_dim, output_dim, kernel_size=1)  # 1x1 卷积以改变通道数
        self.conv2 = nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=2, padding=1)  # 跨步卷积减小尺寸
        self.bn = nn.BatchNorm2d(output_dim)
        self.relu = nn.ReLU()
        self.output_size = output_size

    def forward(self, x):
        # 调整x的维度以适应注意力层: (batch_size, channels, height, width) -> (batch_size, height*width, channels)
        batch_size, channels, height, width = x.shape
        x = x.reshape(batch_size, channels, height * width).transpose(1, 2)

        # 应用注意力机制
        x = self.attention_layer(x)

        # 转换回原来的维度并应用卷积层
        x = x.transpose(1, 2).reshape(batch_size, channels, height, width)
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv2(x)

        # 可选：调整输出尺寸
        if x.size(-1) != self.output_size:
            x = F.adaptive_avg_pool2d(x, (self.output_size, self.output_size))

        return x


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class SelfAttention(nn.Module):
    def __init__(self, num_attention_heads, input_size, hidden_size, hidden_dropout_prob):
        super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = hidden_size

        self.query = nn.Linear(input_size, self.all_head_size)
        self.key = nn.Linear(input_size, self.all_head_size)
        self.value = nn.Linear(input_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(0.5)

        # 做完self-attention 做一个前馈全连接 LayerNorm 输出
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]

        # attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # Fixme
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


if __name__ == '__main__':
    # 假设的输入尺寸
    batch_size = 32  # 批量大小
    seq_length = 196  # 序列长度
    input_size = 256  # 输入特征维度

    # 创建随机输入数据
    input_tensor = torch.rand(32, 512, 14, 14)

    # 实例化SelfAttention类
    # 参数：注意力头数、输入尺寸、隐藏层大小、隐藏层dropout概率
    num_attention_heads = 4
    hidden_size = 256
    hidden_dropout_prob = 0.1
    self_attention = SelfAttention(num_attention_heads, 14, 512, hidden_dropout_prob)

    # 通过SelfAttention模块传递数据
    output = self_attention(input_tensor)

    # 打印输出
    print(output)
    print("Output shape:", output.shape)


class SimAM(torch.nn.Module):
    def __init__(self, e_lambda=1e-4):
        super(SimAM, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, c, h, w = x.size()

        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)
