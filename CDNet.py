# -*- coding: utf-8 -*-
from math import sqrt

import torch
import torch.nn as nn

from AttentionModule import  AdaptiveFeatureResizer,  Attention_block
from MBConv import MBConvBlock
from SelfAttention import ScaledDotProductAttention
from Vit import VisionTransformer, Reconstruct
from pixlevel import PixLevelModule, DAConv


def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()


def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU'):
    layers = []
    layers.append(ConvBatchNorm(in_channels, out_channels, activation))
    for _ in range(nb_Conv - 1):
        layers.append(ConvBatchNorm(out_channels, out_channels, activation))
    return nn.Sequential(*layers)


class ConvBatchNorm(nn.Module):
    """(卷积 => [批量归一化] => ReLU)"""

    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(ConvBatchNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)


class DownBlock(nn.Module):
    """Downscaling with maxpool convolution"""

    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(DownBlock, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x):
        out = self.maxpool(x)
        return self.nConvs(out)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class UpblockAttention(nn.Module):
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.pixModule = PixLevelModule(in_channels // 2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)
        self.att = Attention_block(out_channels * 2, out_channels * 2, out_channels)

    def forward(self, x, skip_x):
        up = self.up(x)
        up = self.att(up, skip_x)
        skip_x_att = self.pixModule(skip_x)

        x = torch.cat([skip_x_att, up], dim=1)  # dim 1 is the channel dimension
        return self.nConvs(x)


class Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.vis = args.vis
        self.n_channels = args.n_channels
        self.n_classes = args.n_labels
        self.in_channels = args.base_channel
        self.image_size = args.img_size
        self.inc = ConvBatchNorm(args.n_channels, self.in_channels)
        self.down1 = DownBlock(self.in_channels, self.in_channels * 2, nb_Conv=2)
        self.down2 = DownBlock(self.in_channels * 2, self.in_channels * 4, nb_Conv=2)
        self.down3 = DownBlock(self.in_channels * 4, self.in_channels * 8, nb_Conv=2)
        self.down4 = DownBlock(self.in_channels * 8, self.in_channels * 8, nb_Conv=2)

        self.downVit = VisionTransformer(args, img_size=224, channel_num=64, patch_size=16, embed_dim=64)
        self.downVit1 = VisionTransformer(args, img_size=112, channel_num=128, patch_size=8, embed_dim=128)
        self.downVit2 = VisionTransformer(args, img_size=56, channel_num=256, patch_size=4, embed_dim=256)
        self.downVit3 = VisionTransformer(args, img_size=28, channel_num=512, patch_size=2, embed_dim=512)

        self.out_chs = args.out_chs
        self.maxpool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpool1d = nn.MaxPool1d(kernel_size=2, stride=2)
        self.maxpool = nn.MaxPool1d(kernel_size=1, stride=1)
        #
        self.s0 = nn.Sequential(
            nn.Conv2d(self.n_channels, self.n_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.n_channels, self.n_channels, kernel_size=3, padding=1)
        )
        self.mlp0 = nn.Sequential(
            nn.Conv2d(self.n_channels, self.out_chs[0], kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(self.out_chs[0], self.out_chs[0], kernel_size=1)
        )

        self.s1 = MBConvBlock(ksize=3, input_filters=self.out_chs[0], output_filters=self.out_chs[0],
                              image_size=self.image_size // 2)
        self.mlp1 = nn.Sequential(
            nn.Conv2d(self.out_chs[0], self.out_chs[1], kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(self.out_chs[1], self.out_chs[1], kernel_size=1)
        )

        self.s2 = MBConvBlock(ksize=3, input_filters=self.out_chs[1], output_filters=self.out_chs[1],
                              image_size=self.image_size // 4)
        self.mlp2 = nn.Sequential(
            nn.Conv2d(self.out_chs[1], self.out_chs[2], kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(self.out_chs[2], self.out_chs[2], kernel_size=1)
        )

        self.s3 = MBConvBlock(ksize=3, input_filters=self.out_chs[2], output_filters=self.out_chs[2],
                              image_size=self.image_size // 8)
        self.mlp3 = nn.Sequential(
            nn.Conv2d(self.out_chs[2], self.out_chs[3], kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(self.out_chs[3], self.out_chs[3], kernel_size=1)
        )

        self.s4 = ScaledDotProductAttention(self.out_chs[2], self.out_chs[2] // 8, self.out_chs[2] // 8, 8)
        self.mlp4 = nn.Sequential(
            nn.Linear(self.out_chs[2], self.out_chs[3]),
            nn.ReLU(),
            nn.Linear(self.out_chs[3], self.out_chs[3])
        )

        self.s5 = ScaledDotProductAttention(self.out_chs[3], self.out_chs[3] // 8, self.out_chs[3] // 8, 8)
        self.mlp5 = nn.Sequential(
            nn.Linear(self.out_chs[3], self.out_chs[4]),
            nn.ReLU(),
            nn.Linear(self.out_chs[4], self.out_chs[4])
        )

        self.fusion_layer1 = DAConv(in_channels1=args.base_channel,in_channels2=args.base_channel, out_channels=args.base_channel, args=args)
        self.fusion_layer2 = DAConv(in_channels1=args.base_channel * 2,in_channels2=args.base_channel * 2, out_channels=args.base_channel * 2, args=args)
        self.fusion_layer3 = DAConv(in_channels1=args.base_channel * 4,in_channels2=args.base_channel * 4, out_channels=args.base_channel * 4, args=args)
        self.fusion_layer4 = DAConv(in_channels1=args.base_channel * 8,in_channels2=args.base_channel * 8, out_channels=args.base_channel * 8, args=args)
        self.fusion_layer5 = DAConv(in_channels1=args.base_channel * 8,in_channels2=args.base_channel * 8, out_channels=args.base_channel * 8, args=args)



    def forward(self, x):
        x = x.float()  # x [32,3,224,224]
        B, C, H, W = x.shape

        # stage1
        x1 = self.inc(x)  # x1 [32, 64, 224, 224]
        z1 = self.mlp0(self.s0(x))  # (1,64,224,224)

        x1 = self.fusion_layer1(x1, z1)
        w = self.maxpool2d(x1)  # (1,64,112,112)

        # stage2
        x2 = self.down1(x1)  # (32,128,112,112)
        z2 = self.mlp1(self.s1(w))  # (1,128,112,112)

        x2 = self.fusion_layer2(x2, z2)
        w = self.maxpool2d(x2)  # (1,128,56,56)

        # stage3
        x3 = self.down2(x2)  # (32,256,56,56)
        z3 = self.mlp2(self.s2(w))  # (1,256,56,56)

        x3 = self.fusion_layer3(x3, z3)
        w = self.maxpool2d(x3)  # (1,256,28,28)

        # stage4
        x4 = self.down3(x3)  # (32,512,28,28)
        z4 = self.mlp3(self.s3(w))  # (1,512,28,28)

        x4 = self.fusion_layer4(x4, z4)
        w = self.maxpool2d(x4)  # (1,512,14,14)

        # stage5
        x5 = self.down4(x4)  # (32,512,14,14)
        w = w.reshape(B, self.out_chs[2], -1).permute(0, 2, 1)  # B,N,C   # (1,768,256)
        w = self.mlp4(self.s4(w, w, w))  # (1,768,512)
        w = self.maxpool1d(w.permute(0, 2, 1)).permute(0, 2, 1)  # (1,392,512)
        w = self.mlp5(self.s5(w, w, w))  # (1,392,512)
        w = self.maxpool(w.permute(0, 2, 1))  # (1,512,196)
        N = w.shape[-1]  # 196
        #
        z5 = w.reshape(B, self.out_chs[4], int(sqrt(N)), int(sqrt(N)))  # (1,512,14,14)

        x5 = self.fusion_layer5(x5, z5)

        # vit
        y1 = self.downVit(x1, x1)  # y1 (32,196,64)
        y2 = self.downVit1(x2, y1)  # (32,196,128)
        y3 = self.downVit2(x3, y2)  # (32,196,256)
        y4 = self.downVit3(x4, y3)  # (32,196,512)

        return y4, y3, y2, y1, x5, x4, x3, x2, x1


class Decoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.classes = args.n_labels
        self.in_channels = args.base_channel
        self.upVit = VisionTransformer(args, img_size=224, channel_num=64, patch_size=16, embed_dim=64)
        self.upVit1 = VisionTransformer(args, img_size=112, channel_num=128, patch_size=8, embed_dim=128)
        self.upVit2 = VisionTransformer(args, img_size=56, channel_num=256, patch_size=4, embed_dim=256)
        self.upVit3 = VisionTransformer(args, img_size=28, channel_num=512, patch_size=2, embed_dim=512)
        self.up4 = UpblockAttention(args.in_channels * 16, args.in_channels * 4, nb_Conv=2)
        self.up3 = UpblockAttention(args.in_channels * 8, args.in_channels * 2, nb_Conv=2)
        self.up2 = UpblockAttention(args.in_channels * 4, args.in_channels, nb_Conv=2)
        self.up1 = UpblockAttention(args.in_channels * 2, args.in_channels // 2, nb_Conv=2)
        self.outc = nn.Conv2d(args.in_channels // 2, args.n_labels, kernel_size=(1, 1), stride=(1, 1))

        self.reconstruct1 = Reconstruct(in_channels=64, out_channels=64, kernel_size=1, scale_factor=(16, 16))
        self.reconstruct2 = Reconstruct(in_channels=128, out_channels=128, kernel_size=1, scale_factor=(8, 8))
        self.reconstruct3 = Reconstruct(in_channels=256, out_channels=256, kernel_size=1, scale_factor=(4, 4))
        self.reconstruct4 = Reconstruct(in_channels=512, out_channels=512, kernel_size=1, scale_factor=(2, 2))
        self.pix_module1 = PixLevelModule(64)
        self.pix_module2 = PixLevelModule(128)
        self.pix_module3 = PixLevelModule(256)
        self.pix_module4 = PixLevelModule(512)
        self.last_activation = nn.Sigmoid()
        self.multi_activation = nn.Softmax()


    def forward(self, y4, y3, y2, y1, x5, x4, x3, x2, x1):
        # x1, x2, x3, x4 = self.DCA([x1, x2, x3, x4])


        y4 = self.upVit3(y4, y4, True)  # (32,196,512)
        y3 = self.upVit2(y3, y4, True)  # (32,196,256)
        y2 = self.upVit1(y2, y3, True)  # (32,196,128)
        y1 = self.upVit(y1, y2, True)  # (32,196,64)


        x1 = self.reconstruct1(y1) + x1  # (32,64,224,224)
        x2 = self.reconstruct2(y2) + x2  # (32,64,112,112)
        x3 = self.reconstruct3(y3) + x3  # (32,64,56,56)
        x4 = self.reconstruct4(y4) + x4  # (32,64,28,28)


        x = self.up4(x5, x4)  # (32,256,28,28)
        x = self.up3(x, x3)  # (32,128,28,28)
        x = self.up2(x, x2)  # (32,64,112,112)
        x = self.up1(x, x1)  # (32,32,224,224)
        logits = self.outc(x)  # (32,2,224,224)
        if self.classes==1:
            return self.last_activation(logits)
        return logits
        # return out




class CDNet(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.in_channels = args.base_channel
        self.batch_size = args.batch_size
        self.Encoder = Encoder(args)
        self.Decoder = Decoder(args)
        self.rfs = AdaptiveFeatureResizer(512, 64, 7)
        self.register_buffer('pre_features', torch.zeros(args.batch_size, args.feature_dim))
        self.register_buffer('pre_weight1', torch.ones(args.batch_size, 1))


    def forward(self, x):
        y4, y3, y2, y1, x5, x4, x3, x2, x1 = self.Encoder(x)

        logits = self.Decoder(y4, y3, y2, y1, x5, x4, x3, x2, x1)

        return logits, self.rfs(x5).reshape(self.batch_size, -1)
        # return logits, x5






