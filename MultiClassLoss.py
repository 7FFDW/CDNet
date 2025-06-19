import numpy as np
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

from config import get_config


class DiceLoss(nn.Module):
    def __init__(self, n_classes, reduce=True):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes
        self.reduce = reduce


    def forward(self, inputs, target):
        target = self._one_hot_encoder(target)
        # print(np.unique(target.cpu().detach().numpy()))
        inputs = torch.softmax(inputs, dim=1) if inputs.shape[1] > 1 else torch.sigmoid(inputs)
        intersection = torch.sum(inputs * target, dim=(2, 3))
        union = torch.sum(inputs + target, dim=(2, 3))
        dice_score = (2. * intersection + 1e-5) / (union + 1e-5)
        loss = 1 - dice_score
        return loss.mean()

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        # print(np.unique(input_tensor.cpu().detach().numpy()))
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        # print(np.unique(output_tensor.cpu().detach().numpy()))
        return output_tensor.float()

class Block_DiceLoss(nn.Module):
    def __init__(self, n_classes, block_num):
        super(Block_DiceLoss, self).__init__()
        self.n_classes = n_classes
        self.block_num = block_num
        self.dice_loss = DiceLoss(n_classes, reduce=False)  # Individual losses for flexibility

    def forward(self, inputs, target, softmax=False):
        shape = inputs.shape
        img_size = shape[-1]
        block_size = math.ceil(img_size / self.block_num)
        block_losses = []
        # print(np.unique(target.cpu().detach().numpy()))
        for i in range(self.block_num):
            for j in range(self.block_num):
                h_start, h_end = i * block_size, min((i + 1) * block_size, shape[2])
                w_start, w_end = j * block_size, min((j + 1) * block_size, shape[3])
                block_features = inputs[:, :, h_start:h_end, w_start:w_end]
                block_labels = target[:, h_start:h_end, w_start:w_end]
                # print(np.unique(block_labels.cpu().detach().numpy()))
                block_losses.append(self.dice_loss(block_features, block_labels))

        loss_per_block = torch.stack(block_losses)
        return loss_per_block.mean()  # Overall mean loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2.0,n_classes=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def forward(self, inputs, targets):
        targets = self._one_hot_encoder(targets)
        # 计算交叉熵损失，但不进行平均或求和，保留每个样本的损失
        CE_loss = F.cross_entropy(inputs, targets)

        # 计算每个样本的pt值，这是模型对每个真实类别的估计概率
        pt = torch.exp(-CE_loss)

        # 根据Focal Loss的定义计算损失
        F_loss = self.alpha * ((1 - pt) ** self.gamma) * CE_loss

        # 根据reduce标志，返回损失的平均值或保留其原样
        return F_loss  # 返回每个样本的损失向量

class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, weights=None, reduce=False, Num_classes=1):
        super(WeightedCrossEntropyLoss, self).__init__()
        # 如果未指定权重，则默认为均匀权重

        self.reduce = reduce

    def forward(self, logit_pixel, truth_pixel):

        # 计算交叉熵损失
        criterion = nn.CrossEntropyLoss(reduction='none')
        loss = criterion(logit_pixel, truth_pixel)

        # 求和并平均
        if not self.reduce:
            return loss.mean()
        return loss.mean(dim=[1, 2])

class MultiClassLoss(nn.Module):
    def __init__(self, args, weights=None, reduce=False,only_ce = False):
        super(MultiClassLoss, self).__init__()
        if weights is None:
            weights = {'dice': args.comdice, 'ce': args.comce, 'focal': args.comfl}
        self.weights = weights

        self.dice_loss = Block_DiceLoss(n_classes=args.n_labels, block_num=4)
        self.focal_loss = FocalLoss(n_classes=args.n_labels)
        self.ce_loss = WeightedCrossEntropyLoss(Num_classes=args.n_labels, reduce=reduce)

        self.dice_weight = nn.Parameter(torch.tensor(0.5, requires_grad=True))
        self.focal_weight = nn.Parameter(torch.tensor(0.5, requires_grad=True))


        self.only_ce = only_ce
    def forward(self, inputs, targets, sampleWeight=None,softmax=True):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)


        if self.only_ce:
            return self.ce_loss(inputs, targets)
        # loss = self.weights['dice'] * self.dice_loss(inputs, targets)
        weights_sum = self.dice_weight + self.focal_weight
        dice_weight = self.dice_weight / weights_sum
        focal_weight = self.focal_weight / weights_sum
        # print(np.unique(targets.cpu().numpy()))
        loss = dice_weight * self.dice_loss(inputs, targets)
        loss += focal_weight * self.focal_loss(inputs, targets)

        # loss += self.weights['focal'] * self.focal_loss(inputs, targets)


        if sampleWeight != None:
            loss += self.weights['ce'] * (self.ce_loss(inputs, targets).view(1, -1).mm(sampleWeight).view(1)).mean()
        else:
            loss += self.weights['ce'] * (self.ce_loss(inputs, targets)).mean()

        return loss

    def _show_dice(self, logits, targets):


        # 将预测转换为类别索引
        inputs = torch.argmax(logits, dim=1)
        num_classes = logits.shape[1]

        dice_list = []
        for _ in range(0, num_classes):
            dice_list.append(self.dice_coefficient(inputs, targets, class_index=_))



        return dice_list



