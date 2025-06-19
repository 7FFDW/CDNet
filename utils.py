import logging
import math
import os
import random
import uuid
import warnings
import weakref
from functools import wraps

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import jaccard_score
from torch import nn
from torch.autograd import Variable
from torch.nn.utils import prune
from torch.optim import Optimizer
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler




def set_seed(seed):
    # 为CPU设置种子用于生成随机数，以使得结果是确定的
    torch.manual_seed(seed)

    # 如果程序使用了GPU，还应该设置CUDA的随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 如果使用多个GPU

    # 设置Python自带的随机库的随机种子
    random.seed(seed)

    # 设置NumPy的随机种子
    np.random.seed(seed)

    # 确保PyTorch的随机性操作在不同运行间有确定的结果
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False




def pad_if_too_small(image, new_shape, pad_value=None):
    """Padding a image according to the new shape.

    The result shape will be [max(image[0], new_shape[0]), max(image[1], new_shape[1])].
    e.g.,
    1. image:[10,20], new_shape:(30,30), the res shape is [30,30].
    2. image:[10,20], new_shape:(10,10), the res shape is [10,20].
    3. image:[3,10,20], new_shape:(3,20,20), the res shape is [3,20,20].

    Args:
      image: a numpy array.
      new_shape: a tuple, # elements should be the same as the image.
      pad_value: padding value, default to 0.

    Returns:
      res: a numpy array.
    """
    shape = tuple(list(image.shape))
    new_shape = tuple(np.max(np.concatenate((shape, new_shape)).reshape((2, len(shape))), axis=0))
    if pad_value is None:
        if len(shape) == 2:
            pad_value = image[0, 0]
        elif len(shape) == 3:
            pad_value = image[0, 0, 0]
        else:
            raise ValueError("Image must be either 2 or 3 dimensional")
    res = np.ones(list(new_shape), dtype=image.dtype) * pad_value
    start = np.array(new_shape) / 2. - np.array(shape) / 2.
    if len(shape) == 2:
        res[int(start[0]):int(start[0]) + int(shape[0]), int(start[1]):int(start[1]) + int(shape[1])] = image
    elif len(shape) == 3:
        res[int(start[0]):int(start[0]) + int(shape[0]), int(start[1]):int(start[1]) + int(shape[1]),
        int(start[2]):int(start[2]) + int(shape[2])] = image
    return res


def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    # img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg)
    else:
        # use this function if image is grayscale
        plt.imshow(npimg[0, :, :], 'gray')
        # use this function if image is RGB
        # plt.imshow(np.transpose(npimg, (1, 2, 0)))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CosineAnnealingWarmRestarts(_LRScheduler):
    r"""Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr, :math:`T_{cur}`
    is the number of epochs since the last restart and :math:`T_{i}` is the number
    of epochs between two warm restarts in SGDR:

    .. math::
        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 +
        \cos\left(\frac{T_{cur}}{T_{i}}\pi\right)\right)

    When :math:`T_{cur}=T_{i}`, set :math:`\eta_t = \eta_{min}`.
    When :math:`T_{cur}=0` after restart, set :math:`\eta_t=\eta_{max}`.

    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_0 (int): Number of iterations for the first restart.
        T_mult (int, optional): A factor increases :math:`T_{i}` after a restart. Default: 1.
        eta_min (float, optional): Minimum learning rate. Default: 0.
        last_epoch (int, optional): The index of last epoch. Default: -1.

    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min

        super(CosineAnnealingWarmRestarts, self).__init__(optimizer, last_epoch)

        self.T_cur = self.last_epoch

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", DeprecationWarning)

        return [self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
                for base_lr in self.base_lrs]

    def step(self, epoch=None):
        """Step could be called after every batch update

        Example:
            >>> scheduler = CosineAnnealingWarmRestarts(optimizer, T_0, T_mult)
            >>> iters = len(dataloader)
            >>> for epoch in range(20):
            >>>     for i, sample in enumerate(dataloader):
            >>>         inputs, labels = sample['inputs'], sample['labels']
            >>>         scheduler.step(epoch + i / iters)
            >>>         optimizer.zero_grad()
            >>>         outputs = net(inputs)
            >>>         loss = criterion(outputs, labels)
            >>>         loss.backward()
            >>>         optimizer.step()

        This function can be called in an interleaved way.

        Example:
            >>> scheduler = CosineAnnealingWarmRestarts(optimizer, T_0, T_mult)
            >>> for epoch in range(20):
            >>>     scheduler.step()
            >>> scheduler.step(26)
            >>> scheduler.step() # scheduler.step(27), instead of scheduler(20)
        """

        if epoch is None and self.last_epoch < 0:
            epoch = 0

        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.T_cur = self.T_cur - self.T_i
                self.T_i = self.T_i * self.T_mult
        else:
            if epoch < 0:
                raise ValueError("Expected non-negative epoch, but got {}".format(epoch))
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
        self.last_epoch = math.floor(epoch)

        class _enable_get_lr_call:

            def __init__(self, o):
                self.o = o

            def __enter__(self):
                self.o._get_lr_called_within_step = True
                return self

            def __exit__(self, type, value, traceback):
                self.o._get_lr_called_within_step = False
                return self

        with _enable_get_lr_call(self):
            for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
                param_group['lr'] = lr

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]






class SelfCELoss(nn.Module):
    def __init__(self):
        super(SelfCELoss, self).__init__()

        self.CE_Loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        loss = self.CE_Loss(inputs, targets)

        return loss.mean(dim=[1, 2])





def dice_coefficient_per_class(prediction, target, num_classes=3):
    # 用于存储每个类别的Dice系数
    dice_scores = []

    for class_id in range(1, num_classes):  # 从1开始，因为0是背景类
        # 针对当前类别的预测和目标
        pred_class = (torch.argmax(prediction, dim=1) == class_id).float()
        target_class = (target == class_id).float()

        # 计算交集和各自大小
        intersection = (pred_class * target_class).sum(dim=[1, 2])
        pred_sum = pred_class.sum(dim=[1, 2])
        target_sum = target_class.sum(dim=[1, 2])

        # 计算当前类别的Dice系数
        dice_class = (2 * intersection + 1e-6) / (pred_sum + target_sum + 1e-6)
        dice_scores.append(dice_class)

    # 返回每个类别的Dice系数
    return dice_scores




def iou_on_batch(pred, masks):
    ious = []

    # 使用torch.argmax从两个通道中选择最有可能的类别
    pred = torch.argmax(pred, dim=1)

    for i in range(pred.shape[0]):
        pred_tmp = pred[i].cpu().detach().numpy()
        mask_tmp = masks[i].cpu().detach().numpy()

        # 二值化预测和真实标签
        pred_tmp = (pred_tmp == 1).astype(np.uint8)
        mask_tmp = (mask_tmp == 1).astype(np.uint8)

        # 计算每个样本的IoU并添加到列表中
        ious.append(jaccard_score(mask_tmp.reshape(-1), pred_tmp.reshape(-1)))

    return np.mean(ious)


def save_on_batch(masks, pred, vis_path, names):
    pred = torch.argmax(pred, dim=1)

    # 定义五个类别的颜色
    colors = [(0, 0, 0), (255, 0, 0), (0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 0, 153)]

    for i in range(pred.shape[0]):
        # 分别处理预测和真实标签
        pred_tmp = pred[i].cpu().detach().numpy()
        mask_tmp = masks[i].cpu().detach().numpy()

        # 创建RGB图像，初始全黑
        pred_colored = np.zeros((pred_tmp.shape[0], pred_tmp.shape[1], 3), dtype=np.uint8)
        mask_colored = np.zeros((mask_tmp.shape[0], mask_tmp.shape[1], 3), dtype=np.uint8)

        # 确定存在的类别
        unique_pred_classes = np.unique(pred_tmp)
        unique_mask_classes = np.unique(mask_tmp)

        # 为预测中的每个存在的类别应用颜色
        for cls in unique_pred_classes:
            if cls < len(colors):  # 确保类别索引在颜色数组范围内
                pred_colored[pred_tmp == cls] = colors[cls]

        # 为真实标签中的每个存在的类别应用颜色
        for cls in unique_mask_classes:
            if cls < len(colors):  # 确保类别索引在颜色数组范围内
                mask_colored[mask_tmp == cls] = colors[cls]

        # 生成唯一文件名

        # 保存预测和真实标签图像
        cv2.imwrite(os.path.join(vis_path, names[0] + "_pred.jpg"), pred_colored)
        cv2.imwrite(os.path.join(vis_path, names[0] + "_gt.jpg"), mask_colored)


def calculate_accuracy(preds, labels):
    """
    计算多分类分割任务的准确率
    参数:
    preds: 模型的输出，形状为 (batch_size, num_classes, height, width)
    labels: 真实的标签，形状为 (batch_size, height, width)
    """
    # 将输出转换为预测的类别
    _, predicted_classes = torch.max(preds, dim=1)

    # 计算正确分类的像素数
    correct = (predicted_classes == labels).sum().item()

    # 计算总像素数
    total = labels.numel()

    # 计算准确率
    accuracy = correct / total

    return accuracy


def logger_config(log_path, name):
    '''
    配置日志记录器并返回。

    参数:
        log_path (str): 日志文件的保存路径。

    返回:
        logging.Logger: 配置好的日志记录器。

    配置日志记录器，设置日志级别为INFO，将日志记录到文件和控制台。
    '''
    logger = logging.getLogger(name)  # 获取日志记录器实例

    # 配置文件处理器，将日志写入指定文件
    if not logger.handlers:
        # 配置控制台处理器，将日志打印到控制台
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        logger.addHandler(console)

    # 将处理器添加到日志记录器
    handler = logging.FileHandler(log_path, encoding='UTF-8')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')  # 设置日志格式
    handler.setFormatter(formatter)
    logger.setLevel(level=logging.INFO)  # 设置日志级别为INFO
    logger.addHandler(handler)

    return logger  # 返回配置好的日志记录器实例


class _LRScheduler(object):

    def __init__(self, optimizer, last_epoch=-1):

        # Attach optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        # Initialize epoch and base learning rates
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.last_epoch = last_epoch

        # Following https://github.com/pytorch/pytorch/issues/20124
        # We would like to ensure that `lr_scheduler.step()` is called after
        # `optimizer.step()`
        def with_counter(method):
            if getattr(method, '_with_counter', False):
                # `optimizer.step()` has already been replaced, return.
                return method

            # Keep a weak reference to the optimizer instance to prevent
            # cyclic references.
            instance_ref = weakref.ref(method.__self__)
            # Get the unbound method for the same purpose.
            func = method.__func__
            cls = instance_ref().__class__
            del method

            @wraps(func)
            def wrapper(*args, **kwargs):
                instance = instance_ref()
                instance._step_count += 1
                wrapped = func.__get__(instance, cls)
                return wrapped(*args, **kwargs)

            # Note that the returned function here is no longer a bound method,
            # so attributes like `__func__` and `__self__` no longer exist.
            wrapper._with_counter = True
            return wrapper

        self.optimizer.step = with_counter(self.optimizer.step)
        self.optimizer._step_count = 0
        self._step_count = 0

        self.step()

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_last_lr(self):
        """ Return last computed learning rate by current scheduler.
        """
        return self._last_lr

    def get_lr(self):
        # Compute learning rate using chainable form of the scheduler
        raise NotImplementedError

    def step(self, epoch=None):
        # Raise a warning if old pattern is detected
        # https://github.com/pytorch/pytorch/issues/20124
        if self._step_count == 1:
            if not hasattr(self.optimizer.step, "_with_counter"):
                warnings.warn("Seems like `optimizer.step()` has been overridden after learning rate scheduler "
                              "initialization. Please, make sure to call `optimizer.step()` before "
                              "`lr_scheduler.step()`. See more details at "
                              "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate", UserWarning)

            # Just check if there were two first lr_scheduler.step() calls before optimizer.step()
            elif self.optimizer._step_count < 1:
                warnings.warn("Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
                              "In PyTorch 1.1.0 and later, you should call them in the opposite order: "
                              "`optimizer.step()` before `lr_scheduler.step()`.  Failure to do this "
                              "will result in PyTorch skipping the first value of the learning rate schedule. "
                              "See more details at "
                              "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate", UserWarning)
        self._step_count += 1

        class _enable_get_lr_call:

            def __init__(self, o):
                self.o = o

            def __enter__(self):
                self.o._get_lr_called_within_step = True
                return self

            def __exit__(self, type, value, traceback):
                self.o._get_lr_called_within_step = False
                return self

        with _enable_get_lr_call(self):
            if epoch is None:
                self.last_epoch += 1
                values = self.get_lr()
            else:
                self.last_epoch = epoch
                if hasattr(self, "_get_closed_form_lr"):
                    values = self._get_closed_form_lr()
                else:
                    values = self.get_lr()

        for param_group, lr in zip(self.optimizer.param_groups, values):
            param_group['lr'] = lr

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]


def weight_learner(cfeatures, pre_features, pre_weight1, args, global_epoch=0, iter=0):
    softmax = nn.Softmax(0)
    weight = Variable(torch.ones(cfeatures.size()[0], 1).cuda())
    weight.requires_grad = True
    cfeaturec = Variable(torch.FloatTensor(cfeatures.size()).cuda())
    cfeaturec.data.copy_(cfeatures.data)
    all_feature = torch.cat([cfeaturec, pre_features.detach()], dim=0)
    optimizerbl = torch.optim.SGD([weight], lr=args.lr, momentum=0.9)

    for epoch in range(args.epochb):
        lr_setter(optimizerbl, epoch, args, bl=True)
        all_weight = torch.cat((weight, pre_weight1.detach()), dim=0)
        optimizerbl.zero_grad()

        lossb = lossb_expect(all_feature, softmax(all_weight), args.num_f, args.sum)
        lossp = softmax(weight).pow(args.decay_pow).sum()
        lambdap = args.lambdap * max((args.lambda_decay_rate ** (global_epoch // args.lambda_decay_epoch)),
                                     args.min_lambda_times)
        lossg = lossb / lambdap + lossp
        if global_epoch == 0:
            lossg = lossg * args.first_step_cons

        lossg.backward(retain_graph=True)
        optimizerbl.step()

    if global_epoch == 0 and iter < 10:
        pre_features = (pre_features * iter + cfeatures) / (iter + 1)
        pre_weight1 = (pre_weight1 * iter + weight) / (iter + 1)

    elif cfeatures.size()[0] < pre_features.size()[0]:
        pre_features[:cfeatures.size()[0]] = pre_features[:cfeatures.size()[0]] * args.presave_ratio + cfeatures * (
                1 - args.presave_ratio)
        pre_weight1[:cfeatures.size()[0]] = pre_weight1[:cfeatures.size()[0]] * args.presave_ratio + weight * (
                1 - args.presave_ratio)

    else:
        pre_features = pre_features * args.presave_ratio + cfeatures * (1 - args.presave_ratio)
        pre_weight1 = pre_weight1 * args.presave_ratio + weight * (1 - args.presave_ratio)

    softmax_weight = softmax(weight)

    return softmax_weight, pre_features, pre_weight1


def lr_setter(optimizer, epoch, args, bl=False):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""

    lr = args.lr
    if bl:
        lr = 1.0 * (0.1 ** (epoch // (20 * 0.5)))
    else:
        if 1:
            lr *= ((0.01 + math.cos(0.5 * (math.pi * epoch / args.epochs))) / 1.01)
        else:
            if epoch >= [24, 30][0]:
                lr *= 0.1
            if epoch >= [24, 30][1]:
                lr *= 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def lossb_expect(cfeaturec, weight, num_f, sum=True):
    cfeaturecs = random_fourier_features_gpu(cfeaturec, num_f=num_f, sum=sum).cuda()
    loss = Variable(torch.FloatTensor([0]).cuda())
    weight = weight.cuda()
    for i in range(cfeaturecs.size()[-1]):
        cfeaturec = cfeaturecs[:, :, i]

        cov1 = cov(cfeaturec, weight)
        cov_matrix = cov1 * cov1
        loss += torch.sum(cov_matrix) - torch.trace(cov_matrix)

    return loss


def random_fourier_features_gpu(x, w=None, b=None, num_f=None, sum=True, sigma=None, seed=None):
    if num_f is None:
        num_f = 1
    n = x.size(0)
    r = x.size(1)
    x = x.view(n, r, 1)
    c = x.size(2)
    if sigma is None or sigma == 0:
        sigma = 1
    if w is None:
        w = 1 / sigma * (torch.randn(size=(num_f, c)))
        b = 2 * np.pi * torch.rand(size=(r, num_f))
        b = b.repeat((n, 1, 1))

    Z = torch.sqrt(torch.tensor(2.0 / num_f).cuda())

    mid = torch.matmul(x.cuda(), w.t().cuda())

    mid = mid + b.cuda()
    mid -= mid.min(dim=1, keepdim=True)[0]
    mid /= mid.max(dim=1, keepdim=True)[0].cuda()
    mid *= np.pi / 2.0

    if sum:
        Z = Z * (torch.cos(mid).cuda() + torch.sin(mid).cuda())
    else:
        Z = Z * torch.cat((torch.cos(mid).cuda(), torch.sin(mid).cuda()), dim=-1)

    return Z


def cov(x, w=None):
    if w is None:
        n = x.shape[0]
        cov = torch.matmul(x.t(), x) / n
        e = torch.mean(x, dim=0).view(-1, 1)
        res = cov - torch.matmul(e, e.t())
    else:
        w = w.view(-1, 1)
        cov = torch.matmul((w * x).t(), x)
        e = torch.sum(w * x, dim=0).view(-1, 1)
        res = cov - torch.matmul(e, e.t())

    return res


def save_checkpoint(logger, state, save_path):
    '''
        Save the current model.
        If the model is the best model since beginning of the training
        it will be copy
    '''
    logger.info('\t Saving to {}'.format(save_path))
    if not os.path.isdir(os.path.join(save_path, 'model')):
        os.makedirs(os.path.join(save_path, 'model'))
    epoch = state['epoch']  # epoch no
    best_model = state['best_model']  # bool
    model = state['model']  # model type

    if best_model:
        filename = os.path.join(save_path, 'model', '{}.pth'.format(model))

    else:
        filename = os.path.join(save_path, 'model', 'model_{}_{:02d}.pth'.format(model, epoch))

    torch.save(state, filename)

    # torch.save(save_dict, os.path.join(args.model_result_dir, "best.pth"))  # 保存当前模型状态



