# -*- coding: utf-8 -*-
import warnings
from datetime import datetime
import json

import torch.optim
import torch.nn as nn

from tensorboardX import SummaryWriter
import os

from torch.backends import cudnn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from torch.utils.data import DataLoader

from utils.DataSetLoad import SupervisedDataset
from config import get_config



from utils.MultiClassLoss import MultiClassLoss

from utils.metric import iou_with_dice
from utils import save_checkpoint, iou_on_batch, save_on_batch, logger_config, weight_learner, calculate_accuracy, set_seed

from models.CDNet import CDNet

warnings.filterwarnings("ignore")





##################################################################################
# =================================================================================
#          Main Loop: load model,
# =================================================================================
##################################################################################


def main_loop(args=None):
    # Load train and img data
    result_dir = args.model_type + args.save
    model = CDNet(args=args)

    if args.restart:
        logger.info('loading from saved model ' + args.pretrained_model_path)
        dict = torch.load(args.pretrained_model_path,
                          map_location=lambda storage, loc: storage)
        save_model = dict["state_dict"]
        model_dict = model.state_dict()
        # we only need to load the parameters of the encoder
        state_dict = {k.replace('module.', ''): v for k, v in save_model.items()}
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)
    model.to(args.device)
    num_parameters = sum([l.nelement() for l in model.parameters()])
    logger.info(f"number of parameters: {num_parameters}")

    with open("data_split.json") as f:
        datamap = json.load(f)

    train_dataset = SupervisedDataset(args, isTrain='Train', )
    validate_dataset = SupervisedDataset(args, isTrain='Val', )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.num_works, drop_last=True)
    val_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.num_works, drop_last=False)

    lr = args.lr

    model = model.cuda()
    if torch.cuda.device_count() > 1:
        print("Let's use {0} GPUs!".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    criterion = MultiClassLoss(args, reduce=True, only_ce=False)


    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr,
                                  weight_decay=2e-2)  # Choose optimize

    lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.9, patience=10, verbose=True,
                                         min_lr=args.min_lr)

    if args.tensorboard:
        log_dir = os.path.join(args.runs_dir, 'train', args.model_type + args.save, 'tensorboard_logs')
        logger.info('log dir: '.format(log_dir))
        writer = SummaryWriter(log_dir)
    else:
        writer = None

    max_dice = 0.0
    best_epoch = 1
    for epoch in range(args.epochs):  # loop over the dataset multiple times
        logger.info('\n========= Epoch [{}/{}] ========='.format(epoch + 1, args.epochs))
        logger.info(args.model_type)
        # train for one epoch
        model.train(True)
        logger.info('Training with batch size : {}'.format(args.batch_size))
        train_one_epoch(train_loader, model, criterion, optimizer, writer, epoch, None, logger, args)  # sup

        # evaluate on validation set
        logger.info('Validation')
        with torch.no_grad():
            model.eval()
            val_loss, val_dice, val_iou = train_one_epoch(val_loader, model, criterion,
                                                          optimizer, writer, epoch, lr_scheduler, logger, args)
        # =============================================================
        #       Save best model
        # =============================================================
        if val_dice > max_dice:
            if epoch > 200:
                result_dir = args.model_type + args.save + '_' + str(epoch + 1)
            best_epoch = epoch + 1
            logger.info(
                '\t Saving best model, mean dice increased from: {:.4f} to {:.4f} in epoch {}'.format(max_dice,
                                                                                                      val_dice,
                                                                                                      best_epoch))
            max_dice = val_dice
            save_checkpoint(logger, {'epoch': epoch,
                                     'best_model': True,
                                     'model': result_dir,
                                     'state_dict': model.state_dict(),
                                     'val_loss': val_loss,
                                     'optimizer': optimizer.state_dict()}, args.save_path)
        else:
            logger.info('\t Mean dice:{:.4f} does not increase, '
                        'the best is still: {:.4f} in epoch {}'.format(val_dice, max_dice, best_epoch))
        early_stopping_count = epoch - best_epoch + 1
        logger.info('\t early_stopping_count: {}/{}'.format(early_stopping_count, args.early_stopping_patience))

        if early_stopping_count >= args.early_stopping_patience:
            logger.info('\t early_stopping!')
            break
    return model


def train_one_epoch(loader, model, criterion, optimizer, writer, epoch, lr_scheduler, logger, args):
    logging_mode = 'Train' if model.training else 'Val'
    time_sum, loss_sum = 0, 0
    dice_sum, iou_sum, acc_sum, average_loss, train_dice_avg = 0.0, 0.0, 0.0, 0.0, 0.0
    dices = []
    loop = tqdm(enumerate(loader, 1), total=len(loader), leave=True, initial=1)
    for i, (sampled_batch, image_name) in loop:
        try:
            loss_name = criterion._get_name()
        except AttributeError:
            loss_name = criterion.__name__

        # Take variable and put them to GPU
        images, masks = sampled_batch['image'], sampled_batch['label']
        # print(np.unique(masks.numpy()))
        images, masks = images.cuda(), masks.cuda()

        # ====================================================
        #             Compute loss
        # ====================================================
        preds, cfeatures = model(images)
        # preds = model(images)
        pre_features = model.pre_features
        pre_weight1 = model.pre_weight1
        if model.training:
            weight1, pre_features, pre_weight1 = weight_learner(cfeatures, pre_features, pre_weight1, args, epoch, i)
            model.pre_features.data.copy_(pre_features)
            model.pre_weight1.data.copy_(pre_weight1)
            # weight1 = None
            out_loss = criterion(preds, masks.long(), weight1)
            out_loss = out_loss.mean()
            optimizer.zero_grad()
            out_loss.backward()
            optimizer.step()
        else:
            out_loss = criterion(preds, masks)
            out_loss = out_loss.mean()
        preds = nn.Softmax(dim=1)(preds)
        if args.n_labels > 2:
            train_dice, train_iou = iou_with_dice(preds, masks, args.n_labels)
            train_dice = train_dice.mean()
            train_iou = train_iou.mean()
        else:
            train_iou = iou_on_batch(preds, masks)
            train_dice = criterion._show_dice(preds, masks.float())[1]
        train_acc = calculate_accuracy(preds, masks)

        if epoch % args.vis_frequency == 0 and logging_mode == 'Val':
            vis_path = os.path.join(args.runs_dir, "train", args.model_type + args.save, args.visualize_path,
                                    str(epoch))
            if not os.path.isdir(vis_path):
                os.makedirs(vis_path)
            save_on_batch(masks, preds, vis_path, image_name)
        dices.append(train_dice)

        loss_sum += len(images) * out_loss.item()
        iou_sum += len(images) * train_iou
        acc_sum += len(images) * train_acc
        dice_sum += len(images) * train_dice.item()

        if i == len(loader):
            average_loss = loss_sum / (args.batch_size * (i - 1) + len(images))
            train_iou_avg = iou_sum / (args.batch_size * (i - 1) + len(images))
            train_acc_avg = acc_sum / (args.batch_size * (i - 1) + len(images))
            train_dice_avg = dice_sum / (args.batch_size * (i - 1) + len(images))
        else:
            average_loss = loss_sum / (i * args.batch_size)
            train_iou_avg = iou_sum / (i * args.batch_size)
            train_acc_avg = acc_sum / (i * args.batch_size)
            train_dice_avg = dice_sum / (i * args.batch_size)

        torch.cuda.empty_cache()
        out_loss = out_loss.item()
        train_dice = train_dice.item()
        if i % args.print_frequency == 0:
            logger.info(
                f' [{str(logging_mode)}-{epoch + 1}-{i}] Loss: {out_loss:.4f} (Avg {average_loss:.4f}) Acc: {train_acc:.4f} (Avg {train_acc_avg:.4f}) IoU: {train_iou:.3f} (Avg {train_iou_avg:.4f}) Dice: {train_dice:.4f} (Avg {train_dice_avg:.4f}) LR {min(g["lr"] for g in optimizer.param_groups):.2e}  GPU Memory Usage: {torch.cuda.memory_allocated() / 1024 ** 3 :.2f} GB / {torch.cuda.max_memory_allocated() / 1024 ** 3 :.2f} GB')
        step = epoch * len(loader) + i
        writer.add_scalar(logging_mode + '_' + loss_name, out_loss, step)

        # plot metrics in tensorboard
        writer.add_scalar(logging_mode + '_iou', train_iou, step)
        writer.add_scalar(logging_mode + '_acc', train_acc, step)
        writer.add_scalar(logging_mode + '_dice', train_dice, step)

        torch.cuda.empty_cache()

    if lr_scheduler is not None:
        lr_scheduler.step()

    return average_loss, train_dice_avg, train_iou_avg


if __name__ == '__main__':

    args = get_config()
    deterministic = True
    args.model_type = 'supervised_' + str(args.epochb) + '_' + str(args.num_f) + '_'
    if not deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    set_seed(args.seed)

    if args.save == '':
        args.save = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    args.save_path = os.path.join(args.results_dir, 'train', args.model_type + args.save)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    if not os.path.exists(os.path.join(args.runs_dir, 'train', args.model_type + args.save, args.save + '.log')):
        os.makedirs(os.path.join(args.runs_dir, 'train', args.model_type + args.save))
    args.logger_path = os.path.join(args.runs_dir, 'train', args.model_type + args.save, args.save + '.log')
    logger = logger_config(log_path=args.logger_path, name='train')
    model = main_loop(args=args)
