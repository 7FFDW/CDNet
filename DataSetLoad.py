# -*- coding: utf-8 -*-
from math import comb

import numpy as np
import openslide
import torch
import random

from matplotlib import pyplot as plt
from PIL import Image, ImageEnhance, ImageFilter
from batchgenerators.utilities.file_and_folder_operations import subfiles, join

from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as F
import os
import cv2

from PIL import ImageFile


import albumentations as A
from albumentations.pytorch import ToTensorV2

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

# 定义数据增强
transform = A.Compose(
    [
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.25),
        A.RandomBrightnessContrast(p=0.25),
        A.ShiftScaleRotate(shift_limit=0, p=0.25),
        A.CoarseDropout(),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])


def random_rotate_flip_color(image, mask, size=(224, 224)):
    # 随机旋转角度
    k = np.random.randint(0, 4)

    # 随机翻转轴
    axis = np.random.randint(0, 2)

    # 应用旋转和翻转
    image = np.rot90(image, k)
    mask = np.rot90(mask, k)
    image = np.flip(image, axis=axis)
    mask = np.flip(mask, axis=axis)

    # 转换为PIL图像进行色彩调整
    image = Image.fromarray(image)

    # 随机调整颜色强度
    if random.random() < 0.5:
        enhancer = ImageEnhance.Color(image)
        factor = random.uniform(0.5, 1.5)  # 调整范围
        image = enhancer.enhance(factor)

    # 随机应用高斯模糊
    if random.random() < 0.5:
        image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 2.0)))

    # 随机裁剪
    if random.random() < 0.5:
        image = image.resize(size, Image.Resampling.LANCZOS)
        mask = Image.fromarray(mask).resize(size, Image.Resampling.NEAREST)

    # 转换回numpy数组
    image = np.array(image)
    mask = np.array(mask)


    return image, mask


def to_long_tensor(pic):
    # 将输入的图像转换为NumPy数组，数据类型为无符号8位整型（uint8）
    img = torch.from_numpy(np.array(pic, np.uint8))
    # 将NumPy数组转换为PyTorch的长整型张量（long tensor）
    return img.long()



# 定义一个数据增强类
class RandomGenerator(object):
    def __init__(self, output_size):
        # 初始化函数，指定输出图像的大小
        self.output_size = output_size

    # 定义对象实例被调用时执行的方法
    def __call__(self, sample):
        # 从输入样本中获取图像、标签和文本数据

        image, label = sample['image'], sample['label']

        image, label = image.astype(np.uint8), label.astype(np.uint8)

        
        image, label = F.to_pil_image(image), F.to_pil_image(label)




        image, label = random_rotate_flip_color(image, label,self.output_size)

        # 转换为PyTorch张量格式
        image, label = F.to_tensor(image), to_long_tensor(label)


        # 返回处理后的样本
        sample = {'image': image, 'label': label}
        return sample


# 定义一个验证数据增强类
class ValGenerator(object):
    def __init__(self, output_size):
        # 初始化函数，指定输出图像的大小
        self.output_size = output_size

    def __call__(self, sample):
        # 获取图像、标签和文本数据
        image, label = sample['image'], sample['label']

        # 将图像和标签转换为PIL图像格式
        image, label = F.to_pil_image(image.astype(np.uint8)), F.to_pil_image(label.astype(np.uint8))

        # 转换为PyTorch张量格式
        image, label = F.to_tensor(image), to_long_tensor(label)


        # 构建处理后的样本字典并返回

        sample = {'image': image, 'label': label}




        return sample


class SupervisedDataset(Dataset):

    def __init__(self, args, isTrain='Train',) -> None:
        self.image_size = args.img_size
        self.isTrain = isTrain
        self.classes = args.n_labels
        if isTrain == 'Train':
            self.img_path = args.img_data_dir
            self.label_path = args.mask_data_dir
            self.joint_transform = transforms.Compose([RandomGenerator(output_size=[self.image_size, self.image_size])])
        elif isTrain == 'Val':
            self.img_path = args.img_data_dir
            self.label_path = args.mask_data_dir
            self.joint_transform = ValGenerator(output_size=[self.image_size, self.image_size])
        else:
            self.img_path = args.testdata_dir
            self.joint_transform = ValGenerator(output_size=[self.image_size, self.image_size])



        self.images_list = []  # 初始化一个空列表以存储文件路径
        self.labels_list = []  # 初始化一个空列表以存储文件路径
        self.text = {}  # 存储扫描中切片的相对位置





        if isTrain == 'Train':
            for key in os.listdir(args.train_data_dir):
                self.images_list.append(
                    {'path': join(args.train_data_dir, key), 'filename': key})
        elif isTrain == 'Val':
            for key in os.listdir(args.val_data_dir):
                self.images_list.append(
                    {'path': join(args.val_data_dir, key), 'filename': key})
        else:
            for key in os.listdir(args.test_data_dir):
                self.images_list.append(
                    {'path': join(args.test_data_dir, key), 'filename': key})




    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):

        image_path = self.images_list[idx]['path']
        image_name = self.images_list[idx]['filename']

        label_path = image_path.replace('img', 'labelcol').replace('.jpg', '_mask.jpg').replace('Images','masks').replace('tif','png').replace('image','mask')

        img = openslide.open_slide(image_path)
        image = np.array(img.read_region((0, 0), 0, (img.dimensions))).astype(np.float32)

        image = image[:, :, :3]




        mask = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)


        image = cv2.resize(image, (self.image_size, self.image_size))
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        if len(mask.shape) >= 3:
            mask = mask[..., 0]
        # change
        if self.classes <= 2:

            mask[mask == 0] = 0
            mask[mask > 0] = 1

        sample = {'image': image, 'label': mask}



        sample = self.joint_transform(sample)
        

        return sample, image_name











