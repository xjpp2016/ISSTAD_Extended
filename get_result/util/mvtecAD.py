#coding=utf-8
from os.path import join
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
import numpy as np
import torchvision.transforms as transforms
import os
import random

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.tif','.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.

    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape)
    result = result.scatter_(1, input.cpu(), 1)
    return result


def get_transform(convert=True, normalize=False):
    transform_list = []
    if convert:
        transform_list += [
                            transforms.ToTensor(),
                            transforms.Resize([224,224])
                           ]
    if normalize:
        transform_list += [
            # transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
            transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5)),
            transforms.Resize([224,224])]
    return transforms.Compose(transform_list)

class AD_TEST(Dataset):
    def __init__(self, args, img_path, lab_path):
        super(AD_TEST, self).__init__()
        # 获取图片列表
        self.img_filenames = []
        self.lab_filenames = []
        for root, dirs, files in os.walk(img_path):  # 获取所有文件
                for file in files:  # 遍历所有文件名
                    full_path = os.path.join(root, file)
                    if os.path.splitext(file)[1] == '.png' and 'good' not in full_path.replace('\\', '/'):
                        self.img_filenames.append(full_path)
        for root, dirs, files in os.walk(lab_path):  # 获取所有文件
            for file in files:  # 遍历所有文件名
                if os.path.splitext(file)[1] == '.png':   # 指定尾缀  ***重要***
                    self.lab_filenames.append(os.path.join(root, file))  # 拼接处绝对路径并放入列表


        self.transform = get_transform(convert=True, normalize=True)  # convert to tensor and normalize to [-1,1]
        self.label_transform = get_transform()  # only convert to tensor
        self.n = 0


    def __getitem__(self, index):
        img = Image.open(self.img_filenames[index]).convert('RGB')
        img = self.transform(img)
        pn = 1
        if 'good' in self.img_filenames[index]:
            label = Image.new('L', [1024,1024], 0)
            self.n = self.n + 1
            pn = 0
        else:
            label = Image.open(self.lab_filenames[index-self.n])                      
        label = self.label_transform(label)

        return img, pn, label, self.img_filenames[index].replace('.png', '.tif').replace('/test/', '/result/admap/')

    def __len__(self):
        return len(self.img_filenames)
