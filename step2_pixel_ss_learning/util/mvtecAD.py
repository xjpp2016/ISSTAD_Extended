#coding=utf-8
from os.path import join
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
import numpy as np
import torchvision.transforms as transforms
import os
import random
from util.mpdd_aug import ImageAugmentation, Aug_old

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



class AD_TRAIN(Dataset):
    def __init__(self, data_path, args):
        super(AD_TRAIN, self).__init__()

        self.filenames = self.get_image_paths(data_path)
        self.object_name = args.object_name

        # object_name_list = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']
        # object_name = [s for s in object_name_list if s in data_path][0]
        # self.object_name = object_name

        self.img_transform = get_transform(convert=True, normalize=True)
        self.lab_transform = get_transform()
        self.aug = ImageAugmentation(object_name=self.object_name)
        self.aug_old = Aug_old()
        self.aug_type = args.aug_type

        self.rad_max = args.rad_max

    def get_image_paths(self, directory):
        image_paths = []
        for root, _, files in os.walk(directory):
            for file in files:
                if is_image_file(file):
                    image_paths.append(os.path.join(root, file))
        return image_paths

    def __getitem__(self, index):
        image1 = Image.open(self.filenames[index]).convert('RGB')
        label = Image.new('L', image1.size, 0)

        fc_label = random.randint(0,self.rad_max)
        if fc_label!=0:
             if self.aug_type == 'old':
                 image1, label = self.aug_old(image1, label)
             else:
                 image1, label = self.aug(image1, label)
        
        image1 = self.img_transform(image1)
        label = self.lab_transform(label)
        label = make_one_hot(label.unsqueeze(0).long(), 2).squeeze(0)
        return image1, label

    def __len__(self):
        return len(self.filenames)
    


class AD_TEST(Dataset):
    def __init__(self, args, img_path, lab_path):
        super(AD_TEST, self).__init__()
        # 获取图片列表
        self.img_filenames = []
        self.lab_filenames = []
        for root, dirs, files in os.walk(img_path):  # 获取所有文件
                for file in files:  # 遍历所有文件名
                    if os.path.splitext(file)[1] == '.png':   # 指定尾缀  ***重要***
                        self.img_filenames.append(os.path.join(root, file))  # 拼接处绝对路径并放入列表
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


def ob_p(args):

    objects_list = ['bottle', 'cable', 'capsule',  'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']

    if 'bottle' == args.object_name:
        args.p = 0.1
        args.w_m = 1
        args.w_r = 1
        args.num_epochs = 200
        args.rad_max = 2
        args.aug_type = 'new'
    elif 'cable' == args.object_name:
        args.p = 0.1
        args.w_m = 1
        args.w_r = 0.01
        args.num_epochs = 200
        args.rad_max = 2
        args.aug_type = 'new'
    elif 'capsule' == args.object_name:
        args.p = 0.1
        args.w_m = 1
        args.w_r = 0.1
        args.num_epochs = 200
        args.rad_max = 5
        args.aug_type = 'old'
    elif 'carpet' == args.object_name:
        args.p = 0.1
        args.w_m = 1
        args.w_r = 0.1
        args.num_epochs = 200
        args.rad_max = 5
        args.aug_type = 'old'
    elif 'grid' == args.object_name:
        args.p = 0.1
        args.w_m = 1
        args.w_r = 1
        args.num_epochs = 200
        args.rad_max = 5
        args.aug_type = 'old'
    elif 'hazelnut' == args.object_name:
        args.p = 0.1
        args.w_m = 1
        args.w_r = 0.1
        args.num_epochs = 200
        args.rad_max = 2
        args.aug_type = 'new'
    elif 'leather' == args.object_name:
        args.p = 0.1
        args.w_m = 1
        args.w_r = 0.1
        args.num_epochs = 200
        args.rad_max = 2
        args.aug_type = 'new'
    elif 'metal_nut' == args.object_name:
        args.p = 0.1
        args.w_m = 1
        args.w_r = 1
        args.num_epochs = 200
        args.rad_max = 2
        args.aug_type = 'new'
    elif 'pill' == args.object_name:
        args.p = 0.1
        args.w_m = 1
        args.w_r = 1
        args.num_epochs = 200
        args.rad_max = 2
        args.aug_type = 'new'
    elif 'screw' == args.object_name:
        args.p = 0.1
        args.w_m = 0.1
        args.w_r = 1
        args.num_epochs = 200
        args.rad_max = 2
        args.aug_type = 'new'
    elif 'tile' == args.object_name:
        args.p = 0.1
        args.w_m = 1
        args.w_r = 1
        args.num_epochs = 200
        args.rad_max = 2
        args.aug_type = 'new'
    elif 'toothbrush' == args.object_name:
        args.p = 0.1
        args.w_m = 1
        args.w_r = 1
        args.num_epochs = 200
        args.rad_max = 2
        args.aug_type = 'new'
    elif 'transistor' == args.object_name:
        args.p = 0.1
        args.w_m = 1
        args.w_r = 0.1
        args.num_epochs = 200
        args.rad_max = 5
        args.aug_type = 'old'
    elif 'wood' == args.object_name:
        args.p = 0.1
        args.w_m = 0.1
        args.w_r = 1
        args.num_epochs = 200
        args.rad_max = 5
        args.aug_type = 'old'
    elif 'zipper' == args.object_name:
        args.p = 0.1
        args.w_m = 0.1
        args.w_r = 1
        args.num_epochs = 200
        args.rad_max = 5
        args.aug_type = 'old'
    else:
        args.p = 0.1
        args.w_m = 1
        args.w_r = 1
        args.num_epochs = 200
        args.rad_max = 2
    
    return args
