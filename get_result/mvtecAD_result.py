# coding=utf-8

from mimetypes import guess_all_extensions
import os
import datetime
import time
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import torch.backends.cudnn as cudnn
import torch
import random
import matplotlib.cm as cm
from scipy.ndimage import gaussian_filter

import timm
assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

from sklearn.metrics import roc_auc_score
from PIL import Image
import subprocess

from util.mvtecAD import AD_TEST
from util.mvtecAD_options import parser
from model import models

from util.misc import seed_torch

def setdir(file_path):
    if not os.path.exists(file_path):  
        os.makedirs(file_path)

def save_heatmap(array, save_path):
    # 归一化到 0-1
    array = (array - np.min(array)) / (np.ptp(array) + 1e-8)
    # 转为伪彩色（jet colormap）
    heatmap = cm.get_cmap('jet')(array)[:, :, :3]  # 去掉 alpha 通道
    heatmap = (heatmap * 255).astype(np.uint8)
    Image.fromarray(heatmap).save(save_path)


def replace_path(tiff_path, object_name, maps_dir):
    path = tiff_path.replace('\\', '/')  # 统一为正斜杠，便于处理
    parts = path.split('/')
    try:
        idx = parts.index(object_name)
        new_path = os.path.join(maps_dir, *parts[idx+1:])
        return new_path
    except ValueError:
        raise ValueError(f"{object_name} not found in path: {path}")

if __name__ == '__main__':
    
    args = parser.parse_args()
    object_name = args.object_name

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # set seeds
    seed = args.seed
    if args.seed is None:
        args.seed = random.randint(1, 10000)
    seed_torch(seed=args.seed)

    # set paths
    step1_saved_model = args.step1_saved_models_dir + object_name + '/checkpoint-0.75.pth'
    auc_saved_path = args.auc_saved_dir + object_name + '/'
    if auc_saved_path:
        setdir(auc_saved_path)
    spro_saved_path = args.spro_saved_dir + object_name + '/'
    if spro_saved_path:
        setdir(spro_saved_path)
    log_saved_path = args.log_saved_dir + object_name + '/'
    if log_saved_path:
        setdir(log_saved_path)
    model_saved_path = args.saved_models_dir + object_name + '/'
    if model_saved_path:
        setdir(model_saved_path)


    # define model
    model_ad = getattr(models, args.arch)()
    model_ad = model_ad.to(device)


    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model_ad = torch.nn.DataParallel(model_ad, device_ids=range(torch.cuda.device_count()))

    test_set = AD_TEST(args, args.test_data_dir, args.test_label_dir)
    test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)

    model_ad.load_state_dict(torch.load(model_saved_path + 'net_ad.pth'))

    # testing
    model_ad.eval()
    with torch.no_grad():
        val_bar = tqdm(test_loader)
        inter, unin = 0,0
        last_inter, last_unin = 0,0
        valing_results = {'batch_sizes': 0}

        pn_pixel_list = []
        mix_pixel_list = []
        m_pinxel_list = []
        r_pinxel_list = []

        pn_mean_list = []
        mean_list = []

        for img, pn, label, tiff_path in val_bar:
            valing_results['batch_sizes'] += args.val_batchsize

            img = img.to(device, dtype=torch.float)

            rimg, mask_list = model_ad(img)

            pred_softmax_mean = 0
            for mask in mask_list:
                pred_softmax_1 = torch.softmax(mask.permute(0,2,3,1),dim=-1)[:,:,:,1]
                pred_softmax_mean = pred_softmax_mean + pred_softmax_1

            pred_softmax_mean = (pred_softmax_mean/len(mask_list))
            
            m_admap = pred_softmax_mean
            r_admap = ((rimg - img)**2).mean(dim=1)
            pmap = m_admap
            rmap = r_admap
            m_admap = gaussian_filter(m_admap.squeeze(0).cpu(), sigma=4)
            r_admap = gaussian_filter(r_admap.squeeze(0).cpu(), sigma=4)
            admap = gaussian_filter(m_admap * r_admap, sigma=4)


            min_value = torch.min(rmap)
            max_value = torch.max(rmap)
            rmap = (rmap - min_value) / (max_value - min_value)
            mixmap =  (pmap * rmap)
            pmap = gaussian_filter(pmap.squeeze(0).cpu(), sigma=4)
            rmap = gaussian_filter(rmap.squeeze(0).cpu(), sigma=4)
            mixmap = gaussian_filter(mixmap.squeeze(0).cpu(), sigma=4)


            save_img_path = replace_path(tiff_path[0], args.object_name, args.maps_dir)
            dir_temp_path = os.path.dirname(save_img_path)
            os.makedirs(dir_temp_path, exist_ok=True)

            new_path = save_img_path.replace(".tif", "_ad.png")
            save_heatmap(mixmap, new_path)

            new_path = save_img_path.replace(".tif", "_m.png")
            save_heatmap(pmap, new_path)

            new_path = save_img_path.replace(".tif", "_r.png")
            save_heatmap(rmap, new_path)


            pn_pixel_list.extend(label.squeeze(0).squeeze(0).cpu().numpy().astype(int).ravel())
            mix_pixel_list.extend(mixmap.ravel())


        pixel_auc = roc_auc_score(pn_pixel_list, mix_pixel_list)


        today = time.strftime("%Y-%m-%d", time.localtime())
        now = datetime.datetime.now().strftime('%y%m%d%H')
        f = open( './result/auc/MVTecAD/all_result' + today + now +'.txt','a')
        print(args.object_name, 
            "pixel_auc",  "{:.2f}".format(pixel_auc*100), "pixel_auc", 
            file=f)
        f.close()

                    

