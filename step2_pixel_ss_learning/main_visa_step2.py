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
from scipy.ndimage import gaussian_filter

import timm
assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

from sklearn.metrics import roc_auc_score
from PIL import Image
import subprocess

from util.losses import cross_entropy
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.simm import SSIM
from util.visa import AD_TEST, AD_TRAIN, ob_p
from util.visa_options import parser
from model import models

from util.misc import seed_torch, compute_pro, mean_of_top

def setdir(file_path):
    if not os.path.exists(file_path):  
        os.makedirs(file_path)

if __name__ == '__main__':
    
    args = parser.parse_args()
    object_name = args.object_name
    args = ob_p(args)

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

    min_loss = 0

    # define model
    model_ad = getattr(models, args.arch)()

    checkpoint = torch.load(step1_saved_model, map_location='cpu', weights_only=False)
    msg = model_ad.load_state_dict(checkpoint['model'], strict=False)

    model_ad = model_ad.to(device)


    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model_ad = torch.nn.DataParallel(model_ad, device_ids=range(torch.cuda.device_count()))

    param_groups = optim_factory.add_weight_decay(model_ad, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))

    weight = torch.tensor([3.0, 1.0]).to(device, dtype=torch.float)
    ce_loss = cross_entropy(weight=weight).to(device, dtype=torch.float)
    ssim_loss = SSIM()

    loss_scaler = NativeScaler()

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.T_max, eta_min=args.min_lr)
    best_auc = 0
    best_pixel_auc = 0
    best_pixel_apro = 0
    best_m_image_auc = 0
    best_r_image_auc = 0

    test_set = AD_TEST(args, args.test_data_dir, args.test_label_dir)
    test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)

    # training
    for epoch in range(1, args.num_epochs+1):

        # load data
        train_set = AD_TRAIN(args.train_data_dir, args)
        train_loader = DataLoader(dataset=train_set, num_workers=args.num_workers, batch_size=args.batchsize, shuffle=True)
        
        train_bar = tqdm(train_loader)
        running_results = {'batch_sizes': 0, 'loss': 0 }

        model_ad.train(True)
        optimizer.zero_grad()
        for img, label in train_bar:
            running_results['batch_sizes'] += args.batchsize

            img = img.to(device, dtype=torch.float)
            label = label.to(device, dtype=torch.float)
            label = torch.argmax(label, 1).unsqueeze(1).float()

            rimg, mask_list = model_ad(img)
            loss_m, loss_fc = 0, 0  

            for mask in mask_list:
                loss_m = loss_m + ce_loss(mask, label)
            loss_m = loss_m/len(mask_list)
            loss_r = ((rimg - img)**2).mean()
            loss_s = 1 - ssim_loss(rimg,img)       

            loss = loss_m + 0.5*loss_r + 0.1*loss_s

            accum_iter = args.accum_iter
            loss /= accum_iter
            loss_scaler(loss, optimizer, parameters=model_ad.parameters())

            torch.cuda.synchronize()

            running_results['loss'] += loss.item() * args.batchsize

            train_bar.set_description(
                desc='[%d/%d] loss: %.4f' % (
                    epoch, args.num_epochs,
                    running_results['loss'] / running_results['batch_sizes'],))
        scheduler.step()

        #torch.save(model_ad.state_dict(),  model_saved_path + 'net_ad.pth')

        # testing
        model_ad.eval()
        with torch.no_grad():
            val_bar = tqdm(test_loader)
            inter, unin = 0,0
            last_inter, last_unin = 0,0
            valing_results = {'batch_sizes': 0}

            m = 0
            pn_mean_list = []
            mean_list = []
            pn_pixel_list = []
            pixel_list = [] 
            pn_label_list = []
            preimg_list = []
            m_mean_list = []  
            r_mean_list = []    

            for img, pn, label, tiff_path in val_bar:
                valing_results['batch_sizes'] += args.val_batchsize

                img = img.to(device, dtype=torch.float)

                rimg, mask_list = model_ad(img)

                pred_softmax_mean = 0
                for mask in mask_list:
                    pred_softmax_1 = torch.softmax(mask.permute(0,2,3,1),dim=-1)[:,:,:,1]
                    pred_softmax_mean = pred_softmax_mean + pred_softmax_1

                pred_softmax_mean = (pred_softmax_mean/len(mask_list))

                anomaly_map = gaussian_filter(pred_softmax_mean.squeeze(0).cpu(), sigma=3)
                
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
                
                
                image = Image.fromarray(mixmap*255.0)
                tif_save_path = folder_path = os.path.dirname(tiff_path[0])
                if tif_save_path:
                    setdir(tif_save_path)
                image.save(tiff_path[0], 'TIFF')

                new_path = tiff_path[0].replace(".tif", "_m.tif")
                image = Image.fromarray(pmap*255.0)
                image.save(new_path, 'TIFF')
                new_path = tiff_path[0].replace(".tif", "_r.tif")
                image = Image.fromarray(rmap*255.0)
                image.save(new_path, 'TIFF')



                mean_list.append(((1+mean_of_top(pmap, args.p))**args.w_m) * ((1+mean_of_top(r_admap, args.p))**args.w_r))
                m_mean_list.append(mean_of_top(pmap, args.p))
                r_mean_list.append(mean_of_top(r_admap, args.p))
                pn_mean_list.append(pn[0])

                pn_pixel_list.extend(label.squeeze(0).squeeze(0).cpu().numpy().astype(int).ravel())
                pixel_list.extend(mixmap.ravel())

                pn_label_list.append(label.squeeze(0).squeeze(0).cpu().numpy().astype(int))
                preimg_list.append(mixmap)
                        

            now = datetime.datetime.now()
            now = now.strftime('%y%m%d%H')
            f = open( log_saved_path + now + 'log.txt','a')

            image_auc = roc_auc_score(pn_mean_list, mean_list)
            pixel_auc = roc_auc_score(pn_pixel_list, pixel_list)

            m_image_auc = roc_auc_score(pn_mean_list, m_mean_list)
            r_image_auc = roc_auc_score(pn_mean_list, r_mean_list)

            print(
                "image_auc",  "{:.2f}".format(image_auc*100),  
                "pixel_auc",  "{:.2f}".format(pixel_auc*100), 
                "m_image_auc",  "{:.2f}".format(m_image_auc*100),
                "r_image_auc",  "{:.2f}".format(r_image_auc*100), file=f)
            f.close()


            if best_auc < image_auc:
                best_auc = image_auc               
                f = open( auc_saved_path + 'image_level_result.txt','w')
                best_auc_ = best_auc
                best_auc_ = "{:.2f}".format(best_auc_*100)
                print("image_auc", best_auc_, file=f)
                f.close()   

            if best_pixel_auc < pixel_auc:
                best_pixel_auc = pixel_auc              
                f = open( auc_saved_path + 'pixel_level_result.txt','w')
                best_pixel_auc_ = best_pixel_auc
                best_pixel_auc_ = "{:.2f}".format(best_pixel_auc_*100)
                print("pixel_auc", best_pixel_auc_, file=f)
                f.close()

                torch.save(model_ad.state_dict(),  model_saved_path + 'net_ad.pth')

            if best_m_image_auc < m_image_auc:
                best_m_image_auc = m_image_auc              
                f = open( auc_saved_path + 'best_m_image_auc.txt','w')
                best_m_image_auc_ = best_m_image_auc
                best_m_image_auc_ = "{:.2f}".format(best_m_image_auc_*100)
                print("m_image_auc", best_m_image_auc_, file=f)
                f.close()

            if best_r_image_auc < r_image_auc:
                best_r_image_auc = r_image_auc              
                f = open( auc_saved_path + 'best_r_image_auc.txt','w')
                best_r_image_auc_ = best_r_image_auc
                best_r_image_auc_ = "{:.2f}".format(best_r_image_auc_*100)
                print("r_image_auc", best_r_image_auc_, file=f)
                f.close()
    
    today = time.strftime("%Y-%m-%d", time.localtime())
    f = open( './result/auc/VisA/all_result' + today +'.txt','a')
    best_auc_ = best_auc
    best_auc_ = "{:.2f}".format(best_auc_*100)
    best_pixel_auc_ = best_pixel_auc
    best_pixel_auc_ = "{:.2f}".format(best_pixel_auc_*100)
    print(args.object_name, "image_auc", best_auc_, "pixel_auc", best_pixel_auc_, "best_m_image_auc", best_m_image_auc_, "best_r_image_auc", best_r_image_auc_, file=f)
    f.close()


