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

from util.misc import seed_torch,  mean_of_top

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
    Image_Ham_auc_best = 0
    Image_W_auc_best = 0
    Image_alpha_auc_best = 0
    Pixel_Ham_auc_best = 0
    Pixel_alpha_auc_best = 0
    Pixel_MAX_auc_best = 0



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
            image_gt_list = []
            pixel_gt_list = []
            Pixel_Ham_list = []
            Pixel_alpha_list = []
            Pixel_MAX_list = []
            Image_Ham_list = []
            Image_W_list = []
            Image_alpha_list = []

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


                Pixel_Ham_mixmap = gaussian_filter(mixmap.squeeze(0).cpu(), sigma=4)
                Pixel_alpha_mixmanp= 0.5*pmap + (1-0.5)*rmap
                Pixel_MAX_mixmanp = np.maximum(pmap, rmap) 




                image_gt_list.append(pn[0])
                pixel_gt_list.extend(label.squeeze(0).squeeze(0).cpu().numpy().astype(int).ravel())

                Image_Ham_list.append(np.mean(m_admap * r_admap))
                Image_W_list.append(((1+mean_of_top(pmap, args.p))**args.w_m) * ((1+mean_of_top(r_admap, args.p))**args.w_r))
                Image_alpha_list.append(np.mean(Pixel_alpha_mixmanp))

                Pixel_Ham_list.extend(Pixel_Ham_mixmap.ravel())
                Pixel_alpha_list.extend(Pixel_alpha_mixmanp.ravel())
                Pixel_MAX_list.extend(Pixel_MAX_mixmanp.ravel())

                        

            now = datetime.datetime.now()
            now = now.strftime('%y%m%d%H')
            f = open( log_saved_path + now + 'log.txt','a')

            image_Ham_auc = roc_auc_score(image_gt_list, Image_Ham_list)
            image_W_auc = roc_auc_score(image_gt_list, Image_W_list)
            image_alpha_auc = roc_auc_score(image_gt_list, Image_alpha_list)

            Pixel_Ham_auc = roc_auc_score(pixel_gt_list, Pixel_Ham_list)
            Pixel_alpha_auc = roc_auc_score(pixel_gt_list, Pixel_alpha_list)
            Pixel_MAX_auc = roc_auc_score(pixel_gt_list, Pixel_MAX_list)


            print(
                "image_Ham_auc",  "{:.2f}".format(image_Ham_auc*100),
                "image_W_auc",  "{:.2f}".format(image_W_auc*100),
                "image_alpha_auc",  "{:.2f}".format(image_alpha_auc*100),
                "Pixel_Ham_auc",  "{:.2f}".format(Pixel_Ham_auc*100),
                "Pixel_alpha_auc",  "{:.2f}".format(Pixel_alpha_auc*100),
                "Pixel_MAX_auc",  "{:.2f}".format(Pixel_MAX_auc*100),                
                file=f)
            f.close()

            if Image_Ham_auc_best < image_Ham_auc:
                Image_Ham_auc_best = image_Ham_auc
            if Image_W_auc_best < image_W_auc:
                Image_W_auc_best = image_W_auc
            if Image_alpha_auc_best < image_alpha_auc:
                Image_alpha_auc_best = image_alpha_auc
            if Pixel_Ham_auc_best < Pixel_Ham_auc:
                Pixel_Ham_auc_best = Pixel_Ham_auc
                torch.save(model_ad.state_dict(),  model_saved_path + 'net_ad.pth')
            if Pixel_alpha_auc_best < Pixel_alpha_auc:
                Pixel_alpha_auc_best = Pixel_alpha_auc
            if Pixel_MAX_auc_best < Pixel_MAX_auc:
                Pixel_MAX_auc_best = Pixel_MAX_auc

    Image_Ham_auc_best_ = "{:.2f}".format(Image_Ham_auc_best*100)
    Image_W_auc_best_ = "{:.2f}".format(Image_W_auc_best*100)
    Image_alpha_auc_best_ = "{:.2f}".format(Image_alpha_auc_best*100)
    Pixel_Ham_auc_best_ = "{:.2f}".format(Pixel_Ham_auc_best*100)
    Pixel_alpha_auc_best_ = "{:.2f}".format(Pixel_alpha_auc_best*100)
    Pixel_MAX_auc_best_ = "{:.2f}".format(Pixel_MAX_auc_best*100)
    
    today = time.strftime("%Y-%m-%d", time.localtime())
    f = open( './result/auc/VisA/all_result' + today +'.txt','a')

    print(args.object_name, 
          "Image_Ham_auc_best_", Image_Ham_auc_best_ ,
          "Image_W_auc_best_", Image_W_auc_best_ ,
          "Image_alpha_auc_best_", Image_alpha_auc_best_ ,
          "Pixel_Ham_auc_best_", Pixel_Ham_auc_best_ ,
          "Pixel_alpha_auc_best_", Pixel_alpha_auc_best_ ,
          "Pixel_MAX_auc_best_", Pixel_MAX_auc_best_ ,
           file=f)
    f.close()


