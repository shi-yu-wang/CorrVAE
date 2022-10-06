# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 21:23:16 2021

@author: Shiyu
"""
import argparse
import logging
import sys
import os
import torch
from configparser import ConfigParser
import numpy as np
from torch import optim
from itertools import chain

from encoders import *
from decoders import *
from model import *
# from utils.datasets import get_dataloaders, get_img_size, DATASETS
from utils.datasets import *
from utils.helpers import (create_safe_directory, get_device, set_seed, get_n_param,
                           get_config_section, update_namespace_, FormatterNoDuplicate)

from utils.visualize import GifTraversalsTraining
from torch.utils.data import SubsetRandomSampler, Dataset, DataLoader
import logging
from losses import LOSSES, RECON_DIST, get_loss_f
from main import *
from disvae.Trainer import *
from disvae.Evaluator import *
from disvae.utils.math import (log_density_gaussian, log_importance_weight_matrix,
                               matrix_log_density_gaussian)

import pickle
import time


encoder = eval("EncoderControlvae")
decoder = eval("DecoderControlvae")

################################# Parameters ##################################
args = parse_arguments("control_dsprites -d dsprites --lr 0.0005 -e 61 -b 64 -l btcvae_property -num_prop 3".split())

device = get_device(is_gpu=not args.no_cuda)
RES_DIR = "results"
exp_dir = os.path.join(RES_DIR, "CVAE")

dataset_size=480000
indices=list(range(dataset_size))
split=390000
np.random.shuffle(indices)
train_indices, test_indices=indices[:split],indices[split:]
train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)

epochs = 100
batch_size = 64

formatter = logging.Formatter('%(asctime)s %(levelname)s - %(funcName)s: %(message)s',
                                  "%H:%M:%S")
logger = logging.getLogger(__name__)
logger.setLevel("INFO")
stream = logging.StreamHandler()
stream.setLevel("INFO")
stream.setFormatter(formatter)
logger.addHandler(stream)
logger.info("Root directory for saving and loading experiments: {}".format(exp_dir))

beta = 1
taus = 0.2

args.img_size = get_img_size(args.dataset)

#################################### Model ####################################
model = ControlVAE(args.img_size, encoder, decoder, args.latent_dim, args.num_prop)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
# lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
#         optimizer,
#         T_max = 32,
#         verbose=True)
lr_scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[10, 20, 30, 50, 70, 100, 130, 160, 200],
        gamma=0.1)

model = model.to(device)

################################### Functions ##################################
def _kl_normal_loss(mean, logvar):
    latent_dim = mean.size(1)
    # batch mean of kl for each latent dimension
    latent_kl = 0.5 * (-1 - logvar + mean.pow(2) + logvar.exp()).mean(dim=0)
    total_kl = latent_kl.sum()

    return total_kl

def _get_log_pz_qz_prodzi_qzCx(latent_sample, latent_dist, n_data, is_mss=True):
    batch_size, hidden_dim = latent_sample.shape

    # calculate log q(z|x)
    log_q_zCx = log_density_gaussian(latent_sample, *latent_dist).sum(dim=1)

    # calculate log p(z)
    # mean and log var is 0
    zeros = torch.zeros_like(latent_sample)
    log_pz = log_density_gaussian(latent_sample, zeros, zeros).sum(1)

    mat_log_qz = matrix_log_density_gaussian(latent_sample, *latent_dist)


    log_qz = torch.logsumexp(mat_log_qz.sum(2), dim=1, keepdim=False)
    log_prod_qzi = torch.logsumexp(mat_log_qz, dim=1, keepdim=False).sum(1)

    return log_pz, log_qz, log_prod_qzi, log_q_zCx

def _get_log_pzw_qzw_prodzw_qzwCx(latent_sample_z,latent_sample_w, latent_dist_z, latent_dist_w,n_data, is_mss=True):
    batch_size, hidden_dim_z = latent_sample_z.shape
    batch_size, hidden_dim_w = latent_sample_w.shape
    hidden_dim=hidden_dim_z+hidden_dim_w
    latent_dist=(torch.cat([latent_dist_z[0],latent_dist_w[0]],dim=-1), torch.cat([latent_dist_z[1],latent_dist_w[1]],dim=-1))
    latent_sample=torch.cat([latent_sample_z,latent_sample_w],dim=-1)
    
    # calculate log q(z,w|x)
    log_q_zwCx = log_density_gaussian(latent_sample, *latent_dist).sum(dim=1)

    # calculate log p(z,w)
    # mean and log var is 0
    zeros = torch.zeros_like(latent_sample)
    log_pzw = log_density_gaussian(latent_sample, zeros, zeros).sum(1)

    mat_log_qzqw = matrix_log_density_gaussian(latent_sample, *latent_dist)
    mat_log_qz = matrix_log_density_gaussian(latent_sample_z, *latent_dist_z)
    mat_log_qw = matrix_log_density_gaussian(latent_sample_w, *latent_dist_w)

    if is_mss:
        # use stratification
        log_iw_mat = log_importance_weight_matrix(batch_size, n_data).to(latent_sample.device)
        mat_log_qzqw = mat_log_qzqw + log_iw_mat.view(batch_size, batch_size, 1)
        log_iw_mat_z = log_importance_weight_matrix(batch_size, n_data).to(latent_sample_z.device)
        mat_log_qz = mat_log_qz + log_iw_mat_z.view(batch_size, batch_size, 1)        
        log_iw_mat_w = log_importance_weight_matrix(batch_size, n_data).to(latent_sample_w.device)
        mat_log_qw = mat_log_qw + log_iw_mat_w.view(batch_size, batch_size, 1)

    log_qzw = torch.logsumexp(mat_log_qzqw.sum(2), dim=1, keepdim=False)
    log_qz = torch.logsumexp(mat_log_qz.sum(2), dim=1, keepdim=False)
    log_qw = torch.logsumexp(mat_log_qw.sum(2), dim=1, keepdim=False)
    log_prod_qzqw = log_qz + log_qw

    return log_pzw, log_qzw, log_prod_qzqw, log_q_zwCx


def loss_function(mse_loss, reconstruct, y_reconstruct, data, label, batch_size, num_prop,
                  latent_dist):
    # Reconstruction loss
    rec_loss = F.mse_loss(reconstruct * 255, data * 255, reduction="sum") / 255
    rec_loss = rec_loss / batch_size
    rec_loss_prop=[]
    for idx in range(num_prop):
        rec_loss_prop.append(mse_loss(y_reconstruct[:,idx],label[:,idx].float().cuda()))
        # rec_loss_prop.append(torch.nn.MSELoss(recon_batch[1][:,idx],label[:,idx].float(), reduction="sum"))
    
    rec_loss_prop_all= sum(rec_loss_prop)  
    
    # KL loss
    kl_loss = _kl_normal_loss(*latent_dist)


def reg_mask(mask):
    l1norm = torch.sum(torch.abs(mask))
    
    return l1norm

mse_loss = torch.nn.MSELoss(reduction="sum")

#################################### Train ####################################
train_loader = get_dataloaders(args.dataset,train_sampler,batch_size=args.batch_size,logger=logger)
model.train()


recon_loss_prop_rec = []
recon_loss_rec = []
kl_loss_rec = []
pwwi_loss_rec = []
pwz_loss_rec = []
l1_loss_rec = []
mask_rec = []

idx_kl = 0
w_kl = 100
start_epoch = time.time()


prop_det = []
w_det = []
mask_ori_det = []
w_mask_det = []
# wp_det = []
for epoch in range(epochs):
    if (epoch + 1) % 10 == 0:
        taus = taus * 0.1
    epoch_loss = 0
    lr_scheduler.step()
    i = 0
    start = time.time()
    
    for _, (data, label) in enumerate(train_loader):
        i += 1
        idx_kl += 1
        batch_size, channel, height, width = data.size()
        data = data.to(device)
        
        (reconstruct,y_reconstruct), latent_dist_z, latent_dist_w, latent_sample_z,latent_sample_w, w_mask, mask_ori  = model(data, taus)
        # y_reconstruct, latent_dist_z, latent_dist_w, latent_sample_z,latent_sample_w,p_pred = model(data)
        prop_det.append(y_reconstruct[:,0].detach().cpu().numpy())
        w_det.append(latent_sample_w.detach().cpu().numpy())
        mask_ori_det.append(mask_ori.detach().cpu().numpy())
        w_mask_det.append(w_mask.detach().cpu().numpy())

        # Loss
        latent_sample=torch.cat([latent_sample_w,latent_sample_z],dim=-1)
        batch_size, latent_dim = latent_sample.shape
        latent_dist=(torch.cat([latent_dist_w[0], latent_dist_z[0]],dim=-1),torch.cat([latent_dist_w[1], latent_dist_z[1]],dim=-1))
        num_prop = args.num_prop
        
        ###### Reconstruction loss ######
        rec_loss = F.binary_cross_entropy(reconstruct, data, reduction="sum") / 64
        rec_loss = rec_loss / batch_size
        rec_loss_prop=[]

        rec_loss_prop.append(mse_loss(y_reconstruct[:,0],label[:,2].float().cuda()))
        rec_loss_prop.append(mse_loss(y_reconstruct[:,1],label[:,4].float().cuda()))
        rec_loss_prop.append(mse_loss(y_reconstruct[:,2],label[:,6].float().cuda()))
        rec_loss_prop_all= sum(rec_loss_prop)
        
        ###### KL loss ######
        kl_loss = _kl_normal_loss(*latent_dist)
        
        log_pw, log_qw, log_prod_qwi, log_q_wCx = _get_log_pz_qz_prodzi_qzCx(latent_sample_w,
                                                                         latent_dist_w,
                                                                         len(train_loader.dataset),
                                                                         is_mss=True)
        
        # mi_loss = (log_q_wCx - log_qw).mean()
        tc_loss = (log_qw - log_prod_qwi).mean()
        # dw_kl_loss = (log_prod_qwi - log_pw).mean()
        # anneal_reg = (linear_annealing(0, 1, self.n_train_steps, self.steps_anneal) if is_train else 1)    
        
        pairwise_tc_loss = beta * tc_loss 
        
        log_pwz, log_qwz, log_prod_qwqz, log_q_wzCx = _get_log_pzw_qzw_prodzw_qzwCx(latent_sample_z,
                                                                             latent_sample_w,  
                                                                             latent_dist_z,
                                                                             latent_dist_w,
                                                                             len(train_loader.dataset),
                                                                             is_mss=True)
        groupwise_tc_loss = beta * (log_qwz - log_prod_qwqz).mean() 
        
        l1norm = reg_mask(w_mask).cuda()
    
        if idx_kl <= 100000:
        # if idx_kl <= 100:
            # loss = 1000000*rec_loss + pairwise_tc_loss + 100000*rec_loss_prop_all + groupwise_tc_loss
            loss = 1000000*rec_loss + pairwise_tc_loss + 1000000*rec_loss_prop_all + groupwise_tc_loss + 100000*l1norm
        else:
            if w_kl < 100000:
                w_kl += 1
            loss = 1000000*rec_loss + pairwise_tc_loss + 1000000*rec_loss_prop_all + groupwise_tc_loss + w_kl * kl_loss
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss
        
        # if i == 1:
        #     break
        if i % 50 == 0:
            print("Epoch ", epoch, ", Iteration ", i, ", Total loss ", loss, ", KL loss ", kl_loss, ", Rec loss ", rec_loss, ", Rec prop loss ", rec_loss_prop_all, ", wwi loss ", pairwise_tc_loss, ", wz loss ", groupwise_tc_loss, ", l1 norm ", l1norm, sep = "")
            print("tau: ", taus)
            print("Time: ", time.time() - start)
            print(torch.sum(w_mask))
            if idx_kl <= 100000:
                print("Reduce L1 norm")
            print("===============================================================")
            start = time.time()
            
            recon_loss_prop_rec.append(rec_loss_prop_all.detach().cpu().numpy())
            recon_loss_rec.append(rec_loss.detach().cpu().numpy())
            kl_loss_rec.append(kl_loss.detach().cpu().numpy())
            pwwi_loss_rec.append(pairwise_tc_loss.detach().cpu().numpy())
            pwz_loss_rec.append(groupwise_tc_loss.detach().cpu().numpy())
            l1_loss_rec.append(l1norm.detach().cpu().numpy())
            mask_rec.append(torch.sum(w_mask).detach().cpu().numpy())
            
            torch.save({'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        }, "modelCorrVAE.pt")

        
    mean_epoch_loss = epoch_loss / len(train_loader)
    epoch_loss = 0
    print("===============================================================")
    print("Completed Epoch", epoch, ", Totla loss: ", mean_epoch_loss, ", Time: ", time.time() - start_epoch)
    print("===============================================================")
    start_epoch = time.time()
 