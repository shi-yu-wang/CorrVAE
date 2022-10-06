# -*- coding: utf-8 -*-
"""
Created on Sat Jan 15 13:06:13 2022

@author: Shiyu
"""

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
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pylab as plt
import matplotlib.image as mpimg

from optim import *
from evaluate import *

encoder = eval("EncoderControlvae")
decoder = eval("DecoderControlvae")

################################# Parameters ##################################
args = parse_arguments("control_dsprites -d dsprites --lr 0.0005 -e 61 -b 64 -l btcvae_property -num_prop 3".split())
# no_cuda = False
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

epochs = 100 * 2
batch_size = 64 * 2

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

args.img_size = get_img_size(args.dataset)

model = ControlVAE(args.img_size, encoder, decoder, args.latent_dim, args.num_prop)
model.load_state_dict(torch.load("D:/Research/Project4MultiObjOpt/Program/Newtesting/new/PCVAE v3 learn mask as parameter binary low loss10-9 v2 3prop indep v3 mask mse cor/modelPCVAE8.pt")['model_state_dict'])
model.eval()

test_loader = get_dataloaders(args.dataset,test_sampler,batch_size=args.batch_size,logger=logger)

mse_loss = torch.nn.MSELoss(reduction="sum")

scale_pred = []
x_pred = []
xy_pred = []
############################### Prediction MSE
i = 0
for _, (data, label) in enumerate(test_loader):
    i += 1
    if i % 50 == 0:
        print(i)
    
    batch_size, channel, height, width = data.size()
    data = data.to(device)
    
    (reconstruct,y_reconstruct), latent_dist_z, latent_dist_w, latent_sample_z,latent_sample_w, w_mask, mask_ori  = model(data, 2.0000000000000012e-17)
    
    scale_pred.append(mse_loss(y_reconstruct[:,0],label[:,2].float().cuda()).detach().cpu().numpy() / 16)
    x_pred.append(mse_loss(y_reconstruct[:,1],label[:,4].float().cuda()).detach().cpu().numpy() / 16)
    xy_pred.append(mse_loss(y_reconstruct[:,2],label[:,6].float().cuda()).detach().cpu().numpy() / 16)

sum(scale_pred) / len(scale_pred)
sum(x_pred) / len(x_pred)
sum(xy_pred) / len(xy_pred)
########################################## Mask
mask = mask/1000
mask[mask <= 0.5] = 0
mask[mask > 0.5] = 1
mask

########################################## Optimization
muloptim = Optimization(dim_w = args.latent_dim, model = model, num_prop = args.num_prop)

optimizer = optim.SGD(muloptim.parameters(), lr=1e-1)
lr_scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[3000, 8000, 15000, 30000, 40000, 70000, 90000],
        gamma=0.1)

prop = torch.tensor([0.3, 0.8, 1.5]).cuda()

loss_recon = []

epochs = 100000
start = time.time()
for epoch in range(epochs):
    lr_scheduler.step()
    
    prop_pred, loss_range1, loss_value1, loss_value2, loss_inf, lambda1, lambda2, lambda3, lambda4, w = muloptim(prop, mask)
    
    # if epoch <= 20000:
    loss = 1000*loss_value1
    # else:
    #     loss = 100*loss_value2 + 0.01 * loss_range1
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    with torch.no_grad():
        muloptim.lambda1.clamp_(0, 1)
        muloptim.lambda2.clamp_(0, 1)
        muloptim.lambda3.clamp_(0, 1)
        muloptim.lambda4.clamp_(0, 1)

    if (epoch+1) % 100 == 0:
        print("epoch: ", epoch, "loss: ", loss, "time: ", time.time() - start)
        print("loss range: ", loss_range1, "loss value: ", loss_value2, "loss_inf: ", loss_inf)
        print("lambda1: ", lambda1, "lambda2: ", lambda2, "lambda3: ", lambda3, "lambda4: ", lambda4)
        print("Property: ", prop_pred)
        print("w: ", w)
        print(lr_scheduler.get_last_lr())
    
        start = time.time()
        loss_recon.append(loss.detach().cpu().numpy())
        torch.save({'model_state_dict': muloptim.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        }, "D:/Research/Project4MultiObjOpt/Program/Newtesting/new/PCVAE v3 learn mask as parameter binary low loss10-9 v2 3prop indep v3 mask mse cor/optimwt.pt")

