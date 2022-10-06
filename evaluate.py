# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 15:27:59 2022

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

import math
from functools import reduce
from collections import defaultdict
import json
from timeit import default_timer

from tqdm import trange, tqdm
import numpy as np
import torch
import sklearn
# from disvae.models.losses import get_loss_f
from disvae.utils.math import log_density_gaussian
from disvae.utils.modelIO import save_metadata

TEST_LOSSES_FILE = "test_losses.log"
METRICS_FILENAME = "metrics.log"
METRIC_HELPERS_FILE = "metric_helpers.pth"


def avgMI(metric):
    I_mat=np.diag((1,1,1))
    MI_mat=np.concatenate((metric[-3:][:,2].reshape(3,1),metric[-3:][:,-2:]),axis=1)
    return np.linalg.norm(I_mat - metric,ord=2)

def scoreReg(predicY,testY):
    MSE=np.sum(np.power((testY.reshape(-1,1) - predicY),2))/len(testY)
    R2=1-MSE/np.var(testY)
    return MSE, R2

def calc_MI(X,Y,bins=10):
    c_XY = np.histogram2d(X,Y,bins)[0]
    c_X = np.histogram(X,bins)[0]
    c_Y = np.histogram(Y,bins)[0]

    H_X = shan_entropy(c_X)
    H_Y = shan_entropy(c_Y)
    H_XY = shan_entropy(c_XY)

    MI = H_X + H_Y - H_XY
    return 2*MI/(H_X + H_Y)

def shan_entropy(c):
    c_normalized = c / float(np.sum(c))
    c_normalized = c_normalized[np.nonzero(c_normalized)]
    H = -sum(c_normalized* np.log2(c_normalized))  
    return H

def compute_avgMI2(model, dataloader):
    # all_score=np.zeros((model.latent_dim+model.num_prop, 3))
    all_score=np.zeros((model.latent_dim+model.latent_dim, 3))
    latent=[]
    labels=[]
    with torch.no_grad():
        for x, label in dataloader:
            z_mean,w_mean,z_std,w_std= model.encoder(x.cuda(), label.cuda()) 
            mean=torch.cat([w_mean, z_mean],dim=-1)                                     
            latent.append(mean)
            labels.append(label[:, [2, 4, 6]])
    latent=torch.cat(latent,dim=0).cpu().numpy()
    labels=torch.cat(labels,dim=0).cpu().numpy()
    for z_idx in range(latent.shape[1]):
        for y_idx in range(3):
            score=calc_MI(labels[:,y_idx],latent[:,z_idx])
            print(score)
            all_score[z_idx,y_idx]=score
    np.save('score.npy',all_score)
    return all_score

