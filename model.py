# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 21:17:20 2021

@author: Shiyu
"""
import torch
from torch import nn, optim
from torch.nn import functional as F
from disvae.utils.initialization import weights_init
# from torch.distributions.gumbel import Gumbel

class ControlVAE(nn.Module):
    def __init__(self, img_size, encoder, decoder, latent_dim, num_prop):
        """
        Class which defines model and forward pass.

        Parameters
        ----------
        img_size : tuple of ints
            Size of images. E.g. (1, 32, 32) or (3, 64, 64).
        encoder: encoder
        decoder: decoder
        latent_dim: latent dimension
        num_prop: number of properties
        """
        super(ControlVAE, self).__init__()

        if list(img_size[1:]) not in [[32, 32], [64, 64]]:
            raise RuntimeError("{} sized images not supported. Only (None, 32, 32) and (None, 64, 64) supported. Build your own architecture or reshape images!".format(img_size))
        
        # Number of properties
        self.num_prop=num_prop
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.num_pixels = self.img_size[1] * self.img_size[2]
        self.encoder = encoder(img_size, self.latent_dim, self.latent_dim)
        self.decoder = decoder(img_size, self.latent_dim, self.latent_dim, self.num_prop)

        self.reset_parameters()
        
        self.w_mask = torch.nn.Parameter(torch.randn(self.num_prop, self.latent_dim, 2))
        # self.w_mask = torch.eye(3).cuda()
    def reparameterize(self, mean, logvar):
        """
        Samples from a normal distribution using the reparameterization trick.

        Parameters
        ----------
        mean : torch.Tensor
            Mean of the normal distribution. Shape (batch_size, latent_dim)

        logvar : torch.Tensor
            Diagonal log variance of the normal distribution. Shape (batch_size,
            latent_dim)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + std * eps


    def forward(self, x, tau, mask = None, w2=None, w_mask=None, label=None):
        """
        Forward pass of model.

        Parameters
        ----------
        x : torch.Tensor
            Batch of data. Shape (batch_size, n_chan, height, width)
        """

        # if self.training:
        latent_dist_z_mean,latent_dist_w_mean,latent_dist_z_std,latent_dist_w_std = self.encoder(x,label) #for training process
        # else:
            # latent_dist_z_mean,latent_dist_w_mean,latent_dist_z_std,latent_dist_w_std,p_pred = self.encoder(x,label) #for testing process
        
        latent_sample_z = self.reparameterize(latent_dist_z_mean,latent_dist_z_std)
        latent_sample_w = self.reparameterize(latent_dist_w_mean,latent_dist_w_std)

        if w2 != None:
            latent_sample_w = w2.repeat(latent_sample_z.shape[0], 1)

        if mask == None:
            logit = torch.sigmoid(self.w_mask) / (1 - torch.sigmoid(self.w_mask))
            mask = F.gumbel_softmax(logit.cuda(), tau, hard=True)[:, :, 1]

        reconstruct,y_reconstruct,wp = self.decoder(latent_sample_z,latent_sample_w,mask)

        latent_dist_z=(latent_dist_z_mean,latent_dist_z_std)
        latent_dist_w=(latent_dist_w_mean,latent_dist_w_std)
        return (reconstruct,y_reconstruct), latent_dist_z, latent_dist_w, latent_sample_z,latent_sample_w, mask, self.w_mask

    def reset_parameters(self):
        self.apply(weights_init)

    def sample_latent(self, x,p=None):
        """
        Returns a sample from the latent distribution.

        Parameters
        ----------
        x : torch.Tensor
            Batch of data. Shape (batch_size, n_chan, height, width)
        """
        latent_dist_z_mean,latent_dist_w_mean,latent_dist_z_std,latent_dist_w_std,p_pred = self.encoder(x,p)
        latent_dist_z=(latent_dist_z_mean,latent_dist_z_std)
        latent_dist_w=(latent_dist_w_mean,latent_dist_w_std)        
        latent_sample_z = self.reparameterize(*latent_dist_z)
        latent_sample_w = self.reparameterize(*latent_dist_w)
        
        return latent_sample_z, latent_sample_w
    
    def iterate_get_w(self,label,w_latent_idx, maxIter=20):
        #get the w for a kind of given property
        w_n=label.view(-1,1).to('cuda').float()#[N]
        for iter_index in range(maxIter):      
               summand = self.decoder.property_lin_list[w_latent_idx](w_n)
               w_n1 = label.view(-1,1).to('cuda').float() - summand
               print('Iteration of difference:'+str(torch.abs(w_n-w_n1).mean().item()))
               w_n=w_n1.clone()
        return w_n1.view(-1)  
