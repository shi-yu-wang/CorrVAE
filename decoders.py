"""
Module containing the decoders.
"""
import numpy as np

import torch
from torch import nn
from spectral_norm_fc import spectral_norm_fc


# ALL decoders should be called Decoder<Model>
def get_decoder(model_type):
    model_type = model_type.lower().capitalize()
    if model_type=='Csvae':
        return eval("Decoder{}X".format(model_type)),eval("Decoder{}Y".format(model_type))
    else:
        return eval("Decoder{}".format(model_type))


class DecoderControlvae(nn.Module):
    def __init__(self, img_size,
                 latent_dim_z=10, latent_dim_w = 10, num_prop=2):
        r"""Decoder of the model proposed in [1].
        Parameters
        ----------
        img_size : tuple of ints
            Size of images. E.g. (1, 32, 32) or (3, 64, 64).
        latent_dim : int
            Dimensionality of latent output.
        Model Architecture (transposed for decoder)
        ------------
        - 4 convolutional layers (each with 32 channels), (4 x 4 kernel), (stride of 2)
        - 2 fully connected layers (each of 256 units)
        - Latent distribution:
            - 1 fully connected layer of 20 units (log variance and mean for 10 Gaussians)
        References:
            [1] Burgess, Christopher P., et al. "Understanding disentangling in
            $\beta$-VAE." arXiv preprint arXiv:1804.03599 (2018).
        """
        super(DecoderControlvae, self).__init__()

        # Layer parameters
        hid_channels = 32
        kernel_size = 4
        # hidden_dim = 256
        hidden_dim = 512
        # hidden_dim_prop=50
        hidden_dim_prop=512
        self.img_size = img_size
        # Shape required to start transpose convs
        self.reshape = (hid_channels, kernel_size, kernel_size)
        n_chan = self.img_size[0]
        self.img_size = img_size
        self.num_prop=num_prop
        # self.latent_dim=latent_dim_z+num_prop
        self.latent_dim_z=latent_dim_z
        self.latent_dim_w = latent_dim_w
        self.sigmoid=torch.nn.Sigmoid()
        
        # decoder for the property 
        self.property_lin_list=nn.ModuleList()
        for idx in range(num_prop):
            layers=[]
            layers.append(spectral_norm_fc(nn.Linear(1, hidden_dim_prop).to('cuda')))
            layers.append(nn.ReLU())
            layers.append(spectral_norm_fc(nn.Linear(hidden_dim_prop, 1).to('cuda')))
            if num_prop-idx==4:#if deaing with proprty 0-2pi
               layers.append(nn.ReLU())
            else:
               layers.append(nn.Sigmoid())
            self.property_lin_list.append(nn.Sequential(*layers))
        
        # self.hidden_w = 64
        self.hidden_w = 512
        self.wp_lin_list=nn.ModuleList()
        for idx in range(num_prop):
            layers=nn.Sequential(
                nn.Linear(self.latent_dim_w, self.hidden_w),
                nn.ReLU(),
                nn.Linear(self.hidden_w, self.hidden_w),
                nn.ReLU(),
                nn.Linear(self.hidden_w, 1)
                ).cuda()
            self.wp_lin_list.append(nn.Sequential(*layers))
        
        
        # Fully connected layers
        self.lin1 = nn.Linear(self.latent_dim_z + self.latent_dim_w, hidden_dim).cuda()
        self.lin2 = nn.Linear(hidden_dim, hidden_dim).cuda()
        self.lin3 = nn.Linear(hidden_dim, np.product(self.reshape)).cuda()

        # Convolutional layers
        cnn_kwargs = dict(stride=2, padding=1)
        # If input image is 64x64 do fourth convolution
        if self.img_size[1] == self.img_size[2] == 64:
            self.convT_64 = nn.ConvTranspose2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs).cuda()

        self.convT1 = nn.ConvTranspose2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs).cuda()
        self.convT2 = nn.ConvTranspose2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs).cuda()
        self.convT3 = nn.ConvTranspose2d(hid_channels, n_chan, kernel_size, **cnn_kwargs).cuda()

    def mask(self, w, w_mask, batch_size):

        w = w.view(w.shape[0], 1, -1)
        w = w.repeat(1, self.num_prop, 1)
        w = w * w_mask
        
        wp = []
        for idx in range(self.num_prop):
            wp.append(self.wp_lin_list[idx](w[:, idx, :].cuda()))

        return torch.cat(wp,dim=-1)
    
    def forward(self, z,w, w_mask):
        batch_size = z.size(0)
        wz=torch.cat([w,z],dim=-1)
        prop=[]

        wp = self.mask(w, w_mask, batch_size)
        
        #fully connected process for reconstruct the properties
        for idx in range(self.num_prop):
            w_=wp[:,idx].view(-1,1)
            prop.append(self.property_lin_list[idx](w_)+w_)
                
        # Fully connected layers with ReLu activations
        x = torch.relu(self.lin1(wz))
        x = torch.relu(self.lin2(x))
        x = torch.relu(self.lin3(x))
        x = x.view(batch_size, *self.reshape)

        # Convolutional layers with ReLu activations
        if self.img_size[1] == self.img_size[2] == 64:
            x = torch.relu(self.convT_64(x))
        x = torch.relu(self.convT1(x))
        x = torch.relu(self.convT2(x))
        # Sigmoid activation for final conv layer
        x = torch.sigmoid(self.convT3(x))

        return x, torch.cat(prop,dim=-1),wp
        
        # return torch.cat(prop,dim=-1)
 
