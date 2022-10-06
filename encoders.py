"""
Module containing the encoders.
"""
import numpy as np

import torch
from torch import nn
import torch.functional as F

# ALL encoders should be called Enccoder<Model>
def get_encoder(model_type):
    model_type = model_type.lower().capitalize()
    return eval("Encoder{}".format(model_type))
       
class EncoderControlvae(nn.Module):
    def __init__(self, img_size,
                 latent_dim=10,num_prop=2,if_given_property=False):
        """Encoder of the model proposed in [1].

        Parameters
        ----------
        img_size : tuple of ints (channel, hight, width)
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
        super(EncoderControlvae, self).__init__()
        self.if_given_property=if_given_property
        # Layer parameters
        hid_channels = 32
        kernel_size = 4
        hidden_dim = 256
        hidden_dim_prop=50
        self.latent_dim = latent_dim
        self.img_size = img_size
        # Shape required to start transpose convs
        self.reshape = (hid_channels, kernel_size, kernel_size)
        n_chan = self.img_size[0]
        self.num_prop=num_prop
        # 3 convolutional layers
        cnn_kwargs = dict(stride=2, padding=1)
        self.conv1 = nn.Conv2d(n_chan, hid_channels, kernel_size, **cnn_kwargs).cuda()
        self.conv2 = nn.Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs).cuda()
        self.conv3 = nn.Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs).cuda()

        # If input image is 64x64 do fourth convolution
        if self.img_size[1] == self.img_size[2] == 64:
            self.conv_64 = nn.Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs).cuda()


        # # Fully connected layers for property prediction
        # self.prop_lin1_list = []
        # self.prop_lin2_list = []
        # self.prop_lin3_list=[]
        # # Fully connected layers for mean and variance
        # self.prop_mu_logvar_list=[]  
        # self.sigmoid=torch.nn.Sigmoid().to('cuda')
        # for idx in range(num_prop):
        #    self.prop_lin1_list.append(nn.Linear(np.product(self.reshape), hidden_dim_prop).to('cuda')) 
        #    self.prop_lin2_list.append(nn.Linear(hidden_dim_prop, 1).to('cuda')) 
        #    self.prop_lin3_list.append(nn.Linear(hidden_dim,hidden_dim_prop).to('cuda'))       
        #    self.prop_mu_logvar_list.append(nn.Linear(hidden_dim_prop, 2).to('cuda'))
 
        
        # Fully connected layers for unobversed properties
        self.lin1 = nn.Linear(np.product(self.reshape), hidden_dim).cuda()
        self.lin2 = nn.Linear(hidden_dim, hidden_dim).cuda()
        # Fully connected layers for mean and variance
        self.mu_logvar_gen = nn.Linear(hidden_dim, (self.latent_dim+self.num_prop) * 2).cuda()           

    def forward(self, x, label, prop=None):
        
        batch_size = x.size(0)
        # print(x)
        # embed image into z
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        if self.img_size[1] == self.img_size[2] == 64:
            x = torch.relu(self.conv_64(x))        

        x_z = x.view((batch_size, -1))

        x_z = torch.relu(self.lin1(x_z))
        x_z = torch.relu(self.lin2(x_z))        
        
        mu_logvar = self.mu_logvar_gen(x_z)
        mu, logvar = mu_logvar.view(-1, self.latent_dim+self.num_prop, 2).unbind(-1)

        return mu[:,:self.latent_dim],mu[:,self.latent_dim:], logvar[:,:self.latent_dim], logvar[:,self.latent_dim:]

     