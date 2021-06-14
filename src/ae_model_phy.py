"""
This python file contains the Autoencoder models as classes
per model. Architectures include linear, convolution, transpose
convolution, upampling, and ResNet type of NN/layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class Forward_AE(nn.Module):

    def __init__(self, img_dim=28, dropout=.2, in_ch=1, phy_dim=8):
        """
        Parameters
        ----------
        img_dim    : int
            image size, only one dimension, assuming square ratio.
        dropout    : float
            dropout probability
        in_ch      : int
            number of channels in input/output image
        """
        super(Forward_AE, self).__init__()
        self.img_width = self.img_height = img_dim
        self.img_size = self.img_width * self.img_height
        self.in_ch = in_ch
        self.phy_dim = phy_dim

        # Decoder specification
        self.dec_linear = nn.Sequential(
            nn.Linear(phy_dim,
                      128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 16 * 4 * 4, bias=False),
            nn.BatchNorm1d(16 * 4 * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16 * 4 * 4, 16 * 8 * 8, bias=False),
            nn.BatchNorm1d(16 * 8 * 8),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16 * 8 * 8, 16 * 16 * 16, bias=False),
            nn.ReLU(),
        )
        self.dec_transconv = nn.Sequential(
            nn.ConvTranspose2d(16, 16, 4, stride=2, bias=False,
                               output_padding=1, padding=0),
            nn.Conv2d(16, 16, 4, bias=False),
            nn.BatchNorm2d(16, momentum=0.005),
            nn.ReLU(),
            nn.Conv2d(16, 8, 4, bias=False),
            nn.BatchNorm2d(8, momentum=0.005),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 8, 4, stride=2, bias=False,
                               output_padding=1, padding=0),
            nn.Conv2d(8, 8, 4, bias=False),
            nn.BatchNorm2d(8, momentum=0.005),
            nn.ReLU(),
            nn.Conv2d(8, 4, 4, bias=False),
            nn.BatchNorm2d(4, momentum=0.005),
            nn.ReLU(),
            nn.ConvTranspose2d(4, 4, 4, stride=2, bias=False,
                               output_padding=1, padding=0),
            nn.Conv2d(4, 4, 4, bias=False),
            nn.BatchNorm2d(4, momentum=0.005),
            nn.ReLU(),
            nn.Conv2d(4, 4, 4, bias=False),
            nn.BatchNorm2d(4, momentum=0.005),
            nn.ReLU(),
            nn.ConvTranspose2d(4, 4, 4, stride=2, bias=False,
                               output_padding=1, padding=0),
            nn.Conv2d(4, 4, 4, bias=False),
            nn.BatchNorm2d(4, momentum=0.005),
            nn.ReLU(),
            nn.Conv2d(4, in_ch, 7),
            nn.Sigmoid()
        )

    def forward(self, phy):
        """
        Parameters
        ----------
        z : tensor
            latent code [N, latent_dim]
        Returns
        -------
            reconstructed image [N, C, H, W]
        """
        z = self.dec_linear(phy)
        z = z.view(-1, 16, 16, 16)
        z = self.dec_transconv(z)

        z = F.interpolate(z, size=(self.img_width, self.img_height),
                          mode='nearest')
        return z
  

class Dev_Forward_AE(nn.Module):

    def __init__(self, img_dim=28, dropout=.2, in_ch=1, phy_dim=8):
        """
        Parameters
        ----------
        img_dim    : int
            image size, only one dimension, assuming square ratio.
        dropout    : float
            dropout probability
        in_ch      : int
            number of channels in input/output image
        """
        super(Dev_Forward_AE, self).__init__()
        self.img_width = self.img_height = img_dim
        self.img_size = self.img_width * self.img_height
        self.in_ch = in_ch
        self.phy_dim = phy_dim

        # Linear layers
        self.dec_linear = nn.Sequential(
            nn.Linear(phy_dim,
                      128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(128, 16 * 4 * 4, bias=False),
            nn.BatchNorm1d(16 * 4 * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(16 * 4 * 4, 16 * 8 * 8, bias=False),
            nn.BatchNorm1d(16 * 8 * 8),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(16 * 8 * 8, 16 * 16 * 8, bias=False),
            nn.BatchNorm1d(16*16*8),
            nn.ReLU(),
            nn.Dropout(dropout),
            #Last linear layer
            nn.Linear(16 * 16 * 8, 16 * 16 * 16, bias=False), #double check math on output layer
            nn.ReLU()
        )
        
        #convolutional layers
        self.dec_transconv = nn.Sequential(
            nn.ConvTranspose2d(16, 16, 4, stride=2, bias=False,
                               output_padding=1, padding=0),
            nn.Conv2d(16, 16, 4, bias=False),
            nn.BatchNorm2d(16, momentum=0.005),
            nn.ReLU(),
            
            nn.Conv2d(16, 8, 4, bias=False),
            nn.BatchNorm2d(8, momentum=0.005),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 8, 4, stride=2, bias=False,
                               output_padding=1, padding=0),
            
            nn.Conv2d(8, 8, 4, bias=False),
            nn.BatchNorm2d(8, momentum=0.005),
            nn.ReLU(),
            
            nn.Conv2d(8, 4, 4, bias=False),
            nn.BatchNorm2d(4, momentum=0.005),
            nn.ReLU(),
            ###output layer
            nn.ConvTranspose2d(4, 4, 4, stride=2, bias=False,
                               output_padding=1, padding=0),
            nn.Conv2d(4, in_ch, 7),
            nn.Sigmoid()
        )

    def forward(self, phy):
        """
        Parameters
        ----------
        z : tensor
            latent code [N, latent_dim]
        Returns
        -------
            reconstructed image [N, C, H, W]
        """
        z = self.dec_linear(phy)
        z = z.view(-1, 16, 16, 16)
        z = self.dec_transconv(z)

        z = F.interpolate(z, size=(self.img_width, self.img_height),
                          mode='nearest')
        return z
