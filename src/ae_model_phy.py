"""
This python file contains the Autoencoder models as classes
per model. Architectures include linear, convolution, transpose
convolution, upampling, and ResNet type of NN/layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class Upsampling_model(nn.Module):
    ### NOTE: after extensive testing, we decided that upsampling does not produce
    ###     accurate images nor does it learn as fast. This model is no longer used.

    def __init__(self, img_dim=28, dropout=.2, in_ch=1, phy_dim=8, stride=2, kernel_size=4):
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
        super(Upsampling_model, self).__init__()
        self.img_width = self.img_height = img_dim
        #self.img_size = self.img_width * self.img_height
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

            nn.Linear(16 * 8 * 8, 16 * 12 * 12, bias=False),
            nn.BatchNorm1d(16*12*12),
            nn.ReLU(),
            nn.Dropout(dropout),

            #Last linear layer
            nn.Linear(16 * 12 * 12, 16 * 16 * 16, bias=False),
            nn.ReLU()
        )
        self.dec_transconv = nn.Sequential(
            nn.Upsample(scale_factor= 2, mode='nearest'),
            nn.Conv2d(16, 16, kernel_size, bias=False),
            nn.BatchNorm2d(16, momentum=0.005),
            nn.ReLU(),

            nn.Upsample(scale_factor= 2, mode='nearest'),
            nn.Conv2d(16, 8, kernel_size, bias=False),
            nn.BatchNorm2d(8, momentum=0.005),
            nn.ReLU(),

            nn.Upsample(scale_factor= 2, mode='nearest'),
            nn.Conv2d(8, 8, kernel_size, bias=False),
            nn.BatchNorm2d(8, momentum=0.005),
            nn.ReLU(),

            nn.Upsample(scale_factor= 2, mode='nearest'),
            nn.Conv2d(8, 4, kernel_size, bias=False),
            nn.BatchNorm2d(4, momentum=0.005),
            nn.ReLU(),

            ###output layer
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

    def __init__(self, img_dim=28, dropout=.2, in_ch=1, phy_dim=8, stride=2, kernel_size=4):
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
            nn.Linear(16 * 16 * 8, 16 * 16 * 16, bias=False),
            nn.ReLU()
        )

        #convolutional layers
        self.dec_transconv = nn.Sequential(
            nn.ConvTranspose2d(16, 16,  kernel_size, stride=stride, bias=False,
                               output_padding=1, padding=0),
            nn.Conv2d(16, 16, kernel_size, bias=False),
            nn.BatchNorm2d(16, momentum=0.005),
            nn.ReLU(),

            nn.Conv2d(16, 8, kernel_size, bias=False),
            nn.BatchNorm2d(8, momentum=0.005),
            nn.ReLU(),

            nn.ConvTranspose2d(8, 8, kernel_size, stride=stride, bias=False,
                               output_padding=1, padding=0),

            nn.Conv2d(8, 8, kernel_size, bias=False),
            nn.BatchNorm2d(8, momentum=0.005),
            nn.ReLU(),

            nn.Conv2d(8, 4, kernel_size, bias=False),
            nn.BatchNorm2d(4, momentum=0.005),
            nn.ReLU(),

            ###output layer
            nn.ConvTranspose2d(4, 4, kernel_size, stride=stride, bias=False,
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



class Conv_Forward_AE(nn.Module):

    def __init__(self, img_dim=28, dropout=.2, in_ch=1, phy_dim=8,
        stride=2, kernel_size=4, numb_conv=5, numb_lin=5, a_func=nn.ReLU()):
        """
        Parameters
        ----------
        img_dim    : int
            image size, only one dimension, assuming square ratio.
        dropout    : float
            dropout probability
        in_ch      : int
            number of channels in input/output image
        stride     : int
            stride step size (must be >1)
        kernel_size: int
            kernel size for the convolutional layers
        numb_conv  : int
            number of convolutional layers in the model. 5 by defalult.
            Works best for <=8 layers, anything higher might result in errors.
            NOTE: Does not support 0 convolutional layers as it will always have
            one minimum, by construction.
        numb_lin   : int
            number of linear layers in the model. 5 by defalult.
        """

        super(Conv_Forward_AE, self).__init__()
        self.in_ch = in_ch
        self.img_dim = img_dim

        # Linear layers
        h_ch = 2
        self.lin = nn.Sequential(
        nn.Linear(phy_dim,  16 * h_ch * h_ch, bias=False),
        nn.BatchNorm1d(16 * h_ch * h_ch),
        a_func,
        nn.Dropout(dropout)
        )
        i_ch = h_ch
        h_ch *= 2

        for i in range(numb_lin - 1):
            self.lin.add_module(
            "linear_%i" % (i+1),
            nn.Linear(16 * i_ch * i_ch, 16 * h_ch * h_ch, bias=False)
            )

            if (i != numb_lin - 2):

                self.lin.add_module(
                "bn_%i" % (i + 1),
                nn.BatchNorm1d(16 * h_ch * h_ch)
                )

                self.lin.add_module(
                "activation_%i" % (i + 1),
                a_func
                )

                self.lin.add_module(
                "Dropout_%i" % (i + 2),
                nn.Dropout(dropout)
                )
                i_ch = h_ch
                if (numb_lin > 4 and i >= 3):
                    h_ch += 2
                else:
                    h_ch *= 2

            else:
                self.lin.add_module(
                "activation_output",
                a_func
                )

        self.h_ch = h_ch

        #Convolutional layers
        self.conv = nn.Sequential()
        i_ch = 16
        #o_ch = i_ch * (numb_conv - 1)

        for i in range(numb_conv - 1):
            if (i%2 == 0):
                self.conv.add_module(
                "ConvTranspose2d_%i" % (i+1),
                nn.ConvTranspose2d(i_ch, i_ch, kernel_size, stride=stride, bias=False,
                                    output_padding=1, padding=0)
                )
                self.conv.add_module(
                "conv2d_%i" % (i + 1),
                nn.Conv2d(i_ch, i_ch, kernel_size, bias=False)
                )
                self.conv.add_module(
                "bn_%i" % (i + 1),
                nn.BatchNorm2d(i_ch, momentum=0.005)
                )
                self.conv.add_module(
                "activation_%i" % (i + 1),
                a_func
                )
                #i_ch = o_ch
                o_ch = i_ch//2

            else:
                self.conv.add_module(
                "conv2d_%i" % (i + 1),
                nn.Conv2d(i_ch, o_ch, kernel_size, bias=False),
                )
                self.conv.add_module(
                    "bn_%i" % (i + 1), nn.BatchNorm2d(o_ch, momentum=0.005)
                )
                self.conv.add_module("relu_%i" % (i + 1), nn.ReLU())
                i_ch = o_ch
                #o_ch = o_ch//2

        #output layers
        self.conv.add_module(
        "ConvTransposed2d_output",
        nn.ConvTranspose2d(i_ch, 4, kernel_size, stride=stride, bias=False,
                            output_padding=1, padding=0))

        self.conv.add_module(
        "Conv2d_output",
        nn.Conv2d(4, in_ch, kernel_size - 1)
        )

        self.conv.add_module(
        "Sigmoid",
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
        z = self.lin(phy)
        z = z.view(-1, 16, self.h_ch, self.h_ch)
        z = self.conv(z)

        z = F.interpolate(z, size=(self.img_dim, self.img_dim),
                          mode='nearest')
        return z
