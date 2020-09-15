"""
This python file contains the Autoencoder models as classes
per model. Architectures include linear, convolution, transpose
convolution, upampling, and ResNet type of NN/layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

def conv_out(l0, k, st):
    """
    return the output size after applying a convolution:
    Parameters
    ----------
    l0 : int
        initial size
    k  : int
        kernel size
    st : int
        stride size
    Returns
    -------
    int
        output size
    """
    return int((l0 - k)/st + 1)

def pool_out(l0, k, st):
    """
    return the output size after applying a convolution:
    Parameters
    ----------
    l0 : int
        initial size
    k  : int
        kernel size
    st : int
        stride size
    Returns
    -------
    int
        output size
    """
    return int((l0 - k)/st + 1)


class Linear_AE(nn.Module):
    """
    Autoencoder class with user defined latent space, image size, 
    and number of image channels. Encoder and decoder layers are
    sequences of Dense + Act_fn + Dropout
    
    NOTE: here the number of neurons and layers are tunned to work
    best with [28x28] images, as MNIST
    
    ...
    
    Attributes
    ----------
    latent_dim : int
        size of latent space
    img_width  : int
        width size of image
    img_height : int
        height size of image
    img_size   : float
        total numer of pixels in image
    enc        : pytorch sequential
        encoder layers organized in a sequential module.
    dec        : pytorch sequential
        decoder layers organized in a sequential module.
    Methods
    -------
    encoder(self, x)
        Encoder module
    decoder(self, z)
        Decoder module
    forward(self, x)
        AE forward pass
    """
    def __init__(self, latent_dim=32, img_dim=28, dropout=.2, in_ch=1):
        """
        Parameters
        ----------
        latent_dim : int
            size of the dimensilatent space
        img_dim    : int
            image size, only one dimension, assuming square ratio.
        dropout    : float
            dropout probability
        in_ch      : int
            number of channels in input/output image
        """
        super(Linear_AE, self).__init__()
        self.latent_dim = latent_dim
        self.img_width = self.img_height = img_dim
        self.img_size = self.img_width * self.img_height
        self.in_ch = in_ch

        # Encoder specification, nn.Sequential is a sequiential container
        # of a stack of layers, each layer's output is the input of the
        # following layer. This one contains Linear (aka fully connected
        # or dense) layers followed by ReLU activation func and a dropout
        # layer.
        self.enc = nn.Sequential(
            nn.Linear(self.img_size, 320),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(320, 160),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(160, 80),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(80, self.latent_dim)
        )
        # Decoder specification
        self.dec = nn.Sequential(
            nn.Linear(self.latent_dim, 80),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(80, 160),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(160, 320),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(320, self.img_size),
            #nn.Tanh()
        )

    def encode(self, x):
        """
        Encoder side of autoencoder.
        
        Parameters
        ----------
        x : tensor
            input image with shape [N, C, H, W]
        Returns
        -------
            latent code
        """
        x = x.view([x.size(0), -1])
        x = self.enc(x)
        return x

    def decode(self, z):
        """
        Decoder side of autoencoder.
        
        Parameters
        ----------
        z : tensor
            latent code [N, latent_dim]
        Returns
        -------
            reconstructed image [N, C, H, W]
        """
        z = self.dec(z)
        z = z.view([z.size(0), self.in_ch, self.img_width, self.img_height])
        return z

    def forward(self, x):
        """
        Autoencoder forward pass.
        
        Parameters
        ----------
        x : tensor
            input image with shape [N, C, H, W]
        Returns
        -------
        xhat : tensor
            reconstructe image [N, C, H, W]
        z    : tensor
            latent code [N, latent_dim]
        """
        z = self.encode(x)
        xhat = self.decode(z)
        return xhat, z


class TranConv_AE(nn.Module):
    """
    Autoencoder class with user defined latent dimension, image size, 
    and number of image channels. The encoder is constructed with
    sets of 2Dconv + Act_fn + MaxPooling layers. The decoder
    contains consecutive Transpose 2D Conv + Act_fn layers.
    
    NOTE: here the number of neurons and layers are tunned to work
    best with [28x28] images, as MNIST
    
    ...
    
    Attributes
    ----------
    latent_dim : int
        size of latent space
    img_width  : int
        width size of image
    img_height : int
        height size of image
    enc_conv   : pytorch sequential
        encoder layers organized in a sequential module.
    dec_convt  : pytorch sequential
        decoder layers organized in a sequential module.
    Methods
    -------
    encoder(self, x)
        Encoder module
    decoder(self, z)
        Decoder module
    forward(self, x)
        AE forward pass
    """
    def __init__(self, latent_dim=32, img_dim=28, dropout=.2, in_ch=1):
        """
        Parameters
        ----------
        latent_dim : int
            size of the dimensilatent space
        img_dim    : int
            image size, only one dimension, assuming square ratio.
        dropout    : float
            dropout probability
        in_ch      : int
            number of channels in input/output image
        """
        super(TranConv_AE, self).__init__()
        self.latent_dim = latent_dim
        self.img_width = self.img_height = img_dim

        # Encoder specification, have to update the input/output size
        # according to our PPD images [-1, 1, 187, 187]
        self.enc_conv = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=5),  # (N, 1, 28, 28)->(N,  3, 24, 24)
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),       # (N, 3, 24, 24)->(N,  3, 12, 12)
            nn.Conv2d(3, 6, kernel_size=3),  # (N, 3, 24, 24)->(N,  3, 10, 10)
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),       # (N, 6, 10, 10)->(N,  6, 5, 5)
            nn.Conv2d(6, 6, kernel_size=3),  # (N, 6, 5, 5)  ->(N,  6, 3, 3)
            #nn.ReLU(),
            #nn.AvgPool2d(2, stride=2),       # (N, 6, 4, 4)  ->(N,  6, 2, 2)
        )

        # Decoder specification, ConvTranspose2d is the invertible of
        # Conv2d
        self.dec_convt = nn.Sequential(
            nn.ConvTranspose2d(6, 3, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(3, 3, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(3, 1, kernel_size=2, stride=2),
            #nn.Tanh()
        )

    def encode(self, x):
        """
        Encoder side of autoencoder.
        
        Parameters
        ----------
        x : tensor
            input image with shape [N, C, H, W]
        Returns
        -------
            latent code
        """
        x = self.enc_conv(x)
        return x

    def decode(self, z):
        """
        Decoder side of autoencoder.
        
        Parameters
        ----------
        z : tensor
            latent code [N, latent_dim]
        Returns
        -------
            reconstructed image [N, C, H, W]
        """
        z = self.dec_convt(z)
        return z

    def forward(self, x):
        """
        Autoencoder forward pass.
        
        Parameters
        ----------
        x : tensor
            input image with shape [N, C, H, W]
        Returns
        -------
        xhat : tensor
            reconstructe image [N, C, H, W]
        z    : tensor
            latent code [N, latent_dim]
        """
        z = self.encode(x)
        xhat = self.decode(z)
        return xhat, z.flatten(1)


class ConvLin_AE(nn.Module):
    """
    Autoencoder class with user defined latent dimension, image size, 
    and number of image channels. The encoder is constructed with
    sets of [2Dconv + Act_fn + MaxPooling] blocks, user defined, 
    with a final linear layer to return the latent code.
    The decoder is build using Linear layers.
    
    NOTE: here the number of neurons and layers in the decoder 
    are tunned to work best with [28x28] images, as MNIST
    
    ...
    
    Attributes
    ----------
    latent_dim : int
        size of latent space
    img_width  : int
        width size of image
    img_height : int
        height size of image
    img_size   : float
        total numer of pixels in image
    in_ch      : int
        number of image channels
    enc_conv_blocks   : pytorch sequential
        encoder layers organized in a sequential module
    enc_linear : pytorch sequential
        encoder linear output layer
    dec_linear  : pytorch sequential
        decoder layers organized in a sequential module
    Methods
    -------
    encoder(self, x)
        Encoder module
    decoder(self, z)
        Decoder module
    forward(self, x)
        AE forward pass
    """
    def __init__(self, latent_dim=32, img_dim=28, dropout=.2, in_ch=1,
                 kernel=3, n_conv_blocks=5):
        """
        Parameters
        ----------
        latent_dim : int
            size of the dimensilatent space
        img_dim    : int
            image size, only one dimension, assuming square ratio.
        dropout    : float
            dropout probability
        in_ch      : int
            number of channels in input/output image
        kernel     : int
            size of the convolving kernel
        n_conv_blocks : int
            number of [conv + relu + maxpooling] blocks
        """
        super(ConvLin_AE, self).__init__()
        self.latent_dim = latent_dim
        self.img_width = self.img_height = img_dim
        self.img_size = self.img_width * self.img_height
        self.in_ch = in_ch

        # Encoder specification
        self.enc_conv_blocks = nn.Sequential()
        h_ch = in_ch
        for i in range(n_conv_blocks):
            self.enc_conv_blocks.add_module('conv2d_%i' % (i+1),
                                            nn.Conv2d(h_ch, h_ch*2,
                                                      kernel_size=kernel))
            self.enc_conv_blocks.add_module('relu_%i' % (i+1), nn.ReLU())
            self.enc_conv_blocks.add_module('maxpool_%i' % (i+1),
                                            nn.MaxPool2d(2, stride=2))
            h_ch *= 2
            img_dim = conv_out(img_dim, kernel, 1)
            img_dim = pool_out(img_dim, 2, 2)

        self.enc_linear = nn.Sequential(
            nn.Linear(h_ch * img_dim**2, 50),
            nn.ReLU(),
            nn.Linear(50, self.latent_dim),
        )

        # Decoder specification
        self.dec_linear = nn.Sequential(
            nn.Linear(self.latent_dim, 80),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(80, 160),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(160, 320),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(320, self.img_size),
            nn.Sigmoid()
        )

    def encode(self, x):
        """
        Encoder side of autoencoder.
        
        Parameters
        ----------
        x : tensor
            input image with shape [N, C, H, W]
        Returns
        -------
            latent code
        """
        x = self.enc_conv_blocks(x)
        x = x.flatten(1)
        x = self.enc_linear(x)
        return x

    def decode(self, z):
        """
        Decoder side of autoencoder.
        
        Parameters
        ----------
        z : tensor
            latent code [N, latent_dim]
        Returns
        -------
            reconstructed image [N, C, H, W]
        """
        z = self.dec_linear(z)
        z = z.view([z.size(0), self.in_ch, 
                    self.img_width, self.img_height])
        return z

    def forward(self, x):
        """
        Autoencoder forward pass.
        
        Parameters
        ----------
        x : tensor
            input image with shape [N, C, H, W]
        Returns
        -------
        xhat : tensor
            reconstructe image [N, C, H, W]
        z    : tensor
            latent code [N, latent_dim]
        """
        z = self.encode(x)
        xhat = self.decode(z)
        return xhat, z


class ResNet_Tconv_AE(nn.Module):
    """
    Autoencoder class with user defined latent dimension, image size, 
    and number of image channels. The encoder is constructed with
    a ResNet18 (https://arxiv.org/pdf/1512.03385.pdf) NN with output
    linear layer customized for latent code output.
    The decoder is build using Linear layer to expand the latent code
    into a 2d image, then upscaled using consecutive transpose
    convolutions  + batch norm + act fn.
    
    ...
    
    Attributes
    ----------
    latent_dim : int
        size of latent space
    img_width  : int
        width size of image
    img_height : int
        height size of image
    in_ch      : int
        number of image channels
    resnet    : pytorch sequential
        custom ResNet NN
    enc_linear : pytorch sequential
        encoder linear output layer
    dec_expand  : pytorch sequential
        decoder layers that expand from 1d latent vector to a 2d image
    dec_convTrans: pytorch sequential
        blocks of tranpose conv + batch norm + act fn to upscale 2d images
    Methods
    -------
    encoder(self, x)
        Encoder module
    decoder(self, z)
        Decoder module
    forward(self, x)
        AE forward pass
    """
    def __init__(self, latent_dim=32, img_dim=28, dropout=.2, in_ch=1):
        """
        Parameters
        ----------
        latent_dim : int
            size of the dimensilatent space
        img_dim    : int
            image size, only one dimension, assuming square ratio.
        dropout    : float
            dropout probability
        in_ch      : int
            number of channels in input/output image
        """
        super(ResNet_Tconv_AE, self).__init__()

        self.latent_dim = latent_dim
        self.img_width = self.img_height = img_dim
        self.in_ch = in_ch

        # ENCODER modules
        # ResNet model from pytorch library, pretrained can be changes
        # to False if training from scratch
        resnet = models.resnet18(pretrained=True)
        # delete the last fc layer and replace it with a custom fc layer.
        modules = list(resnet.children())[:-1]      
        # add 1 cnv layer if input channels are different than 3
        if in_ch != 3:
            first_conv_layer = [nn.Conv2d(in_ch, 3, kernel_size=1,
                                          stride=1, padding=0, bias=True)]
            first_conv_layer.extend(modules)
            modules = first_conv_layer
            del first_conv_layer
        # Cutom linear + batch normalization layer
        self.resnet = nn.Sequential(*modules)
        self.enc_linear = nn.Sequential(
            nn.Linear(resnet.fc.in_features, 64, bias=False),
            nn.BatchNorm1d(64, momentum=0.01),
            nn.ReLU(),
            nn.Linear(64, self.latent_dim)
        )

        # DECODER modules
        # expand latent code into a 2d image with 64 channels
        self.dec_expand = nn.Sequential(
            nn.Linear(self.latent_dim, 64, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 64 * 4 * 4, bias=False),
            nn.BatchNorm1d(64 * 4 * 4),
            nn.ReLU()
        )

        # blocks of transpose convolutions + BatchNorm + act_fn
        self.dec_convTrans6 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32,
                               kernel_size=3, stride=2,
                               padding=0, bias=False),
            nn.BatchNorm2d(32, momentum=0.01),
            nn.ReLU(inplace=True),
        )
        self.dec_convTrans7 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=8,
                               kernel_size=3, stride=2,
                               padding=0, bias=False),
            nn.BatchNorm2d(8, momentum=0.01),
            nn.ReLU(inplace=True),
        )
        #self.dec_convTrans8 = nn.Sequential(
        #    nn.ConvTranspose2d(in_channels=32, out_channels=16,
        #                       kernel_size=3, stride=2,
        #                       padding=0),
        #    nn.BatchNorm2d(16, momentum=0.01),
        #    nn.ReLU(inplace=True),
        #)
        #self.dec_convTrans9 = nn.Sequential(
        #    nn.ConvTranspose2d(in_channels=16, out_channels=8,
        #                       kernel_size=3, stride=2,
        #                       padding=0),
        #    nn.BatchNorm2d(8, momentum=0.01),
        #    nn.ReLU(inplace=True),
        #)
        self.dec_convTrans10 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=8, out_channels=in_ch,
                               kernel_size=3, stride=2,
                               padding=0),
            nn.Sigmoid()
        )

    def encode(self, x):
        """
        Encoder side of autoencoder.
        
        Parameters
        ----------
        x : tensor
            input image with shape [N, C, H, W]
        Returns
        -------
            latent code [N, latent_dim]
        """
        x = self.resnet(x)         # ResNet
        x = self.enc_linear(x.flatten(1))     # FC output layers
        return x

    def decode(self, z):
        """
        Decoder side of autoencoder.
        
        Parameters
        ----------
        z : tensor
            latent code [N, latent_dim]
        Returns
        -------
            reconstructed image [N, C, H, W]
        """
        x = self.dec_expand(z).view(-1, 64, 4, 4)
        x = self.dec_convTrans6(x)
        x = self.dec_convTrans7(x)
        # x = self.convTrans8(x)
        # x = self.convTrans9(x)
        x = self.dec_convTrans10(x)
        # interpolation layer to rescale to original image size
        x = F.interpolate(x, size=(self.img_width, self.img_height),
                          mode='bilinear')
        return x

    def forward(self, x):
        """
        Autoencoder forward pass.
        
        Parameters
        ----------
        x : tensor
            input image with shape [N, C, H, W]
        Returns
        -------
        xhat : tensor
            reconstructe image [N, C, H, W]
        z    : tensor
            latent code [N, latent_dim]
        """
        z = self.encode(x)
        xhat = self.decode(z)

        return xhat, z
    
    
    
class ResNet_Linear_AE(nn.Module):
    """
    Autoencoder class with user defined latent dimension, image size, 
    and number of image channels. The encoder is constructed with
    a ResNet18 (https://arxiv.org/pdf/1512.03385.pdf) NN with output
    linear layer customized for latent code output.
    The decoder is a set of consecutive linear + batch norm + act fn
    layers.
    
    ...
    
    Attributes
    ----------
    latent_dim : int
        size of latent space
    img_width  : int
        width size of image
    img_height : int
        height size of image
    in_ch      : int
        number of image channels
    resnet    : pytorch sequential
        custom ResNet NN
    enc_linear : pytorch sequential
        encoder linear output layer
    dec_linear  : pytorch sequential
        decoder linear layers
    Methods
    -------
    encoder(self, x)
        Encoder module
    decoder(self, z)
        Decoder module
    forward(self, x)
        AE forward pass
    """
    def __init__(self, latent_dim=32, img_dim=28, dropout=.2, in_ch=1):
        """
        Parameters
        ----------
        latent_dim : int
            size of the dimensilatent space
        img_dim    : int
            image size, only one dimension, assuming square ratio.
        dropout    : float
            dropout probability
        in_ch      : int
            number of channels in input/output image
        """
        super(ResNet_Linear_AE, self).__init__()

        self.latent_dim = latent_dim
        self.img_width = self.img_height = img_dim
        self.in_ch = in_ch

        # ENCODER modules
        # ResNet model from pytorch library, pretrained can be changes
        # to False if training from scratch
        resnet = models.resnet18(pretrained=True)
        # delete the last fc layer and replace it with a custom fc layer.
        modules = list(resnet.children())[:-1]      
        # add 1 cnv layer if input channels are different than 3
        if in_ch != 3:
            first_conv_layer = [nn.Conv2d(in_ch, 3, kernel_size=1,
                                          stride=1, padding=0, bias=True)]
            first_conv_layer.extend(modules)
            modules = first_conv_layer
            del first_conv_layer
        # Cutom linear + batch normalization layer
        self.resnet = nn.Sequential(*modules)
        self.enc_linear = nn.Sequential(
            nn.Linear(resnet.fc.in_features, 64, bias=False),
            nn.BatchNorm1d(64, momentum=0.01),
            nn.ReLU(),
            nn.Linear(64, self.latent_dim)
        )

        # DECODER modules
        self.dec_linear = nn.Sequential(
            nn.Linear(self.latent_dim, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 32 * 4 * 4, bias=False),
            nn.BatchNorm1d(32 * 4 * 4),
            nn.ReLU(),
            nn.Linear(32 * 4 * 4, 16 * 16 * 16, bias=False),
            nn.BatchNorm1d(16 * 16 * 16),
            nn.ReLU(),
            nn.Linear(16 * 16 * 16, 4 * 96 * 96, bias=False),
            nn.BatchNorm1d(4 * 96 * 96),
            nn.ReLU(),
            nn.Linear(4 * 96 * 96, in_ch * self.img_width * self.img_height),
            #nn.Tanh()
        )



    def encode(self, x):
        """
        Encoder side of autoencoder.
        
        Parameters
        ----------
        x : tensor
            input image with shape [N, C, H, W]
        Returns
        -------
            latent code [N, latent_dim]
        """
        x = self.resnet(x)                # ResNet
        x = self.enc_linear(x.flatten(1)) # FC output layers
        return x

    def decode(self, z):
        """
        Decoder side of autoencoder.
        
        Parameters
        ----------
        z : tensor
            latent code [N, latent_dim]
        Returns
        -------
            reconstructed image [N, C, H, W]
        """
        x = self.dec_linear(z).view(-1, self.in_ch, 
                                    self.img_width,
                                    self.img_height)
        return x

    def forward(self, x):
        """
        Autoencoder forward pass.
        
        Parameters
        ----------
        x : tensor
            input image with shape [N, C, H, W]
        Returns
        -------
        xhat : tensor
            reconstructe image [N, C, H, W]
        z    : tensor
            latent code [N, latent_dim]
        """
        z = self.encode(x)
        xhat = self.decode(z)

        return xhat, z
    
    
    
class ResNet_UpSamp_AE(nn.Module):
    """
    Autoencoder class with user defined latent dimension, image size, 
    and number of image channels. The encoder is constructed with
    a ResNet18 (https://arxiv.org/pdf/1512.03385.pdf) NN with output
    linear layer customized for latent code output.
    The decoder has a linear layer that expands from 1d latent vector
    into 2d images, then a sequence of Upsample and 2d convolutions that
    expand the images to the original size.
    
    ...
    
    Attributes
    ----------
    latent_dim : int
        size of latent space
    img_width  : int
        width size of image
    img_height : int
        height size of image
    in_ch      : int
        number of image channels
    resnet     : pytorch sequential
        custom ResNet NN
    enc_linear : pytorch sequential
        encoder output linear layer
    dec_linear : pytorch sequential
        decoder linear layers
    dec_upscale : pytorch sequential
        decoder stack of upsample + conv layers
    Methods
    -------
    encoder(self, x)
        Encoder module
    decoder(self, z)
        Decoder module
    forward(self, x)
        AE forward pass
    """
    def __init__(self, latent_dim=32, img_dim=28, dropout=.2, in_ch=1):
        """
        Parameters
        ----------
        latent_dim : int
            size of the dimensilatent space
        img_dim    : int
            image size, only one dimension, assuming square ratio.
        dropout    : float
            dropout probability
        in_ch      : int
            number of channels in input/output image
        """
        super(ResNet_UpSamp_AE, self).__init__()

        self.latent_dim = latent_dim
        self.img_width = self.img_height = img_dim
        self.in_ch = in_ch

        # ENCODER modules
        # ResNet model from pytorch library, pretrained can be changes
        # to False if training from scratch
        resnet = models.resnet18(pretrained=True)
        # delete the last fc layer and replace it with a custom fc layer.
        modules = list(resnet.children())[:-1]      
        # add 1 cnv layer if input channels are different than 3
        if in_ch != 3:
            first_conv_layer = [nn.Conv2d(in_ch, 3, kernel_size=1,
                                          stride=1, padding=0, bias=True)]
            first_conv_layer.extend(modules)
            modules = first_conv_layer
            del first_conv_layer
        # Cutom linear + batch normalization layer
        self.resnet = nn.Sequential(*modules)
        self.enc_linear = nn.Sequential(
            nn.Linear(resnet.fc.in_features, 64, bias=False),
            nn.BatchNorm1d(64, momentum=0.01),
            nn.ReLU(),
            nn.Linear(64, self.latent_dim)
        )

        # DECODER modules
        self.dec_linear = nn.Sequential(
            nn.Linear(self.latent_dim, 16 * 6 * 6),
            nn.ReLU()
        )
        self.dec_upscale = nn.Sequential(
            nn.Upsample(scale_factor= 2, mode='bilinear'),
            nn.Conv2d(16, 16, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Upsample(scale_factor= 2, mode='bilinear'),
            nn.Conv2d(16, 16, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Upsample(scale_factor= 2, mode='bilinear'),
            nn.Conv2d(16, 8, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Upsample(scale_factor= 2, mode='bilinear'),
            nn.Conv2d(8, 8, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Upsample(scale_factor= 2, mode='bilinear'),
            nn.Conv2d(8, 4, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(4, in_ch, kernel_size=3, stride=1),
            #nn.Tanh(),
        )
        
        

    def encode(self, x):
        """
        Encoder side of autoencoder.
        
        Parameters
        ----------
        x : tensor
            input image with shape [N, C, H, W]
        Returns
        -------
            latent code [N, latent_dim]
        """
        x = self.resnet(x)                # ResNet
        x = self.enc_linear(x.flatten(1)) # FC output layers
        return x

    def decode(self, z):
        """
        Decoder side of autoencoder.
        
        Parameters
        ----------
        z : tensor
            latent code [N, latent_dim]
        Returns
        -------
            reconstructed image [N, C, H, W]
        """
        x = self.dec_linear(z).view(-1, 16, 6, 6)
        x = self.dec_upscale(x)
        # interpolation to rescale to original image size
        x = F.interpolate(x, size=(self.img_width, self.img_height),
                          mode='bilinear')
        return x

    def forward(self, x):
        """
        Autoencoder forward pass.
        
        Parameters
        ----------
        x : tensor
            input image with shape [N, C, H, W]
        Returns
        -------
        xhat : tensor
            reconstructe image [N, C, H, W]
        z    : tensor
            latent code [N, latent_dim]
        """
        z = self.encode(x)
        xhat = self.decode(z)

        return xhat, z
    
    
class ConvUpSamp_AE(nn.Module):
    """
    Autoencoder class with user defined latent dimension, image size, 
    and number of image channels. 
    The encoder has Conv + pool layers and a linear layer for output.
    The decoder has a linear layer to expand 1d vector --> 2d image, then
    a stack of Upsample + 2d conv.
    
    ...
    
    Attributes
    ----------
    latent_dim : int
        size of latent space
    img_width  : int
        width size of image
    img_height : int
        height size of image
    in_ch      : int
        number of image channels
    enc_conv   : pytorch sequential
        encoder stack of convolutions
    enc_linear : pytorch sequential
        encoder linear output layer
    dec_linear : pytorch sequential
        decoder linear layers
    dec_upscale : pytorch sequential
        decoder stack of upsample + 2d conv
    Methods
    -------
    encoder(self, x)
        Encoder module
    decoder(self, z)
        Decoder module
    forward(self, x)
        AE forward pass
    """
    def __init__(self, latent_dim=32, img_dim=28, dropout=.2, in_ch=1):
        """
        Parameters
        ----------
        latent_dim : int
            size of the dimensilatent space
        img_dim    : int
            image size, only one dimension, assuming square ratio.
        dropout    : float
            dropout probability
        in_ch      : int
            number of channels in input/output image
        """
        super(ConvUpSamp_AE, self).__init__()

        self.latent_dim = latent_dim
        self.img_width = self.img_height = img_dim
        self.in_ch = in_ch

        # ENCODER modules
        self.enc_conv = nn.Sequential(
            nn.Conv2d(in_ch, 4, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(4, 4, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(4, 8, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 16, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.enc_linear = nn.Sequential(
            nn.Linear(16 * 5 * 5, 64),
            nn.ReLU(),
            nn.Linear(64, self.latent_dim),
        )

        # DECODER modules
        self.dec_linear = nn.Sequential(
            nn.Linear(self.latent_dim, 16 * 6 * 6),
            nn.ReLU()
        )
        self.dec_upscale = nn.Sequential(
            nn.Upsample(scale_factor= 2, mode='bilinear'),
            nn.Conv2d(16, 16, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Upsample(scale_factor= 2, mode='bilinear'),
            nn.Conv2d(16, 16, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Upsample(scale_factor= 2, mode='bilinear'),
            nn.Conv2d(16, 8, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Upsample(scale_factor= 2, mode='bilinear'),
            nn.Conv2d(8, 8, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Upsample(scale_factor= 2, mode='bilinear'),
            nn.Conv2d(8, 4, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(4, in_ch, kernel_size=3, stride=1),
            #nn.Tanh()
        )


    def encode(self, x):
        """
        Encoder side of autoencoder.
        
        Parameters
        ----------
        x : tensor
            input image with shape [N, C, H, W]
        Returns
        -------
            latent code [N, latent_dim]
        """
        x = self.enc_conv(x)
        x = self.enc_linear(x.flatten(1))
        return x

    def decode(self, z):
        """
        Decoder side of autoencoder.
        
        Parameters
        ----------
        z : tensor
            latent code [N, latent_dim]
        Returns
        -------
            reconstructed image [N, C, H, W]
        """
        x = self.dec_linear(z).view(-1, 16, 6, 6)
        x = self.dec_upscale(x)
        
        x = F.interpolate(x, size=(self.img_width, self.img_height),
                          mode='bilinear')
        return x

    def forward(self, x):
        """
        Autoencoder forward pass.
        
        Parameters
        ----------
        x : tensor
            input image with shape [N, C, H, W]
        Returns
        -------
        xhat : tensor
            reconstructe image [N, C, H, W]
        z    : tensor
            latent code [N, latent_dim]
        """
        z = self.encode(x)
        xhat = self.decode(z)

        return xhat, z
    
    
    
class ConvLinUp_AE(nn.Module):
    """
    Autoencoder class with user defined latent dimension, image size, 
    and number of image channels. The encoder is constructed with
    sets of [2Dconv + Act_fn + MaxPooling] blocks, user defined, 
    with a final linear layer to return the latent code.
    The decoder is build using Linear layers.
    
    ...
    
    Attributes
    ----------
    latent_dim : int
        size of latent space
    img_width  : int
        width size of image
    img_height : int
        height size of image
    img_size   : float
        total numer of pixels in image
    in_ch      : int
        number of image channels
    enc_conv_blocks   : pytorch sequential
        encoder layers organized in a sequential module
    enc_linear : pytorch sequential
        encoder linear output layer
    dec_linear  : pytorch sequential
        decoder layers organized in a sequential module
    Methods
    -------
    encoder(self, x)
        Encoder module
    decoder(self, z)
        Decoder module
    forward(self, x)
        AE forward pass
    """
    def __init__(self, latent_dim=32, img_dim=28, dropout=.2, in_ch=1,
                 kernel=3, n_conv_blocks=5):
        """
        Parameters
        ----------
        latent_dim : int
            size of the dimensilatent space
        img_dim    : int
            image size, only one dimension, assuming square ratio.
        dropout    : float
            dropout probability
        in_ch      : int
            number of channels in input/output image
        kernel     : int
            size of the convolving kernel
        n_conv_blocks : int
            number of [conv + relu + maxpooling] blocks
        """
        super(ConvLinUp_AE, self).__init__()
        self.latent_dim = latent_dim
        self.img_width = self.img_height = img_dim
        self.img_size = self.img_width * self.img_height
        self.in_ch = in_ch

        # Encoder specification
        self.enc_conv_blocks = nn.Sequential()
        h_ch = in_ch
        for i in range(n_conv_blocks):
            self.enc_conv_blocks.add_module('conv2d_%i' % (i+1),
                                            nn.Conv2d(h_ch, h_ch*2,
                                                      kernel_size=kernel))
            self.enc_conv_blocks.add_module('relu_%i' % (i+1), nn.ReLU())
            self.enc_conv_blocks.add_module('maxpool_%i' % (i+1),
                                            nn.MaxPool2d(2, stride=2))
            h_ch *= 2
            img_dim = conv_out(img_dim, kernel, 1)
            img_dim = pool_out(img_dim, 2, 2)

        self.enc_linear = nn.Sequential(
            nn.Linear(h_ch * img_dim**2, 48),
            nn.ReLU(),
            nn.Linear(48, self.latent_dim),
        )

        # Decoder specification
        self.dec_linear = nn.Sequential(
            nn.Linear(self.latent_dim, 48),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(48, 8 * 4 * 4),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(8 * 4 * 4, 8 * 8 * 8),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(8 * 8 * 8, 8 * 14 * 14),
            nn.Dropout(dropout),
            nn.ReLU()
        )
        
        self.dec_upscaleconv = nn.Sequential(
            nn.Upsample(scale_factor= 2, mode='bilinear'),
            nn.Conv2d(8, 8, kernel_size=3, stride=1),
            nn.ReLU(),
            #nn.Conv2d(8, 8, kernel_size=1, stride=1),
            #nn.ReLU(),
            nn.Upsample(scale_factor= 2, mode='bilinear'),
            nn.Conv2d(8, 4, kernel_size=3, stride=1),
            nn.ReLU(),
            #nn.Conv2d(4, 4, kernel_size=1, stride=1),
            #nn.ReLU(),
            nn.Upsample(scale_factor= 2, mode='bilinear'),
            nn.Conv2d(4, 4, kernel_size=3, stride=1),
            nn.ReLU(),
            #nn.Conv2d(4, in_ch, kernel_size=1, stride=1),
            #nn.ReLU(),
            nn.Upsample(scale_factor= 2, mode='bilinear'),
            #nn.Conv2d(4, 4, kernel_size=3, stride=1),
            #nn.ReLU(),
            nn.Conv2d(4, in_ch, kernel_size=1, stride=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        """
        Encoder side of autoencoder.
        
        Parameters
        ----------
        x : tensor
            input image with shape [N, C, H, W]
        Returns
        -------
            latent code
        """
        x = self.enc_conv_blocks(x)
        x = self.enc_linear(x.flatten(1))
        return x

    def decode(self, z):
        """
        Decoder side of autoencoder.
        
        Parameters
        ----------
        z : tensor
            latent code [N, latent_dim]
        Returns
        -------
            reconstructed image [N, C, H, W]
        """
        z = self.dec_linear(z).view(-1, 8, 14, 14)
        z = self.dec_upscaleconv(z)
        z = F.interpolate(z, size=(self.img_width, self.img_height),
                          mode='bilinear')
        return z

    def forward(self, x):
        """
        Autoencoder forward pass.
        
        Parameters
        ----------
        x : tensor
            input image with shape [N, C, H, W]
        Returns
        -------
        xhat : tensor
            reconstructe image [N, C, H, W]
        z    : tensor
            latent code [N, latent_dim]
        """
        z = self.encode(x)
        xhat = self.decode(z)
        return xhat, z
    
    
class ConvLinTrans_AE(nn.Module):
    """
    Autoencoder class with user defined latent dimension, image size, 
    and number of image channels. The encoder is constructed with
    sets of [2Dconv + Act_fn + MaxPooling] blocks, user defined, 
    with a final linear layer to return the latent code.
    The decoder is build using Linear layers.
    
    ...
    
    Attributes
    ----------
    latent_dim : int
        size of latent space
    img_width  : int
        width size of image
    img_height : int
        height size of image
    img_size   : float
        total numer of pixels in image
    in_ch      : int
        number of image channels
    enc_conv_blocks   : pytorch sequential
        encoder layers organized in a sequential module
    enc_linear : pytorch sequential
        encoder linear output layer
    dec_linear  : pytorch sequential
        decoder layers organized in a sequential module
    Methods
    -------
    encoder(self, x)
        Encoder module
    decoder(self, z)
        Decoder module
    forward(self, x)
        AE forward pass
    """
    def __init__(self, latent_dim=32, img_dim=28, dropout=.2, in_ch=1,
                 kernel=3, n_conv_blocks=5):
        """
        Parameters
        ----------
        latent_dim : int
            size of the dimensilatent space
        img_dim    : int
            image size, only one dimension, assuming square ratio.
        dropout    : float
            dropout probability
        in_ch      : int
            number of channels in input/output image
        kernel     : int
            size of the convolving kernel
        n_conv_blocks : int
            number of [conv + relu + maxpooling] blocks
        """
        super(ConvLinTrans_AE, self).__init__()
        self.latent_dim = latent_dim
        self.img_width = self.img_height = img_dim
        self.img_size = self.img_width * self.img_height
        self.in_ch = in_ch

        # Encoder specification
        self.enc_conv_blocks = nn.Sequential()
        h_ch = in_ch
        for i in range(n_conv_blocks):
            self.enc_conv_blocks.add_module('conv2d_%i1' % (i+1),
                                            nn.Conv2d(h_ch, h_ch*2,
                                                      kernel_size=kernel, bias=False))
            self.enc_conv_blocks.add_module('bn_%i1' % (i+1), 
                                            nn.BatchNorm2d(h_ch*2, momentum=0.005))
            self.enc_conv_blocks.add_module('relu_%i1' % (i+1), 
                                            nn.ReLU())
            self.enc_conv_blocks.add_module('conv2d_%i2' % (i+1),
                                            nn.Conv2d(h_ch*2, h_ch*2,
                                                      kernel_size=kernel, bias=False))
            self.enc_conv_blocks.add_module('bn_%i2' % (i+1), 
                                            nn.BatchNorm2d(h_ch*2, momentum=0.005))
            self.enc_conv_blocks.add_module('relu_%i2' % (i+1), 
                                            nn.ReLU())
            self.enc_conv_blocks.add_module('maxpool_%i' % (i+1),
                                            nn.MaxPool2d(2, stride=2))
            h_ch *= 2
            img_dim = conv_out(img_dim, kernel, 1)
            img_dim = conv_out(img_dim, kernel, 1)
            img_dim = pool_out(img_dim, 2, 2)

        self.enc_linear = nn.Sequential(
            nn.Linear(h_ch * img_dim**2, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, self.latent_dim),
        )

        # Decoder specification
        self.dec_linear = nn.Sequential(
            nn.Linear(self.latent_dim, 128, bias=False),
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
            nn.ConvTranspose2d(16, 16, 5, stride=2, output_padding=1, bias=False),
            #nn.Conv2d(16, 16, 3, bias=False),
            nn.BatchNorm2d(16, momentum=0.005),
            nn.ReLU(),
            nn.Conv2d(16, 8, 3, bias=False),
            nn.BatchNorm2d(8, momentum=0.005),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 8, 5, stride=2, output_padding=1, bias=False),
            #nn.Conv2d(8, 8, 3, bias=False),
            nn.BatchNorm2d(8, momentum=0.005),
            nn.ReLU(),
            nn.Conv2d(8, 4, 3, bias=False),
            nn.BatchNorm2d(4, momentum=0.005),
            nn.ReLU(),
            nn.ConvTranspose2d(4, 4, 5, stride=2, output_padding=1, bias=False),
            #nn.Conv2d(4, 4, 3, bias=False),
            nn.BatchNorm2d(4, momentum=0.005),
            nn.ReLU(),
            nn.Conv2d(4, 4, 3, bias=False),
            nn.BatchNorm2d(4, momentum=0.005),
            nn.ReLU(),
            nn.ConvTranspose2d(4, 4, 3, stride=2, output_padding=1, bias=False),
            #nn.Conv2d(4, 4, 3, bias=False),
            nn.BatchNorm2d(4, momentum=0.005),
            nn.ReLU(),
            nn.Conv2d(4, in_ch, 3),
            nn.Sigmoid()
        )

    def encode(self, x):
        """
        Encoder side of autoencoder.
        
        Parameters
        ----------
        x : tensor
            input image with shape [N, C, H, W]
        Returns
        -------
            latent code
        """
        x = self.enc_conv_blocks(x)
        x = self.enc_linear(x.flatten(1))
        return x

    def decode(self, z):
        """
        Decoder side of autoencoder.
        
        Parameters
        ----------
        z : tensor
            latent code [N, latent_dim]
        Returns
        -------
            reconstructed image [N, C, H, W]
        """
        z = self.dec_linear(z).view(-1, 16, 16, 16)
        z = self.dec_transconv(z)
        
        z = F.interpolate(z, size=(self.img_width, self.img_height),
                          mode='bilinear')
        return z

    def forward(self, x):
        """
        Autoencoder forward pass.
        
        Parameters
        ----------
        x : tensor
            input image with shape [N, C, H, W]
        Returns
        -------
        xhat : tensor
            reconstructe image [N, C, H, W]
        z    : tensor
            latent code [N, latent_dim]
        """
        z = self.encode(x)
        xhat = self.decode(z)
        return xhat, z
    
    
    
class ResLinTrans_AE(nn.Module):
    """
    Autoencoder class with user defined latent dimension, image size, 
    and number of image channels. The encoder is constructed with
    sets of [2Dconv + Act_fn + MaxPooling] blocks, user defined, 
    with a final linear layer to return the latent code.
    The decoder is build using Linear layers.
    
    ...
    
    Attributes
    ----------
    latent_dim : int
        size of latent space
    img_width  : int
        width size of image
    img_height : int
        height size of image
    img_size   : float
        total numer of pixels in image
    in_ch      : int
        number of image channels
    enc_conv_blocks   : pytorch sequential
        encoder layers organized in a sequential module
    enc_linear : pytorch sequential
        encoder linear output layer
    dec_linear  : pytorch sequential
        decoder layers organized in a sequential module
    Methods
    -------
    encoder(self, x)
        Encoder module
    decoder(self, z)
        Decoder module
    forward(self, x)
        AE forward pass
    """
    def __init__(self, latent_dim=32, img_dim=28, dropout=.2, in_ch=1):
        """
        Parameters
        ----------
        latent_dim : int
            size of the dimensilatent space
        img_dim    : int
            image size, only one dimension, assuming square ratio.
        dropout    : float
            dropout probability
        in_ch      : int
            number of channels in input/output image
        kernel     : int
            size of the convolving kernel
        n_conv_blocks : int
            number of [conv + relu + maxpooling] blocks
        """
        super(ResLinTrans_AE, self).__init__()
        self.latent_dim = latent_dim
        self.img_width = self.img_height = img_dim
        self.img_size = self.img_width * self.img_height
        self.in_ch = in_ch
        
        # ENCODER modules
        # ResNet model from pytorch library, pretrained can be changes
        # to False if training from scratch
        resnet = models.resnet18(pretrained=True)
        # delete the last fc layer and replace it with a custom fc layer.
        modules = list(resnet.children())[:-1]      
        # add 1 cnv layer if input channels are different than 3
        if in_ch != 3:
            first_conv_layer = [nn.Conv2d(in_ch, 3, kernel_size=1,
                                          stride=1, padding=0, bias=True)]
            first_conv_layer.extend(modules)
            modules = first_conv_layer
            del first_conv_layer
        # Cutom linear + batch normalization layer
        self.resnet = nn.Sequential(*modules)
        self.enc_linear = nn.Sequential(
            nn.Linear(resnet.fc.in_features, 128),
            nn.BatchNorm1d(128, momentum=0.01),
            nn.ReLU(),
            nn.Linear(128, self.latent_dim)
        )

        # Decoder specification
        self.dec_linear = nn.Sequential(
            nn.Linear(self.latent_dim, 48, bias=False),
            nn.BatchNorm1d(48),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(48, 8 * 4 * 4, bias=False),
            nn.BatchNorm1d(8 * 4 * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(8 * 4 * 4, 8 * 8 * 8, bias=False),
            nn.BatchNorm1d(8 * 8 * 8),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(8 * 8 * 8, 8 * 15 * 15, bias=False),
            nn.BatchNorm1d(8 * 15 * 15),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        self.dec_transconv = nn.Sequential(
            nn.ConvTranspose2d(8, 8, 3, stride=2, padding=0, bias=False),
            #nn.Conv2d(8, 8, 3, bias=False),
            nn.BatchNorm2d(8, momentum=0.005),
            nn.ReLU(),
            nn.Conv2d(8, 8, 3, bias=False),
            nn.BatchNorm2d(8, momentum=0.005),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 4, 3, stride=2, padding=0, bias=False),
            #nn.Conv2d(4, 4, 3, bias=False),
            nn.BatchNorm2d(4, momentum=0.005),
            nn.ReLU(),
            nn.Conv2d(4, 4, 3, bias=False),
            nn.BatchNorm2d(4, momentum=0.005),
            nn.ReLU(),
            nn.ConvTranspose2d(4, 4, 3, stride=2, padding=0, bias=False),
            #nn.Conv2d(4, 4, 3, bias=False),
            nn.BatchNorm2d(4, momentum=0.005),
            nn.ReLU(),
            nn.Conv2d(4, 4, 3, bias=False),
            nn.BatchNorm2d(4, momentum=0.005),
            nn.ReLU(),
            nn.ConvTranspose2d(4, 4, 3, stride=2, padding=0, bias=False),
            #nn.Conv2d(4, 4, 3, bias=False),
            nn.BatchNorm2d(4, momentum=0.005),
            nn.ReLU(),
            nn.Conv2d(4, in_ch, 3),
            nn.Sigmoid()
        )

    def encode(self, x):
        """
        Encoder side of autoencoder.
        
        Parameters
        ----------
        x : tensor
            input image with shape [N, C, H, W]
        Returns
        -------
            latent code
        """
        x = self.resnet(x)
        x = self.enc_linear(x.flatten(1))
        return x

    def decode(self, z):
        """
        Decoder side of autoencoder.
        
        Parameters
        ----------
        z : tensor
            latent code [N, latent_dim]
        Returns
        -------
            reconstructed image [N, C, H, W]
        """
        z = self.dec_linear(z).view(-1, 8, 15, 15)
        z = self.dec_transconv(z)
        
        z = F.interpolate(z, size=(self.img_width, self.img_height),
                          mode='bilinear')
        return z

    def forward(self, x):
        """
        Autoencoder forward pass.
        
        Parameters
        ----------
        x : tensor
            input image with shape [N, C, H, W]
        Returns
        -------
        xhat : tensor
            reconstructe image [N, C, H, W]
        z    : tensor
            latent code [N, latent_dim]
        """
        z = self.encode(x)
        xhat = self.decode(z)
        return xhat, z