import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from model import config as cfg 
# import matplotlib.pyplot as plt
from torch.nn import Module, Conv2d, Parameter, Softmax
from .encoder import Encoder
from .encoder import Encoder_V1 as Encoder_VMamba
from .encoder import Encoder_V2 as Encoder_VMamba_V2
from .encoder import Encoder_V3 as Encoder_VMamba_V3
from .encoder import Encoder_Vmamba2_V4 as Encoder_Vmamba2_V4
from .encoder import Encoder_Vmamba2
from .encoder import LightweightEncoder
from .encoder import Encoder_ConvNextV2
from .caam import CAAM


class ConvBatchnormRelu(nn.Module):
    '''
    This class defines the convolution layer with batch normalization and PReLU activation
    '''

    def __init__(self, nIn, nOut, kSize=3, stride=1, groups=1,dropout_rate=0.0):
        '''

        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        '''
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, kSize, stride=stride, padding=padding, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(nOut)
        self.act = nn.PReLU(nOut)
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else None

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        # output = self.conv1(output)
        output = self.bn(output)
        output = self.act(output)
        if self.dropout:
            output = self.dropout(output)
        return output


class ConvBatchnormReluFactorial(ConvBatchnormRelu):
    def __init__(self, nIn, nOut, kSize=3, stride=1, groups=1,dropout_rate=0.0):
        super().__init__(nIn, nOut, kSize, stride, groups, dropout_rate)
        padding = int((kSize - 1) / 2)
        self.conv = nn.Sequential(
            nn.Conv2d(nIn, nOut, (kSize, 1), stride=stride, padding=(padding, 0), bias=False, groups=groups),
            nn.Conv2d(nOut, nOut, (1, kSize), stride=stride, padding=(0, padding), bias=False, groups=groups)
        )


# class C(nn.Module):
#     '''
#     This class is for a convolutional layer.
#     '''

#     def __init__(self, nIn, nOut, kSize, stride=1, groups=1):
#         '''

#         :param nIn: number of input channels
#         :param nOut: number of output channels
#         :param kSize: kernel size
#         :param stride: optional stride rate for down-sampling
#         '''
#         super().__init__()
#         padding = int((kSize - 1) / 2)
#         self.conv = nn.Conv2d(nIn, nOut, kSize, stride=stride, padding=padding, bias=False,
#                               groups=groups)

#     def forward(self, input):
#         '''
#         :param input: input feature map
#         :return: transformed feature map
#         '''
#         output = self.conv(input)
#         return output

class UpSimpleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2, padding=0, output_padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-03)
        self.act = nn.PReLU(out_channels)

    def forward(self, input):
        output = self.deconv(input)
        output = self.bn(output)
        output = self.act(output)
        return output

class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, sub_dim=3, last=False,kernel_size = 3):
        super(UpConvBlock, self).__init__()
        self.last=last
        self.up_conv = UpSimpleBlock(in_channels, out_channels)
        if not last:
            self.conv1 = ConvBatchnormRelu(out_channels+sub_dim,out_channels,kernel_size)
        self.conv2 = ConvBatchnormRelu(out_channels,out_channels,kernel_size)

    def forward(self, x, ori_img=None):
        x = self.up_conv(x)
        if not self.last:
            x = torch.cat([x, ori_img], dim=1)
            x = self.conv1(x)
        x = self.conv2(x)
        return x


class UpConvBlockFactorial(UpConvBlock):
    def __init__(self, in_channels, out_channels, sub_dim=3, last=False, kernel_size = 3):
        super(UpConvBlockFactorial, self).__init__(in_channels, out_channels, sub_dim, last, kernel_size)
        if not last:
            self.conv1 = ConvBatchnormReluFactorial(out_channels+sub_dim,out_channels,kernel_size)
        self.conv2 = ConvBatchnormReluFactorial(out_channels,out_channels,kernel_size)


class SingleLiteNetPlus(nn.Module):
    '''
    This class defines the ESPNet network
    '''

    def __init__(self, args=None):

        super().__init__()
        chanel_img = cfg.chanel_img
        model_cfg = cfg.sc_ch_dict[args.config] 
        self.encoder = Encoder(args.config)


        self.caam = CAAM(feat_in=cfg.sc_ch_dict[args.config]['chanels'][2], num_classes=cfg.sc_ch_dict[args.config]['chanels'][2],bin_size =(2,4), norm_layer=nn.BatchNorm2d)
        self.conv_caam = ConvBatchnormRelu(cfg.sc_ch_dict[args.config]['chanels'][2],cfg.sc_ch_dict[args.config]['chanels'][1])

        self.up_1 = UpConvBlock(cfg.sc_ch_dict[args.config]['chanels'][1],cfg.sc_ch_dict[args.config]['chanels'][0]) # out: Hx4, Wx4
        self.up_2 = UpConvBlock(cfg.sc_ch_dict[args.config]['chanels'][0],8) #out: Hx2, Wx2
        self.out = UpConvBlock(8,2,last=True)


    def forward(self, input):
        '''
        :param input: RGB image
        :return: transformed feature map
        '''
        out_encoder,inp1,inp2=self.encoder(input)
        #visualize_feature_map_subset(out_encoder, "outencoder", 128)

        out_caam=self.caam(out_encoder)
        out_caam=self.conv_caam(out_caam)


        out=self.up_1(out_caam,inp2)
        out=self.up_2(out,inp1)
        out=self.out(out)


        return out

class TwinLiteNetPlus(nn.Module):
    '''
    This class defines the ESPNet network
    '''

    def __init__(self, args=None):

        super().__init__()
        chanel_img = cfg.chanel_img
        model_cfg = cfg.sc_ch_dict[args.config] 
        # self.encoder = Encoder(args.config)
        self.encoder = Encoder(args.config)
        self.sigle_ll = False
        self.sigle_da = False

        self.caam = CAAM(feat_in=cfg.sc_ch_dict[args.config]['chanels'][2], num_classes=cfg.sc_ch_dict[args.config]['chanels'][2],bin_size =(2,4), norm_layer=nn.BatchNorm2d)
        self.conv_caam = ConvBatchnormRelu(cfg.sc_ch_dict[args.config]['chanels'][2],cfg.sc_ch_dict[args.config]['chanels'][1])

        self.up_1_da = UpConvBlock(cfg.sc_ch_dict[args.config]['chanels'][1],cfg.sc_ch_dict[args.config]['chanels'][0]) # out: Hx4, Wx4
        self.up_2_da = UpConvBlock(cfg.sc_ch_dict[args.config]['chanels'][0],8) #out: Hx2, Wx2
        self.out_da = UpConvBlock(8,2,last=True)  

        self.up_1_ll = UpConvBlock(cfg.sc_ch_dict[args.config]['chanels'][1],cfg.sc_ch_dict[args.config]['chanels'][0]) # out: Hx4, Wx4
        self.up_2_ll = UpConvBlock(cfg.sc_ch_dict[args.config]['chanels'][0],8) #out: Hx2, Wx2
        self.out_ll = UpConvBlock(8,2,last=True)


    def forward(self, input):
        '''
        :param input: RGB image
        :return: transformed feature map
        '''
        out_encoder,inp1,inp2=self.encoder(input)
        #visualize_feature_map_subset(out_encoder, "outencoder", 128)

        out_caam=self.caam(out_encoder)
        out_caam=self.conv_caam(out_caam)

        out_da=self.up_1_da(out_caam,inp2)
        out_da=self.up_2_da(out_da,inp1)
        out_da=self.out_da(out_da)

        out_ll=self.up_1_ll(out_caam,inp2)
        out_ll=self.up_2_ll(out_ll,inp1)
        out_ll=self.out_ll(out_ll)


        return out_da,out_ll

class TwinLiteNetPlus_V3(TwinLiteNetPlus):
    def __init__(self, args=None):
        super().__init__(args)
        self.encoder = Encoder_VMamba_V3(args.config)

class TwinLiteNetPlus_Vmamba2(TwinLiteNetPlus):
    def __init__(self, args=None):
        super().__init__(args)
        self.encoder = Encoder_Vmamba2(args.config)

class TwinLiteNetPlus_Vmamba2_V4(TwinLiteNetPlus):
    def __init__(self, args=None):
        super().__init__(args)
        self.encoder = Encoder_Vmamba2_V4(args.config)

class TwinLiteNetPlus_ConvNextV2(TwinLiteNetPlus):
    def __init__(self, args=None):
        super().__init__(args)
        self.encoder = Encoder_ConvNextV2(args.config)

def netParams(model):
    return np.sum([np.prod(parameter.size()) for parameter in model.parameters()])

class TwinLiteNetPlus_Lightweight(TwinLiteNetPlus):
    def __init__(self, args=None):
        super().__init__(args)
        self.encoder = LightweightEncoder(args.config)

        self.up_1_da = UpConvBlockFactorial(cfg.sc_ch_dict[args.config]['chanels'][1],cfg.sc_ch_dict[args.config]['chanels'][0]) # out: Hx4, Wx4
        self.up_2_da = UpConvBlockFactorial(cfg.sc_ch_dict[args.config]['chanels'][0],8) #out: Hx2, Wx2
        self.out_da = UpConvBlockFactorial(8,2,last=True)  

        self.up_1_ll = UpConvBlockFactorial(cfg.sc_ch_dict[args.config]['chanels'][1],cfg.sc_ch_dict[args.config]['chanels'][0]) # out: Hx4, Wx4
        self.up_2_ll = UpConvBlockFactorial(cfg.sc_ch_dict[args.config]['chanels'][0],8) #out: Hx2, Wx2
        self.out_ll = UpConvBlockFactorial(8,2,last=True)