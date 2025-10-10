import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import sys
from model import config as cfg 
import matplotlib.pyplot as plt
from torch.nn import Module, Conv2d, Parameter, Softmax
import cv2
import os
import math
from .vmamba.models.vmamba import VSSBlock, SS2D, LayerNorm2d, SwiGLU


class IVSSBlock(VSSBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.mlp_branch:
            mlp_hidden_dim = int(kwargs['hidden_dim'] * kwargs['mlp_ratio'])
            self.mlp = SwiGLU(
                in_features=kwargs['hidden_dim'],
                hidden_features=mlp_hidden_dim,
                act_layer=kwargs['mlp_act_layer'],
                drop=kwargs['mlp_drop_rate'],
                channels_first=kwargs['channel_first'],
                norm_layer=kwargs['norm_layer'],
                subln=kwargs['subln'],
            )

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


class DilatedConv(nn.Module):
    '''
    This class defines the dilated convolution.
    '''

    def __init__(self, nIn, nOut, kSize, stride=1, d=1, groups=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        '''
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        self.conv = nn.Conv2d(nIn, nOut,kSize, stride=stride, padding=padding, bias=False,
                              dilation=d, groups=groups)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        return output

class BatchnormRelu(nn.Module):
    '''
        This class groups the batch normalization and PReLU activation
    '''
    def __init__(self, nOut):
        '''
        :param nOut: output feature maps
        '''
        super().__init__()
        self.nOut=nOut
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: normalized and thresholded feature map
        '''
        output = self.bn(input)
        output = self.act(output)
        return output


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, nin, nout, kernel_size=3, stride=1, dilation=1):
        super(DepthwiseSeparableConv, self).__init__()
        padding = int((kernel_size - 1) / 2) * dilation
        self.depthwise = nn.Conv2d(nin, nin, kernel_size, stride, padding, dilation, groups=nin, bias=False)
        self.pointwise = nn.Conv2d(nin, nout, 1, 1, 0, 1, 1, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class StrideESP(nn.Module):
    def __init__(self, nIn, nOut):
        super().__init__()
        n = int(nOut/5)
        n1 = nOut - 4*n
        self.c1 = DilatedConv(nIn, n, 3, 2)
        self.d1 = DilatedConv(n, n1, 3, 1, 1)
        self.d2 = DilatedConv(n, n, 3, 1, 2)
        self.d4 = DilatedConv(n, n, 3, 1, 4)
        self.d8 = DilatedConv(n, n, 3, 1, 8)
        self.d16 = DilatedConv(n, n, 3, 1, 16)
        self.bn = nn.BatchNorm2d(nOut, eps=1e-3)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        output1 = self.c1(input)
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d4 = self.d4(output1)
        d8 = self.d8(output1)
        d16 = self.d16(output1)

        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8
        add4 = add3 + d16

        combine = torch.cat([d1, add1, add2, add3, add4],1)
        output = self.bn(combine)
        output = self.act(output)
        return output




class DepthwiseESP(nn.Module):
    '''
    This class defines the ESP block, which is based on the following principle
        Reduce ---> Split ---> Transform --> Merge
    '''
    def __init__(self, nIn, nOut, add=True):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param add: if true, add a residual connection through identity operation. You can use projection too as
                in ResNet paper, but we avoid to use it if the dimensions are not the same because we do not want to
                increase the module complexity
        '''
        super().__init__()
        n = max(int(nOut/5),1)
        n1 = max(nOut - 4*n,1)
        self.c1 = DepthwiseSeparableConv(nIn, n, 1, 1)
        self.d1 = DepthwiseSeparableConv(n, n1, 3, 1, 1) # dilation rate of 2^0
        self.d2 = DepthwiseSeparableConv(n, n, 3, 1, 2) # dilation rate of 2^1
        self.d4 = DepthwiseSeparableConv(n, n, 3, 1, 4) # dilation rate of 2^2
        self.d8 = DepthwiseSeparableConv(n, n, 3, 1, 8) # dilation rate of 2^3
        self.d16 = DepthwiseSeparableConv(n, n, 3, 1, 16) # dilation rate of 2^4
        self.bn = BatchnormRelu(nOut)
        self.add = add

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        # reduce
        output1 = self.c1(input)
        # split and transform
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d4 = self.d4(output1)
        d8 = self.d8(output1)
        d16 = self.d16(output1)
        # heirarchical fusion for de-gridding
        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8
        add4 = add3 + d16
        #merge
        combine = torch.cat([d1, add1, add2, add3, add4], 1)
        if self.add:
            combine = input + combine
        output = self.bn(combine)
        return output

class AvgDownsampler(nn.Module):
    '''
    This class projects the input image to the same spatial dimensions as the feature map.
    For example, if the input image is 512 x512 x3 and spatial dimensions of feature map size are 56x56xF, then
    this class will generate an output of 56x56x3
    '''
    def __init__(self, samplingTimes):
        '''
        :param samplingTimes: The rate at which you want to down-sample the image
        '''
        super().__init__()
        self.pool = nn.ModuleList()
        for i in range(0, samplingTimes):
            #pyramid-based approach for down-sampling
            self.pool.append(nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, input):
        '''
        :param input: Input RGB Image
        :return: down-sampled image (pyramid-based approach)
        '''
        for pool in self.pool:
            input = pool(input)
        return input


class Encoder(nn.Module):
    '''
    This class defines the ESPNet-C network in the paper
    '''
    def __init__(self, config):
        super().__init__()
        chanel_img = cfg.chanel_img
        model_cfg = cfg.sc_ch_dict[config] 
        self.level1 = ConvBatchnormRelu(chanel_img, model_cfg['chanels'][0], stride = 2)
        self.sample1 = AvgDownsampler(1)
        self.sample2 = AvgDownsampler(2)

        self.channel_first = (model_cfg['norm_layer'].lower() in ["bn", "ln2d"])

        dpr = [
            x.item() 
            for x in torch.linspace(
                0, model_cfg['drop_path_rate'], model_cfg['p'] + model_cfg['q']
            )
        ]  # stochastic depth decay rule

        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )

        _ACTLAYERS = dict(
            silu=nn.SiLU, 
            gelu=nn.GELU, 
            relu=nn.ReLU, 
            sigmoid=nn.Sigmoid,
        )

        norm_layer: nn.Module = _NORMLAYERS.get(model_cfg['norm_layer'].lower(), None)
        ssm_act_layer: nn.Module = _ACTLAYERS.get(model_cfg['ssm_act_layer'].lower(), None)
        mlp_act_layer: nn.Module = _ACTLAYERS.get(model_cfg['mlp_act_layer'].lower(), None)

        self.b1 = ConvBatchnormRelu(model_cfg['chanels'][0] + chanel_img,model_cfg['chanels'][1])
        self.level2_0 = StrideESP(model_cfg['chanels'][1], model_cfg['chanels'][2])

        self.level2 = self._make_layer(
            dim=model_cfg['chanels'][2],
            drop_path=dpr[:model_cfg['p']],
            use_checkpoint=model_cfg['use_checkpoint'],
            norm_layer=norm_layer,
            channel_first=self.channel_first,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            ssm_d_state=model_cfg['ssm_d_state'],
            ssm_ratio=model_cfg['ssm_ratio'],
            ssm_dt_rank=model_cfg['ssm_dt_rank'],
            ssm_conv=model_cfg['ssm_conv'],
            ssm_conv_bias=model_cfg['ssm_conv_bias'],
            ssm_drop_rate=model_cfg['ssm_drop_rate'],
            ssm_init=model_cfg['ssm_init'],
            forward_type=model_cfg['forward_type'],
            mlp_ratio=model_cfg['mlp_ratio'],
            mlp_drop_rate=model_cfg['mlp_drop_rate'],
            gmlp=model_cfg['gmlp'],
        )
        # self.level2 = nn.ModuleList()
        # for i in range(0, model_cfg['p']):
            # self.level2.append(DepthwiseESP(model_cfg['chanels'][2] , model_cfg['chanels'][2]))
        self.b2 = ConvBatchnormRelu(model_cfg['chanels'][3] + chanel_img,model_cfg['chanels'][3] + chanel_img)

        self.level3_0 = StrideESP(model_cfg['chanels'][3] + chanel_img, model_cfg['chanels'][3])
        self.level3 = self._make_layer(
            dim=model_cfg['chanels'][3],
            drop_path=dpr[model_cfg['p']:model_cfg['p']+model_cfg['q']],
            use_checkpoint=model_cfg['use_checkpoint'],
            norm_layer=norm_layer,
            channel_first=self.channel_first,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            ssm_d_state=model_cfg['ssm_d_state'],
            ssm_ratio=model_cfg['ssm_ratio'],
            ssm_dt_rank=model_cfg['ssm_dt_rank'],
            ssm_conv=model_cfg['ssm_conv'],
            ssm_conv_bias=model_cfg['ssm_conv_bias'],
            ssm_drop_rate=model_cfg['ssm_drop_rate'],
            ssm_init=model_cfg['ssm_init'],
            forward_type=model_cfg['forward_type'],
            mlp_ratio=model_cfg['mlp_ratio'],
            mlp_drop_rate=model_cfg['mlp_drop_rate'],
            gmlp=model_cfg['gmlp'],
        )
        # self.level3 = nn.ModuleList()
        # for i in range(0, model_cfg['q']):
            # self.level3.append(DepthwiseESP(model_cfg['chanels'][3] , model_cfg['chanels'][3]))
        self.b3 = ConvBatchnormRelu(model_cfg['chanels'][4],model_cfg['chanels'][2])
        
    def forward(self, input):
        '''
        :param input: Receives the input RGB image
        :return: the transformed feature map with spatial dimensions 1/8th of the input image
        '''
        output0 = self.level1(input)
        inp1 = self.sample1(input)
        inp2 = self.sample2(input)
        output0_cat = self.b1(torch.cat([output0, inp1], 1))
        output1_0 = self.level2_0(output0_cat) # down-sampled
        
        for i, layer in enumerate(self.level2):
            if i==0:
                output1 = layer(output1_0)
            else:
                output1 = layer(output1)

        output1_cat = self.b2(torch.cat([output1,  output1_0, inp2], 1))
        output2_0 = self.level3_0(output1_cat)
        for i, layer in enumerate(self.level3):
            if i==0:
                output2 = layer(output2_0)
            else:
                output2 = layer(output2)
        output2_cat=torch.cat([output2_0, output2], 1)
        out_encoder = self.b3(output2_cat)
        
        return out_encoder,inp1,inp2

    @staticmethod
    def _make_layer(
        dim=96, 
        drop_path=[0.1, 0.1], 
        use_checkpoint=False, 
        norm_layer=nn.LayerNorm,
        channel_first=False,
        # ===========================
        ssm_d_state=16,
        ssm_ratio=2.0,
        ssm_dt_rank="auto",       
        ssm_act_layer=nn.SiLU,
        ssm_conv=3,
        ssm_conv_bias=True,
        ssm_drop_rate=0.0, 
        ssm_init="v0",
        forward_type="v2",
        # ===========================
        mlp_ratio=4.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate=0.0,
        gmlp=False,
        # ===========================
        _SS2D=SS2D,
        **kwargs,
    ):
        # if channel first, then Norm and Output are both channel_first
        depth = len(drop_path)
        blocks = []
        for d in range(depth):
            blocks.append(VSSBlock(
                hidden_dim=dim, 
                drop_path=drop_path[d],
                norm_layer=norm_layer,
                channel_first=channel_first,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                gmlp=gmlp,
                use_checkpoint=use_checkpoint,
                _SS2D=_SS2D,
            ))
        
        return nn.Sequential(*blocks,)

class Encoder_V2(nn.Module):
    '''
    This class defines the VMamba-SwiGLU + StrideESP network
    '''
    def __init__(self, config):
        super().__init__()
        chanel_img = cfg.chanel_img
        model_cfg = cfg.sc_ch_dict[config] 
        self.level1 = ConvBatchnormRelu(chanel_img, model_cfg['chanels'][0], stride = 2)
        self.sample1 = AvgDownsampler(1)
        self.sample2 = AvgDownsampler(2)

        self.channel_first = (model_cfg['norm_layer'].lower() in ["bn", "ln2d"])

        dpr = [
            x.item() 
            for x in torch.linspace(
                0, model_cfg['drop_path_rate'], model_cfg['p'] + model_cfg['q']
            )
        ]  # stochastic depth decay rule

        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )

        _ACTLAYERS = dict(
            silu=nn.SiLU, 
            gelu=nn.GELU, 
            relu=nn.ReLU, 
            sigmoid=nn.Sigmoid,
        )

        norm_layer: nn.Module = _NORMLAYERS.get(model_cfg['norm_layer'].lower(), None)
        ssm_act_layer: nn.Module = _ACTLAYERS.get(model_cfg['ssm_act_layer'].lower(), None)
        mlp_act_layer: nn.Module = _ACTLAYERS.get(model_cfg['mlp_act_layer'].lower(), None)

        self.b1 = ConvBatchnormRelu(model_cfg['chanels'][0] + chanel_img,model_cfg['chanels'][1])
        self.level2_0 = StrideESP(model_cfg['chanels'][1], model_cfg['chanels'][2])

        self.level2 = self._make_layer(
            dim=model_cfg['chanels'][2],
            drop_path=dpr[:model_cfg['p']],
            use_checkpoint=model_cfg['use_checkpoint'],
            norm_layer=norm_layer,
            channel_first=self.channel_first,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            ssm_d_state=model_cfg['ssm_d_state'],
            ssm_ratio=model_cfg['ssm_ratio'],
            ssm_dt_rank=model_cfg['ssm_dt_rank'],
            ssm_conv=model_cfg['ssm_conv'],
            ssm_conv_bias=model_cfg['ssm_conv_bias'],
            ssm_drop_rate=model_cfg['ssm_drop_rate'],
            ssm_init=model_cfg['ssm_init'],
            forward_type=model_cfg['forward_type'],
            mlp_ratio=model_cfg['mlp_ratio'],
            mlp_drop_rate=model_cfg['mlp_drop_rate'],
            gmlp=model_cfg['gmlp'],
            subln=model_cfg['subln'],
        )
        self.b2 = ConvBatchnormRelu(model_cfg['chanels'][3] + chanel_img,model_cfg['chanels'][3] + chanel_img)

        self.level3_0 = StrideESP(model_cfg['chanels'][3] + chanel_img, model_cfg['chanels'][3])
        self.level3 = self._make_layer(
            dim=model_cfg['chanels'][3],
            drop_path=dpr[model_cfg['p']:model_cfg['p']+model_cfg['q']],
            use_checkpoint=model_cfg['use_checkpoint'],
            norm_layer=norm_layer,
            channel_first=self.channel_first,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            ssm_d_state=model_cfg['ssm_d_state'],
            ssm_ratio=model_cfg['ssm_ratio'],
            ssm_dt_rank=model_cfg['ssm_dt_rank'],
            ssm_conv=model_cfg['ssm_conv'],
            ssm_conv_bias=model_cfg['ssm_conv_bias'],
            ssm_drop_rate=model_cfg['ssm_drop_rate'],
            ssm_init=model_cfg['ssm_init'],
            forward_type=model_cfg['forward_type'],
            mlp_ratio=model_cfg['mlp_ratio'],
            mlp_drop_rate=model_cfg['mlp_drop_rate'],
            gmlp=model_cfg['gmlp'],
            subln=model_cfg['subln'],
        )
        self.b3 = ConvBatchnormRelu(model_cfg['chanels'][4],model_cfg['chanels'][2])

        self.apply(self._init_weights)

        
    def forward(self, input):
        '''
        :param input: Receives the input RGB image
        :return: the transformed feature map with spatial dimensions 1/8th of the input image
        '''
        output0 = self.level1(input)
        inp1 = self.sample1(input)
        inp2 = self.sample2(input)
        output0_cat = self.b1(torch.cat([output0, inp1], 1))
        output1_0 = self.level2_0(output0_cat) # down-sampled
        
        for i, layer in enumerate(self.level2):
            if i==0:
                output1 = layer(output1_0)
            else:
                output1 = layer(output1)

        output1_cat = self.b2(torch.cat([output1,  output1_0, inp2], 1))
        output2_0 = self.level3_0(output1_cat)
        for i, layer in enumerate(self.level3):
            if i==0:
                output2 = layer(output2_0)
            else:
                output2 = layer(output2)
        output2_cat=torch.cat([output2_0, output2], 1)
        out_encoder = self.b3(output2_cat)
        
        return out_encoder,inp1,inp2

    @staticmethod
    def _make_layer(
        dim=96, 
        drop_path=[0.1, 0.1], 
        use_checkpoint=False, 
        norm_layer=nn.LayerNorm,
        channel_first=False,
        # ===========================
        ssm_d_state=16,
        ssm_ratio=2.0,
        ssm_dt_rank="auto",       
        ssm_act_layer=nn.SiLU,
        ssm_conv=3,
        ssm_conv_bias=True,
        ssm_drop_rate=0.0, 
        ssm_init="v0",
        forward_type="v2",
        # ===========================
        mlp_ratio=4.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate=0.0,
        gmlp=False,
        subln=False,
        # ===========================
        _SS2D=SS2D,
        **kwargs,
    ):
        # if channel first, then Norm and Output are both channel_first
        depth = len(drop_path)
        blocks = []
        for d in range(depth):
            blocks.append(IVSSBlock(
                hidden_dim=dim, 
                drop_path=drop_path[d],
                norm_layer=norm_layer,
                channel_first=channel_first,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                subln=subln,
                gmlp=gmlp,
                use_checkpoint=use_checkpoint,
                _SS2D=_SS2D,
            ))
        
        return nn.Sequential(*blocks,)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)