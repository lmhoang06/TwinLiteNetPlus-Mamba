import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from model import config as cfg 
from torch.nn import Module, Conv2d, Parameter, Softmax


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


class DilatedConvFactorial(nn.Module):
    '''
    Dilated Convolution using factorial conv with 1xk followed by kx1 convolutions.
    '''

    def __init__(self, nIn, nOut, kSize, stride=1, d=1, groups=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        :param groups: number of groups
        '''
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        # First: 1xk
        self.conv1 = nn.Conv2d(
            nIn, nOut, 
            kernel_size=(1, kSize), 
            stride=(1, stride),
            padding=(0, padding), 
            dilation=(1, d), 
            bias=False, 
            groups=groups
        )
        # Second: kx1
        self.conv2 = nn.Conv2d(
            nOut, nOut, 
            kernel_size=(kSize, 1), 
            stride=(stride, 1),
            padding=(padding, 0), 
            dilation=(d, 1), 
            bias=False, 
            groups=groups
        )

    def forward(self, input):
        output = self.conv2(self.conv1(input))
        return output


class StrideESPFactorial(StrideESP):
    def __init__(self, nIn, nOut):
        super().__init__(nIn, nOut)
        n = int(nOut/5)
        n1 = nOut - 4*n
        self.c1 = DilatedConvFactorial(nIn, n, 3, 2)
        self.d1 = DilatedConvFactorial(n, n1, 3, 1, 1)
        self.d2 = DilatedConvFactorial(n, n, 3, 1, 2)
        self.d4 = DilatedConvFactorial(n, n, 3, 1, 4)
        self.d8 = DilatedConvFactorial(n, n, 3, 1, 8)
        self.d16 = DilatedConvFactorial(n, n, 3, 1, 16)
