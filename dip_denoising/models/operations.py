
import torch.nn as nn
import torch
import torch.nn.functional as F
import math

import os
import sys
# dir = os.getcwd()
# sys.path.append(dir)


# from evaluate.dip_denoising.models.common import *
from .common import *

OPS = {
    # conv
    'conv_3x3':lambda C_in, C_out: Conv(C_in,C_out, kernel_size = 3, padding=1),
    'conv_5x5':lambda C_in, C_out: Conv(C_in,C_out, kernel_size = 5, padding=2),
    'conv_7x7':lambda C_in, C_out: Conv(C_in, C_out, kernel_size = 7, padding=3),

    'sep_conv_3x3':lambda C_in, C_out: SepConv(C_in, C_out,kernel_size=3, stride=1, padding=1),
    'sep_conv_5x5': lambda C_in, C_out: SepConv(C_in, C_out, kernel_size=5, stride=1, padding=2),
    'sep_conv_7x7': lambda C_in, C_out: SepConv(C_in, C_out, kernel_size=7, stride=1, padding=3),

    'dil_conv_3x3': lambda C_in, C_out: DilConv(C_in, C_out, kernel_size = 3, stride = 1,padding =2, dilation=2),
    'dil_conv_5x5': lambda C_in, C_out: DilConv(C_in, C_out, kernel_size=5, stride= 1, padding=4, dilation=2 ),

    'conv_3x3_leakyReLU':lambda C_in, C_out: conv_block_sp(C_in, C_out, act_fun="soft"),
    'conv_block_last': lambda C_in, C_out: conv_block_last(C_in, C_out, act_fun="soft"),
    'bn':lambda C_in:bn(C_in, mode='bn'),

    # downsample
    'maxpool2d': lambda C_in, C_out: MaxPool(C_in, C_out, kernel_size=3, stride=2, padding=1),
    'convolution_d': lambda C_in, C_out: ConvDown(C_in, C_out, kernel_size=3, stride=2, act_fun="soft"),
    'bilinear_d': lambda C_in, C_out: Bilinear(stride=0.5, C_in=C_in, C_out=C_out),
    'nearest_d': lambda C_in, C_out: Nearest(stride=0.5, C_in=C_in, C_out=C_out),
    # 'bilinear_d': lambda C_in, C_out: nn.Upsample(scale_factor=0.5, mode='bilinear'),
    # 'nearest_d': lambda C_in, C_out: nn.Upsample(scale_factor=0.5,mode='nearest'),
    'area_d': lambda C_in, C_out: Area(stride=0.5, C_in=C_in, C_out=C_out),

    
    # 'maxpool2d': lambda: nn.MaxPool2d(2)

    # upsmaple
    'deconv': lambda C_in, C_out: nn.ConvTranspose2d(C_in, C_out, kernel_size=3, stride=2,padding=1, output_padding=1 ),
    'sub_pixel_u': lambda C_in, C_out: SUBPIXEL(C_in, C_out, scale_factor=2),
    'bilinear_u': lambda C_in, C_out: Bilinear(stride=2, C_in=C_in, C_out=C_out),
    # 'bilinear_u': lambda C_in, C_out: nn.Upsample(scale_factor=2, mode='bilinear'),
    
    # 'nearset_u': lambda C_in, C_out: nn.Upsample(scale_factor=2, mode='nearest'),
    'nearest_u': lambda C_in, C_out: Nearest(stride=2, C_in=C_in, C_out=C_out),
    'area_u': lambda C_in, C_out: Area(stride=2, C_in=C_in, C_out=C_out),
    

}




class conv_block_last(nn.Module):
    def __init__(self, ch_in, ch_out, down=False, act_fun='LeakyReLU', pad='reflection', group=1, bn_mode = "bn", bias=True):
        super(conv_block_last, self).__init__()
    
        self.conv1 = conv(ch_in, 64, kernel_size=3, stride=1 if down is False else 2, bias=bias, pad=pad, group=group)
        self.conv2 = conv(64, 32, kernel_size=3, stride=1, bias=bias, pad=pad, group=group)
        self.conv3 = conv(32, ch_out, kernel_size=3, stride=1, bias=bias, pad=pad, group=group)
        self.conv = nn.Sequential(
            self.conv1, bn(64, bn_mode if group == 1 else "groupNorm"), act(act_fun),
            self.conv2, bn(32, bn_mode if group == 1 else "groupNorm"), act(act_fun),
            self.conv3)

    def forward(self, x):
        x = self.conv(x)
        return x

class conv_block_sp(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size = 3,  down=False, act_fun='LeakyReLU', pad='reflection', group=1, bn_mode = "bn", bias=True):
        super(conv_block_sp, self).__init__()
        # self.conv1 = conv(ch_in, ch_out, kernel_size=3, stride=1 if down is False else 2, bias=bias, pad=pad,
        #                   group=group) # down
        self.conv1 = conv(ch_in, ch_out, kernel_size=kernel_size, stride=1, bias=bias, pad=pad, group=group)
        self.conv = nn.Sequential(
            self.conv1, bn(ch_out, bn_mode if group == 1 else "groupNorm"), act(act_fun))

    def forward(self, x):
        x = self.conv(x)
        return x
    

# x = torch.randn(128, 64, 40, 40)
# C_in, C_out = 64, 64
# # model = OPS['conv_5x5'](C_in,C_out)
# model = OPS['conv_3x3_leakyRelU'](C_in, C_out)
# # model = Conv(C_in, C_out)
# x = model(x)

# print(x.shape)


class Conv(nn.Module):
    def __init__(self,C_in, C_out, kernel_size, padding):
        super(Conv,self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size= kernel_size, padding=padding, bias= False),
            nn.BatchNorm2d(C_out),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        out = self.op(x)
        return out
    
def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)


class SepConv1(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding,affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            # 两次深度可分离卷积
            # 1次，out_channels= C_in
            # nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=C_in, out_channels=C_in, kernel_size=kernel_size, stride=stride,padding=padding,groups=C_in,bias=False),
            nn.Conv2d(in_channels=C_in, out_channels=C_in, kernel_size=1, padding=0,bias=False),
            nn.BatchNorm2d(C_in, affine=affine),
            # 2次，out_channels=C_out
            nn.ReLU(inplace=True),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size,stride=1, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1,padding=0,bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
            nn.ReLU(inplace=True)

        )
    def forward(self, x):
        return self.op(x)
    
class SepConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding,affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            # 两次深度可分离卷积
            # 1次，out_channels= C_in
            # nn.ReLU(inplace=False),
            # nn.Conv2d(in_channels=C_in, out_channels=C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
            # nn.Conv2d(in_channels=C_in, out_channels=C_in, kernel_size=1, padding=0, bias=False),
            # nn.BatchNorm2d(C_in, affine=affine),
            # # 2次，out_channels=C_out
            # nn.ReLU(inplace=True),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size,stride=stride, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1,padding=0,bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
            nn.ReLU(inplace=True)

        )
        
    def forward(self, x):
        return self.op(x)


class DilConv1(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(
            # nn.ReLU(inplace = False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, 
                      groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
            nn.ReLU(inplace = True)
        )
    def forward(self, x):
        return self.op(x)

class DilConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(
            # nn.ReLU(inplace = False),
            nn.Conv2d(C_in, C_out, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False),
            # nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
            nn.ReLU(inplace = True)
        )
    def forward(self, x):
        return self.op(x)


# ====================downsample===================


class MaxPool(nn.Module):
    def __init__(self, C_in, C_out, kernel_size=3, stride=2, padding=1):
        super(MaxPool, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv =  nn.Conv2d(C_in, C_out,kernel_size=1)
        self.bn = bn(C_out)
    
    def forward(self, x):
        x = self.maxpool(x)
        x = self.conv(x)
        out = self.bn(x)

        return out
    
class ConvDown(nn.Module):
    def __init__(self, C_in, C_out, kernel_size=3, stride=2, pad='reflection', act_fun = 'LeakyReLU' ):
        super(ConvDown, self).__init__()

        # self.down_conv = conv(C_in, C_out, kernel_size=3, stride=2, bias=True, pad=pad)
        self.down_conv = conv(C_in, C_out, kernel_size=3, stride=2, bias=True, pad=pad, group=1) # down
        self.bn = bn(C_out, "bn")
        self.act = act(act_fun)
        self.op = nn.Sequential(self.down_conv, self.bn, self.act)
    
    def forward(self, x):
        return self.op(x)
    




# bilinear
class Bilinear(nn.Module):
    def __init__(self, stride, C_in, C_out):
        super(Bilinear, self).__init__()
        self.scale = stride
        self.op = nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        # return F.interpolate(x, scale_factor=self.scale, mode='bilinear')
        return self.op(F.interpolate(x, scale_factor=self.scale, mode='bilinear'))

# linear
class Linear(nn.Module):
    def __init__(self, stride, C_in, C_out):
        super(Linear, self).__init__()
        self.scale = stride
        self.op = nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        # return F.interpolate(x, scale_factor=self.scale, mode='linear')
        return self.op(F.interpolate(x, scale_factor=self.scale, mode='linear'))

# nearest
class Nearest(nn.Module):
    def __init__(self,stride, C_in, C_out):
        super(Nearest, self).__init__()
        self.scale=stride
        self.op = nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False)
    
    def forward(self, x):
        # return F.interpolate(x, scale_factor=self.scale, mode='nearest')
        return self.op(F.interpolate(x, scale_factor=self.scale, mode='nearest'))
    


# area
class Area(nn.Module):
    def __init__(self, stride, C_in, C_out):
        super(Area, self).__init__()
        self.scale = stride
        self.op = nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False)
    def forward(self, x):
        # return F.interpolate(x, scale_factor=self.scale, mode='area')
        return self.op(F.interpolate(x, scale_factor=self.scale, mode='area'))


class SUBPIXEL(nn.Module):
    def __init__(self, C_in, C_out, scale_factor, conv=default_conv):
        super(SUBPIXEL, self).__init__()
        self.upsample = Upsampler(conv, scale_factor, C_in, act=False)
        self.op = nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False)
    
    def forward(self, x):
        return self.op(self.upsample(x))
        # return self.upsample(x)


#=================UpSampler=====================

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):
        m = []
        if(scale & (scale-1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale,2))):
                m.append(conv(n_feats,4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))
        
        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelUnshuffle(3))

            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))

        else:
            raise NotImplementedError
        super(Upsampler,self).__init__(*m)



# x = torch.randn(1, 48, 512, 512)
# C_in, C_out = 48, 48

# "conv_3x3", "conv_5x5", "sep_conv_3x3", "sep_conv_5x5", "sep_conv_7x7", "dil_conv_3x3","dil_conv_5x5"


# model = OPS['conv_5x5'](C_in, C_out)
# model2 = OPS['sep_conv_3x3'](C_in, C_out)

# # model3 = OPS['bilinear_u2'](C_in, C_out)
# model3 = OPS['sep_conv_5x5'](C_in, C_out)
# model4 = OPS['sep_conv_7x7'](C_in, C_out)

# model5 = OPS['dil_conv_3x3'](C_in, C_out)
# model6 = OPS['dil_conv_5x5'](C_in, C_out)
# # # model = OPS['conv_5x5'](C_in,C_out)
# # model = OPS['deconv'](C_in, C_out)
# # model2 = OPS['bilinear_d'](C_in, C_out)

# # # model3 = OPS['bilinear_u2'](C_in, C_out)
# # model3 = OPS['nearest_d'](C_in, C_out)
# # model4 = OPS['sub_pixel_u'](C_in, C_out)

# # model5 = OPS['area_d'](C_in, C_out)
# # model3 = OPS['bilinear_d2'](C_in, C_out)
# # model4 = OPS['bilinear_d'](C_in, C_out)
# # model5 = OPS['nearest_d'](C_in, C_out)

# # model6 = OPS['bilinear_d2'](C_in, C_out)
# # model7 = OPS['nearest_d2'](C_in, C_out)

# # model = Conv(C_in, C_out)
# x1 = model(x)
# x2 = model2(x)
# x3 = model3(x)
# x4 = model4(x)
# x5 = model5(x)
# x6 = model6(x)

# # # x6 = model6(x)
# # # x7 = model7(x)

# print("x1",x1.shape)
# # print("x1" , x1)

# print("x2", x2.shape)
# # print("x2" , x2)

# print("x3", x3.shape)
# # print("x3" , x3)

# print("x4", x4.shape)
# # print("x4" , x4)

# print("x5", x5.shape)

# print("x6", x6.shape)
# # print("x5" , x5)

# # print("x", x.shape)
# # print("x" , x)

# # # print("x6", x6.shape)
# # # print("x6" , x6)

# # # print("x7", x7.shape)
# # # print("x7" , x7)


# # print("x3",x3.shape)
# # print("x3" , x3)

# # print("x4",x4.shape)
# # print("x4" , x4)
