import torch
import torch.nn as nn
import torch.nn.functional as F
import math

OPS={
    'none': lambda C, stride, affine: Zero(stride),
    'avg_pool_3x3': lambda C, stride, affine: nn.AvgPool2d(3,stride=stride,padding=1,count_include_pad=False),
    'max_pool_3x3': lambda C, stride, affine: nn.MaxPool2d(3,stride=stride,padding=1),
    'skip_connect': lambda C_in, C_out, stride, affine: Identity(),

    'sep_conv_3x3': lambda C_in, C_out, stride, affine: SepConv(C_in, C_out, 3, stride, 1, affine=affine),
    'sep_conv_5x5': lambda C_in, C_out, stride, affine: SepConv(C_in, C_out, 5, stride, 2, affine=affine),
    'sep_conv_7x7': lambda C_in, C_out, stride, affine: SepConv(C_in, C_out, 7, stride, 3, affine=affine),
    'dil_conv_3x3': lambda C_in, C_out, stride, affine: DilConv(C_in, C_out, 3, stride, 2, 2, affine=affine),
    'dil_conv_5x5': lambda C_in, C_out, stride, affine: DilConv(C_in, C_out, 5, stride, 4, 2, affine=affine),
    'conv_7x1_1x7': lambda C, stride, affine: nn.Sequential(
        nn.ReLU(inplace=False),
        nn.Conv2d(C, C, (1,7), stride=(1, stride), padding=(0,3), bias=False),
        nn.Conv2d(C, C, (7,1), stride=(stride, 1), padding=(3,0), bias=False),
        nn.BatchNorm2d(C, affine=affine)
    ),

    'conv_3x3_no_bn': lambda C_in, C_out, stride, affine: ReLUConv(C_in, C_out, 3, stride, 1, affine=affine),
    'conv_5x5_no_bn': lambda C_in, C_out, stride, affine: ReLUConv(C_in, C_out, 5, stride, 2, affine=affine),
    # 'rcab': lambda C_in, C_out, stride, affine: RCAB(C_in, C_out, 3, 3, bias=False, bn=False, act=nn.RELU(True), res_scale=1),
    'rcab': lambda C_in, C_out, stride, affine: RCAB( C_in, 3, 3, bias=True, bn=True, act=nn.ReLU(True), res_scale=1),#first 3 is kernel-size,the second 3 is the number of feature map
    # 'rcab': lambda C_in, C_out, stride, affine: RCAB(C_in, C_out, 3, 3,bias=False, bn=False, act=nn.ReLU(True), res_scale=1),#first 3 is kernel-size,the second 3 is the number of feature map
    'resb': lambda C_in, C_out, stride, affine: ResB(C_in, C_out, 3, bias=False, act=nn.ReLU(True), res_scale=1),

    # downsampling operation:
    'convolution_d': lambda C_in, C_out, stride, affine: Convolution(C_in, C_out, stride=2),
    'bilinear_d': lambda C_in, C_out, stride, affine: Bilinear(stride=0.5, C_in=C_in, C_out=C_out),
    'nearest_d': lambda C_in, C_out, stride, affine: Nearest(stride=0.5, C_in=C_in, C_out=C_out),
    'area_d': lambda C_in, C_out, stride, affine: Area(stride=0.5, C_in=C_in, C_out=C_out),

    # upsampling operation:
    'deconvolution_u': lambda C_in, C_out, stride, affine: Deconvolution(C_in, C_out, stride=2),
    'bilinear_u': lambda C_in, C_out, stride, affine: Bilinear(stride=2, C_in=C_in, C_out=C_out),
    'nearset_u': lambda C_in, C_out, stride, affine: Nearest(stride=2, C_in=C_in, C_out=C_out),
    'area_u': lambda C_in, C_out, stride, affine: Area(stride=2, C_in=C_in, C_out=C_out),
    'sub_pixel_u': lambda C_in, C_out, stride, affine: SUBPIXEL(C_in, C_out, scale_factor=2),
    
    
}


# -----------------------normal operation:---------------------------------------
def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class ReLUConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConv,self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size=kernel_size, stride=stride, padding=padding,bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )
    
    def forward(self, x):
        return self.op(x)

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, conv=default_conv, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(RCAB, self).__init__()
        modules_body=[]
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i ==0 : modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale
    
    def forward(self, x):
        res = self.body(x)
        # res = self.body(x).mul(self.res_scale)
        res += x
        return res

# x = torch.randn(128,64,40,40)
# C_in, C_out = 64, 64
# stride = 1
# model = OPS['rcab'](C_in,C_out,stride,True)
# # model  = RCAB(C_in, 3, 3, bias=True, bn=True, act=nn.ReLU(True), res_scale=1)
# x = model(x)
# print(x.shape)

## Residual Block
class ResB(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, conv=default_conv,bias=False, act=nn.ReLU(True), res_scale=1):
        super(ResB, self).__init__()
        self.C_in = C_in
        self.C_out = C_out
        modules_body=[]
        modules_body.append(nn.ReLU(inplace=False))
        for i in range(2):
            modules_body.append(conv(C_in, C_in, kernel_size, bias=bias))
            if i == 0: modules_body.append(act)
        self.op = nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False)
        self.body = nn.Sequential(*modules_body)
        

    def forward(self, x):
        # print(self.C_in,self.C_out)
        res = self.body(x)
        # res =  self.body(x).mul(self.res_scale)
        res += x
        return self.op(res) 

class DilConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace = False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                      groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )
    def forward(self, x):
        return self.op(x)

class SepConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            # 两次深度可分离卷积
            # 1次，out_channels= C_in
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=C_in, out_channels=C_in, kernel_size=kernel_size, stride=stride,padding=padding,groups=C_in,bias=False),
            nn.Conv2d(in_channels=C_in, out_channels=C_in, kernel_size=1, padding=0,bias=False),
            nn.BatchNorm2d(C_in, affine= affine),
            # 2次，out_channels=C_out
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size,stride=1, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1,padding=0,bias=False),
            nn.BatchNorm2d(C_out,affine=affine)

        )
    def forward(self, x):
        return self.op(x)
    

# x = torch.randn(128,64,40,40)
# model  = SepConv(64,32,5,1,2)
# x = model(x)
# print(x.shape)



class Identity(nn.Module):
    def __init__(self):
        super(Identity,self).__init__()

    def forward(self, x):
        return x

class Zero(nn.Module):
    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride=stride

    def forward(self, x):
        return x.mul(0.)
    

# -------------------downsampling operation:------------------------

class Convolution(nn.Module):
    def __init__(self, C_in, C_out, stride):
        super(Convolution, self).__init__()
        # stride: 2,3,4
        if stride == 2:
            kernel_size = 3
            output_padding = 1
        elif stride == 4:
            kernel_size = 5
            out_padding = 1
        else:
            kernel_size = 3
            output_padding = 0
        self.deconv = nn.Conv2d(C_in, C_out, kernel_size=kernel_size, stride=stride, padding=1)
    
    def forward(self, x):
        return self.deconv(x)
    
# bilinear
class Bilinear(nn.Module):
    def __init__(self, stride, C_in, C_out):
        super(Bilinear, self).__init__()
        self.scale = stride
        self.op = nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        return self.op(F.interpolate(x, scale_factor=self.scale, mode='bilinear'))

# linear
class Linear(nn.Module):
    def __init__(self, stride, C_in, C_out):
        super(Linear, self).__init__()
        self.scale = stride
        self.op = nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        return self.op(F.interpolate(x, scale_factor=self.scale, mode='linear'))

# nearest
class Nearest(nn.Module):
    def __init__(self,stride, C_in, C_out):
        super(Nearest, self).__init__()
        self.scale=stride
        self.op = nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False)
    
    def forward(self, x):
        return self.op(F.interpolate(x, scale_factor=self.scale, mode='nearest'))
    

# area
class Area(nn.Module):
    def __init__(self, stride, C_in, C_out):
        super(Area, self).__init__()
        self.scale = stride
        self.op = nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False)
    def forward(self, x):
        return self.op(F.interpolate(x, scale_factor=self.scale, mode='area'))


# ----------------------upsampling operation:------------------------------------

class Deconvolution(nn.Module):
    def __init__(self, C_in, C_out, stride):
        super(Deconvolution, self).__init__()
        # stride:2,3,4
        if stride==2:
            kernel_size = 3
            output_padding=1
        elif stride==4:
            kernel_size = 5
            output_padding=1
        else:
            kernel_size = 3
            output_padding=0
        self.deconv = nn.ConvTranspose2d(C_in, C_out, kernel_size=kernel_size,stride=stride,padding=1,output_padding=output_padding)
    

    def forward(self, x):
        return self.deconv(x)

class SUBPIXEL(nn.Module):
    def __init__(self, C_in, C_out, scale_factor, conv=default_conv):
        super(SUBPIXEL, self).__init__()
        self.upsample = Upsampler(conv, scale_factor, C_in, act=False)
        self.op = nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False)
    
    def forward(self, x):
        return self.op(self.upsample(x))
        # return self.upsample(x)




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


# test sunbpiXEL

# sub = SUBPIXEL(3,3,scale_factor=3)
# # torch.randn()
# x = torch.randn(128, 3, 20, 20)
# print(x.shape)
# x = sub(x)
# print(x.shape)