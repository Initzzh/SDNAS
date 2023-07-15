import os
import sys 

# dir = os.getcwd()
# dir = dir+'/dip-denoising'
# print(dir)
# sys.path.append(dir)
# print(sys.path)
# from evaluate.dip_denoising.models.operations import OPS
# from evaluate.dip_denoising.models.operations import OPS
from .operations import OPS
import numpy as np
import torch.nn as nn
import torch 

from .common import *


# class ConvBlock(nn.Module):
#     def __init__(self, block_amount, features, conv_type):
#         super(ConvBlock, self).__init__()
#         layers = []
#         for i in range(block_amount):
#             layers.append(nn.Conv2d(in_channels = features,out_channels= features, kernel_size=3,padding=1, bias= False)) # Conv
#             layers.append(nn.BatchNorm2d(features))
#             layers.append(nn.ReLU(inplace=True))
        
#         self.conv_block = nn.Sequential(*layers)
    
#     def forward(self, x):
#         out = self.conv_block(x)
#         return out

class ConvBlock(nn.Module):
    """
        Encoder/Decoder 的卷积块部分
    """
    def __init__(self, block_amount, features, conv_type):
        super(ConvBlock, self).__init__()
        layers = []
        for i in range(block_amount):
            layers.append(OPS[conv_type](features,features))
        self.conv_block = nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv_block(x)
        return out




class DownBlock(nn.Module):
    def __init__(self, features, downsample_type):
        super(DownBlock, self).__init__()
        self.down_block = OPS[downsample_type](features,features)
        
    def forward(self, x):
        out = self.down_block(x)
        return out
    
class UpBlock(nn.Module):
    def __init__(self, features, upsample_type):
        super(UpBlock, self).__init__()
        
        self.up_block = OPS[upsample_type](features, features)
        self.bn = bn(features)

    
    def forward(self, x):

        out  = self.up_block(x)
        out = self.bn(out)
        return out 
    

class Concat_layer(nn.Module):
    def __init__(self, dim):
        super(Concat_layer, self).__init__()
        self.dim = dim

    def forward(self, inputs):
        inputs_shapes2 = [x.shape[2] for x in inputs]
        inputs_shapes3 = [x.shape[3] for x in inputs]

        # 让连个特征图的size 统一，大的特征图的size 变成小的特征图
        if np.all(np.array(inputs_shapes2) == min(inputs_shapes2)) and np.all(
                np.array(inputs_shapes3) == min(inputs_shapes3)):
            inputs_ = inputs
        else:
            target_shape2 = min(inputs_shapes2)
            target_shape3 = min(inputs_shapes3)
            inputs_ = []
            for inp in inputs:
                diff2 = (inp.size(2) - target_shape2) // 2
                diff3 = (inp.size(3) - target_shape3) // 2
                inputs_.append(inp[:, :, diff2: diff2 + target_shape2, diff3:diff3 + target_shape3])
        return torch.cat(inputs_, dim=self.dim)

    def __len__(self):
        return len(self._modules)

# Unet (concat)
class Unet(nn.Module):
    def __init__(self, indi, in_channels = 1, features = 48):
        super(Unet, self).__init__()
        self.concat = Concat_layer(dim = 1)
        self.level_amount = indi.level_amount
        self.middle_unit_amount = indi.middle_unit_amount

        self.encoder_units = indi.encoder_units # indi 的encoder_units
        self.decoder_units = indi.decoder_units 
        self.middle_units =  indi.middle_units

        self.encoder_conv_blocks = nn.ModuleList() 
        self.dsl_blocks = nn.ModuleList()

        self.decoder_conv_blocks =  nn.ModuleList()
        self.usl_blocks = nn.ModuleList()

        self.middle_conv_blocks = nn.ModuleList()

        self.first_conv =  nn.Conv2d(in_channels=in_channels, out_channels=features, kernel_size=3, padding=1, bias=False )
        
        for level in range(self.level_amount):
            
            # encoder
            self.encoder_conv_blocks.append(
                ConvBlock(self.encoder_units[level].block_amount, self.encoder_units[level].features, self.encoder_units[level].conv_type)
                )
            self.dsl_blocks.append(
                DownBlock(self.encoder_units[level].features, self.encoder_units[level].downsample_type)
            )

            # decoder
            self.decoder_conv_blocks.append(
                ConvBlock(self.encoder_units[level].block_amount, self.decoder_units[level].features, self.decoder_units[level].conv_type)
            )
            self.usl_blocks.append(
                UpBlock(self.decoder_units[level].features, self.decoder_units[level].upsample_type)
            )
        
        for middle_index in range(self.middle_unit_amount):
            self.middle_conv_blocks.append(
                ConvBlock(self.middle_units[middle_index].block_amount, self.middle_units[middle_index].features, self.middle_units[middle_index].conv_type)
            )

        # self.final_conv = nn.Conv2d(in_channels=features, out_channels=in_channels, kernel_size=3, padding=1)
        self.conv_last = OPS['conv_block_last'](48+in_channels,in_channels)
        self.Sig = nn.Sigmoid()

    
    def forward(self, x):
        # level_1:
        orgin_x = x
        x = self.first_conv(x)
        encoder_conv_outputs = [None for _ in range(self.level_amount)] # 存放encoder的ConvBlock部分的结果，之后在decoder中进行残差连接

        # Encoder
        for level in range(self.level_amount):
            encoder_conv_outputs[level] = self.encoder_conv_blocks[level](x)
            x = self.dsl_blocks[level](encoder_conv_outputs[level])
        
        for middle_index in range(self.middle_unit_amount):
            x = self.middle_conv_blocks[middle_index](x)
        
        for level in range(self.level_amount-1, -1, -1): # 从3到0
            x = self.usl_blocks[level](x)
            # encoder_conv_out = encoder_conv_outputs[len(encoder_conv_outputs)-level-1]
            # x = torch.cat(encoder_conv_out, x, 1)
            x = encoder_conv_outputs[len(encoder_conv_outputs)-level-1]+x
            x = self.decoder_conv_blocks[level](x)
        
        x = self.concat([orgin_x, x])
        out = self.conv_last(x)
        # out = self.final_conv(x)
        out = self.Sig(out)
        return out


# Unet (resnet)
# class Unet(nn.Module):
#     def __init__(self, indi, in_channels = 1, features = 48):
#         super(Unet, self).__init__()
#         self.concat = Concat_layer(dim = 1)
#         self.level_amount = indi.level_amount
#         self.middle_unit_amount = indi.middle_unit_amount

#         self.encoder_units = indi.encoder_units # indi 的encoder_units
#         self.decoder_units = indi.decoder_units 
#         self.middle_units =  indi.middle_units

#         self.encoder_conv_blocks = nn.ModuleList() 
#         self.dsl_blocks = nn.ModuleList()

#         self.decoder_conv_blocks =  nn.ModuleList()
#         self.usl_blocks = nn.ModuleList()

#         self.middle_conv_blocks = nn.ModuleList()

#         self.first_conv =  nn.Conv2d(in_channels=in_channels, out_channels=features, kernel_size=3, padding=1, bias=False )
        
#         for level in range(self.level_amount):
            
#             # encoder
#             self.encoder_conv_blocks.append(
#                 ConvBlock(self.encoder_units[level].block_amount, self.encoder_units[level].features, self.encoder_units[level].conv_type)
#                 )
#             self.dsl_blocks.append(
#                 DownBlock(self.encoder_units[level].features, self.encoder_units[level].downsample_type)
#             )

#             # decoder
#             self.decoder_conv_blocks.append(
#                 ConvBlock(self.encoder_units[level].block_amount, self.decoder_units[level].features, self.decoder_units[level].conv_type)
#             )
#             self.usl_blocks.append(
#                 UpBlock(self.decoder_units[level].features, self.decoder_units[level].upsample_type)
#             )
        
#         for middle_index in range(self.middle_unit_amount):
#             self.middle_conv_blocks.append(
#                 ConvBlock(self.middle_units[middle_index].block_amount, self.middle_units[middle_index].features, self.middle_units[middle_index].conv_type)
#             )

#         # self.final_conv = nn.Conv2d(in_channels=features, out_channels=in_channels, kernel_size=3, padding=1)
#         self.conv_last = OPS['conv_block_last'](48+in_channels,in_channels)
#         self.Sig = nn.Sigmoid()

    
#     def forward(self, x):
#         # level_1:
#         orgin_x = x
#         x = self.first_conv(x)
#         encoder_conv_outputs = [None for _ in range(self.level_amount)] # 存放encoder的ConvBlock部分的结果，之后在decoder中进行残差连接

#         # Encoder
#         for level in range(self.level_amount):
#             encoder_conv_outputs[level] = self.encoder_conv_blocks[level](x)
#             x = self.dsl_blocks[level](encoder_conv_outputs[level])
        
#         for middle_index in range(self.middle_unit_amount):
#             x = self.middle_conv_blocks[middle_index](x)
        
#         for level in range(self.level_amount):
#             x = self.usl_blocks[level](x)
#             x = encoder_conv_outputs[len(encoder_conv_outputs)-level-1]+x
#             x = self.decoder_conv_blocks[level](x)
        
#         x = self.concat([orgin_x, x])
#         out = self.conv_last(x)
#         # out = self.final_conv(x)
#         out = self.Sig(out)
#         return out


if __name__=='__main__':
    from setting.config import  Config
    from genetic.individual import Individual
    params = Config.train_params
    indi = Individual(Config.population_params,0)
    dict = {
        "level_amount":4,
        "middle_unit_amount":1,
        "encoder_units":[ 
            {'encoder_unit_id': 0, 'block_amount': 2, 'features': 48, 'conv_type': 'conv_3x3_leakyReLU', 'downsample_type': 'maxpool2d'},
            {'encoder_unit_id': 1, 'block_amount': 2, 'features': 48, 'conv_type': 'conv_3x3_leakyReLU', 'downsample_type': 'maxpool2d'},
            {'encoder_unit_id': 2, 'block_amount': 2, 'features': 48, 'conv_type': 'conv_3x3_leakyReLU', 'downsample_type': 'maxpool2d'}, 
            {'encoder_unit_id': 3, 'block_amount': 2, 'features': 48, 'conv_type': 'conv_3x3_leakyReLU', 'downsample_type': 'maxpool2d'},],

        "middle_units": [{'middle_unit_id': 0, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_3x3'}],

        "decoder_units": [
            {'decoder_unit_id': 0, 'block_amount': 2, 'features': 48, 'conv_type': 'conv_3x3_leakyReLU', 'upsample_type': 'deconv'},
            {'decoder_unit_id': 1, 'block_amount': 2, 'features': 48, 'conv_type': 'conv_3x3_leakyReLU', 'upsample_type': 'deconv'},
            {'decoder_unit_id': 2, 'block_amount': 2, 'features': 48, 'conv_type': 'conv_3x3_leakyReLU', 'upsample_type': 'deconv'},
            {'decoder_unit_id': 3, 'block_amount': 2, 'features': 48, 'conv_type': 'conv_3x3_leakyReLU', 'upsample_type': 'deconv'},]
    }
    indi.initialize_by_designed(dict)

    x = torch.randn(1,3,512,512)
    net = Unet(indi, 3, 48)
    x = net(x)
    print(x)

    # net_train(indi,params)
    
    