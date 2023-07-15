from evaluate.models.operations import OPS

import torch.nn as nn
import torch 


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
# class 

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
    
    def forward(self, x):
        out  = self.up_block(x)
        return out 

class Unet(nn.Module):
    def __init__(self, indi, in_channels = 1, features = 64):
        super(Unet, self).__init__()
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

        self.final_conv = nn.Conv2d(in_channels=features, out_channels=in_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        # level_1:

        x = self.first_conv(x)
        encoder_conv_outputs = [None for _ in range(self.level_amount)] # 存放encoder的ConvBlock部分的结果，之后在decoder中进行残差连接

        # Encoder
        for level in range(self.level_amount):
            encoder_conv_outputs[level] = self.encoder_conv_blocks[level](x)
            x = self.dsl_blocks[level](encoder_conv_outputs[level])
        
        for middle_index in range(self.middle_unit_amount):
            x = self.middle_conv_blocks[middle_index](x)
        
        for level in range(self.level_amount):
            x = self.usl_blocks[level](x)
            x = encoder_conv_outputs[len(encoder_conv_outputs)-level-1]+x
            x = self.decoder_conv_blocks[level](x)
        
        out = self.final_conv(x)
        return out
    

        


        

            


    


         

 
