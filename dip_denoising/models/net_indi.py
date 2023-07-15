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

# from .common import *


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
    def __init__(self, block_amount, in_channels, out_channels, conv_type):
        super(ConvBlock, self).__init__()
        layers = []
        for i in range(block_amount):
            layers.append(OPS[conv_type](in_channels, out_channels))
            in_channels, out_channels = out_channels, out_channels
        self.conv_block = nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv_block(x)
        return out




class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample_type):
        super(DownBlock, self).__init__()
        self.down_block = OPS[downsample_type](in_channels, out_channels)
        
    def forward(self, x):
        out = self.down_block(x)
        return out
    
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upsample_type):
        super(UpBlock, self).__init__()
        
        self.up_block = OPS[upsample_type](in_channels, out_channels)
        self.bn = OPS["bn"](out_channels)
        # self.up = nn.Sequential(self.up_block,self.bn)
        # print("up"+'-'*50,self.up)
        

    
    def forward(self, x):

        out  = self.up_block(x)
        out = self.bn(out)
        # out = self.up(x)
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





class s2s_test(nn.Module):
    def __init__(self, indi, img_ch=3, output_ch=3, act_type="LeakyReLU", pad='reflection' ):
        super(s2s_test, self).__init__()
        enc_ch = [48, 48, 48, 48, 48]
        dec_ch = [96, 96, 96, 96, 96]  # fixed
        self.upsample = nn.Upsample(scale_factor=2)
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.concat = Concat_layer(1)
        # self.first_conv = nn.Conv2d(in_channels=img_ch, out_channels=enc_ch[0],kernel_size=3,stride=1,padding=1)
        # self.dsl1  = OPS['convolution_d'](img_ch, 48)
    
        self.Conv1 = nn.Sequential(
            OPS["maxpool2d"](img_ch, 48),

            # OPS['conv_3x3_leakyReLU'](img_ch, 48),
            # OPS['convolution_d'](48, 48),
            OPS['conv_3x3_leakyReLU'](48, 48)
        ) 

        self.Conv2 = nn.Sequential(
            # OPS['convolution_d'](48, 48),
            OPS['maxpool2d'](48,48),
            OPS['conv_3x3_leakyReLU'](48, 48)
        ) 

        self.Conv3 = nn.Sequential(
            # OPS['convolution_d'](48, 48),
            OPS['maxpool2d'](48,48),
            OPS['conv_3x3_leakyReLU'](48, 48)
        ) 

        self.Conv4 = nn.Sequential(
            # OPS['convolution_d'](48, 48),
            OPS['maxpool2d'](48,48),
            OPS['conv_3x3_leakyReLU'](48, 48)
        ) 

        self.Conv5 = nn.Sequential(
            OPS['maxpool2d'](48,48),
            # OPS['convolution_d'](48, 48),
            OPS['conv_3x3_leakyReLU'](48, 48)
        ) 




        self.Up_conv5 = nn.Sequential(
            OPS['conv_3x3_leakyReLU'](48, 96),
            # OPS['conv_3x3_leakyReLU'](96, 96)

        )

        self.Up_conv4 = nn.Sequential(
            OPS['bn'](96+48),
            OPS['conv_3x3_leakyReLU'](96+48, 96),
            OPS['conv_3x3_leakyReLU'](96, 96)

        )

        self.Up_conv3 = nn.Sequential(
            OPS['bn'](96+48),
            OPS['conv_3x3_leakyReLU'](96+48, 96),
            OPS['conv_3x3_leakyReLU'](96, 96)

        )

        self.Up_conv2 = nn.Sequential(
            OPS['bn'](96+48),
            OPS['conv_3x3_leakyReLU'](96+48, 96),
            OPS['conv_3x3_leakyReLU'](96, 96)

        )

        self.Up_conv1 = nn.Sequential(
            OPS['bn'](96+48),
            OPS['conv_3x3_leakyReLU'](96+48, 96),
            OPS['conv_3x3_leakyReLU'](96, 96)

        )

        self.conv_last = OPS['conv_block_last'](96+img_ch, output_ch)
        self.sig = nn.Sigmoid()



    def forward(self, x):
        # encoding path
        # x0 = self.first_conv(x)
        # print(x0.shape)
        x1 = self.Conv1(x)  # h/2 w/2
        x2 = self.Conv2(x1)  # h/4 w/4
        x3 = self.Conv3(x2)  # h/8 w/8
        x4 = self.Conv4(x3)  # h/16 w/16
        x5 = self.Conv5(x4)  # h/32 w/32

        d5 = self.Up_conv5(x5)
        d5 = self.upsample(d5)
        
        d5= self.concat([d5, x4])
        d4 = self.Up_conv4(d5)
        d4 = self.upsample(d4)

        d4= self.concat([d4, x3])
        d3 = self.Up_conv3(d4)
        d3 = self.upsample(d3)

        d3= self.concat([d3, x2])
        d2 = self.Up_conv2(d3)
        d2 = self.upsample(d2)

        d2= self.concat([d2, x1])
        d1 = self.Up_conv1(d2)
        d1 = self.upsample(d1)

        d1 = self.concat([d1, x])
        d0 = self.conv_last(d1)
        return self.sig(d0)


class s2s_test2(nn.Module):
    def __init__(self, indi, img_ch=3, output_ch=3, act_type="LeakyReLU", pad='reflection' ):
        super(s2s_test2, self).__init__()
        enc_ch = [48, 48, 48, 48, 48]
        dec_ch = [96, 96, 96, 96, 96]  # fixed
        self.upsample = nn.Upsample(scale_factor=2)
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.concat = Concat_layer(1)
        # self.first_conv = nn.Conv2d(in_channels=img_ch, out_channels=enc_ch[0],kernel_size=3,stride=1,padding=1)
        # self.dsl1  = OPS['convolution_d'](img_ch, 48)
        # self.first_conv = nn.Conv2d(img_ch, 48 ,kernel_size= 1)
        # self.first_conv = OPS['conv_3x3_leakyReLU'](img_ch, 48)
        
        self.Conv1 = nn.Sequential(
            OPS['conv_3x3_leakyReLU'](img_ch, 48),
            OPS['convolution_d'](img_ch, 48),
            # OPS['conv_3x3_leakyReLU'](48, 48)
        ) 

        self.Conv2 = nn.Sequential(
            OPS['conv_3x3_leakyReLU'](48, 48),
            OPS['convolution_d'](48, 48),
            # OPS['conv_3x3_leakyReLU'](48, 48)
        ) 

        self.Conv3 = nn.Sequential(
            OPS['conv_3x3_leakyReLU'](48, 48),
            OPS['convolution_d'](48, 48),
            # OPS['conv_3x3_leakyReLU'](48, 48)
        ) 

        self.Conv4 = nn.Sequential(
            OPS['conv_3x3_leakyReLU'](48, 48),
            OPS['convolution_d'](48, 48),
            # OPS['conv_3x3_leakyReLU'](48, 48)
        ) 

        self.Conv5 = nn.Sequential(
            OPS['conv_3x3_leakyReLU'](48, 48),
            OPS['convolution_d'](48, 48),
            # OPS['conv_3x3_leakyReLU'](48, 48)
        ) 




        self.Up_conv5 = nn.Sequential(
            OPS['conv_3x3_leakyReLU'](48, 96),
            # OPS['conv_3x3_leakyReLU'](96, 96)

        )

        self.Up_conv4 = nn.Sequential(
            OPS['bn'](96+48),
            OPS['conv_3x3_leakyReLU'](96+48, 96),
            OPS['conv_3x3_leakyReLU'](96, 96)

        )

        self.Up_conv3 = nn.Sequential(
            OPS['bn'](96+48),
            OPS['conv_3x3_leakyReLU'](96+48, 96),
            OPS['conv_3x3_leakyReLU'](96, 96)

        )

        self.Up_conv2 = nn.Sequential(
            OPS['bn'](96+48),
            OPS['conv_3x3_leakyReLU'](96+48, 96),
            OPS['conv_3x3_leakyReLU'](96, 96)

        )

        self.Up_conv1 = nn.Sequential(
            OPS['bn'](96+48),
            OPS['conv_3x3_leakyReLU'](96+48, 96),
            OPS['conv_3x3_leakyReLU'](96, 96)

        )

        self.conv_last = OPS['conv_block_last'](96+img_ch, output_ch)
        self.sig = nn.Sigmoid()



    def forward(self, x):
        # encoding path
        # x0 = self.first_conv(x)
        x1 = self.Conv1(x)  # h/2 w/2
        x2 = self.Conv2(x1)  # h/4 w/4
        x3 = self.Conv3(x2)  # h/8 w/8
        x4 = self.Conv4(x3)  # h/16 w/16
        x5 = self.Conv5(x4)  # h/32 w/32

        d5 = self.Up_conv5(x5)
        d5 = self.upsample(d5)
        
        d5= self.concat([d5, x4])
        d4 = self.Up_conv4(d5)
        d4 = self.upsample(d4)

        d4= self.concat([d4, x3])
        d3 = self.Up_conv3(d4)
        d3 = self.upsample(d3)

        d3= self.concat([d3, x2])
        d2 = self.Up_conv2(d3)
        d2 = self.upsample(d2)

        d2= self.concat([d2, x1])
        d1 = self.Up_conv1(d2)
        d1 = self.upsample(d1)

        d1 = self.concat([d1, x])
        d0 = self.conv_last(d1)
        return self.sig(d0)

    




class Unet(nn.Module):
    def __init__(self, indi, in_channels=3, features=48):
        super(Unet, self).__init__()
        self.concat = Concat_layer(dim = 1)
        self.level_amount = indi.level_amount
        self.encoder_units = indi.encoder_units # indi 的encoder_units
        self.decoder_units = indi.decoder_units 

        self.encoder_conv_blocks = nn.ModuleList([None for _ in range(self.level_amount)]) 
        self.dsl_blocks = nn.ModuleList([None for _ in range(self.level_amount)])

        self.decoder_conv_blocks =  nn.ModuleList([None for _ in range(self.level_amount)])
        self.usl_blocks = nn.ModuleList([None for _ in range(self.level_amount)])
        # self.concat_bn_blocks = nn.ModuleList([None for _ in range(self.level_amount)])
        

        # self.first_conv = nn.Conv2d(in_channels=in_channels, out_channels= features, kernel_size=3, stride=1, padding=1)


        # encoder
        for level in range(self.level_amount):

            # dsl
            if level == 0:
                dsl_in_features = in_channels # in_channels
            else:
                # print("level",level)
                dsl_in_features = self.encoder_units[level-1].features
            # print("level",level)
            self.dsl_blocks[level] = DownBlock(dsl_in_features, self.encoder_units[level].features , self.encoder_units[level].downsample_type)
            self.encoder_conv_blocks[level] = ConvBlock(self.encoder_units[level].block_amount, self.encoder_units[level].features, self.encoder_units[level].features, self.encoder_units[level].conv_type)
           

        for level in range(self.level_amount-1, -1, -1):

            if level == self.level_amount-1:
                conv_in_features = self.encoder_units[level].features
            else:
                conv_in_features = self.encoder_units[level].features + self.decoder_units[level].features
                # self.concat_bn_blocks[level] = OPS['bn'](conv_in_features)
            

            self.decoder_conv_blocks[level] = ConvBlock(self.decoder_units[level].block_amount, conv_in_features, self.decoder_units[level].features, self.decoder_units[level].conv_type)
            self.usl_blocks[level] = UpBlock(self.decoder_units[level].features, self.decoder_units[level].features, self.decoder_units[level].upsample_type)
         
        
        self.conv_last = OPS['conv_block_last'](self.decoder_units[self.level_amount-1].features + in_channels, in_channels)
        self.sig = nn.Sigmoid()

    def forward(self, x):
      
        origin_x =  x
        # x = self.first_conv(x)
        encoder_conv_outputs = [None for _ in range(self.level_amount)] # 存放encoder的ConvBlock部分的结果，之后在decoder中进行残差连接

        # encoder
        for level in range(self.level_amount):
            x = self.dsl_blocks[level](x)
            x = self.encoder_conv_blocks[level](x)
            encoder_conv_outputs[level] = x
        
        # decoder
        for level in range(self.level_amount-1,-1,-1):
            if level != self.level_amount-1:
                x = self.concat([x, encoder_conv_outputs[level]]) # concat
                # x = self.concat_bn_blocks[level](x) # bn
            x = self.decoder_conv_blocks[level](x)
            x = self.usl_blocks[level](x)
        
        # last_conv
        x = self.concat([x, origin_x])
        x = self.conv_last(x)
        out = self.sig(x)

        return out


# 将二维不定长的list平铺为一维list
def flatten(input_list):
    output_list = []
    while True:
        if input_list == []:
            break
        for index, value in enumerate(input_list):

            if type(value) == list:
                input_list = value + input_list[index + 1:]
                break
            else:
                output_list.append(value)
                input_list.pop(index)
                break
    return output_list

def check_active2(nl_num, connect_gene):
    nl_active = [None for _ in range(nl_num)]
    nl_connect = [] # 节点2，3的连接结构[[1],[0,1]]
    start_index = 0
    end_index = 1 # 用来划分connect_gene
    # 划分2，3节点的连接结构, nl_connect 中只有2，3的连接结构，1号节点无
    for _ in range(nl_num-1):
        nl_connect.append(connect_gene[start_index: start_index + end_index]) 
        start_index += end_index
        end_index += 1
    
    # 根据connect判断前驱节点是否激活使用
    for nl_con in nl_connect:
        for nl_index, nl_c in enumerate(nl_con):
            if nl_c==1:
                nl_active[nl_index] = True
    # 根据connect判断当前节点是否激活
    for nl_index, nl_con in enumerate(nl_connect):
        if sum(nl_con) >= 1:
            nl_active[nl_index+1] = True
    
    # 前驱节点list
    pre_index = []
    for nl_index in range(nl_num):
        if(nl_active[nl_index]):
            if nl_index == 0:
                # 1号节点
                pre_index.append([0]) # 上一层输出作为1号节点的输入
            else:
                # 2,3号节点 ， 无入边，有出边,则节点输入为上一层输出，即0号节点
                if sum(nl_connect[nl_index-1]) == 0:
                    pre_index.append([0])
                # 有入边，获取其前驱节点
                else:
                    tmp_index = [] # 前驱节点
                    for index, nl_con in enumerate(nl_connect[nl_index-1]):
                        if nl_con == 1:
                            tmp_index.append(index + 1)
                    pre_index.append(tmp_index)
        else:
            # 节点没有被连接
            pre_index.append([None])

    # 块最终输出边相连的节点
    out_index = []
    for nl_index in range(nl_num):
        # 将2，3号节点的前驱展开
        pre_index_ = flatten(pre_index[nl_index+1:])
        if nl_active[nl_index] and nl_index not in pre_index_:
            # nl激活，且不是其他节点的前驱，作为block的输出
            out_index.append(nl_index+1)
    # 当connect_gene
    if sum(connect_gene) == 0:
        # [0,0,0]
        nl_active = [True,None,None]
        pre_index = [[0],[None],[None]]
        out_index = [1]
    print("connect_arcitecture", nl_active, pre_index,out_index)
    return nl_active, pre_index, out_index


check_active2(3, [0,1,1])  


# unit 中节点的连接结构
def check_active(nl_num, connect_gene):
    """
        connect_gene:[0,0,0],[0,1,1]  000,001,010,011
    """
    # nl_num: block nl数
    nl_active=[None for _ in range(nl_num)]
    nl_connect = [] # [[0],[1,1]] # 连接
    i = 0  #
    j = 1
    # 将connect_gene 划分, such as [0,1,1]-->[[0],[1,1]]
    for _ in range(nl_num-1):
        nl_connect.append(connect_gene[i:i+j])
        i=i+j
        j+=1
    
    # 确定2，3号卷积节点是否使用，节点编号1，2，3 在nl_active的索引为0，1，2
    for p,nl in enumerate(nl_connect):
        # p:0,1 --> 对应的是2，3 号节点，对应nl_active的索引为1，2
        if sum(nl)>=1:
            nl_active[p+1]=True
    
    # 通过遍历前驱方法，来判断此前驱节点是否使用
    for k in range(nl_num-1):
        for nl in nl_connect:
            if k < len(nl) and nl[k]==1:
                # nl[k]在连接结构中的的第k个节点
                nl_active[k] = True

    # 最后一个节点连接
    if sum(connect_gene) >= 1:
        nl_active[nl_num-1] = True

    # 有边节点的前驱节点list
    pre_index = []
    for m in range(nl_num):
        if nl_active[m]:
            if m==0:
                pre_index.append([0])
            else:
                # 节点m有出节点，无入节点，则m的入节点为整个块的输入
                if sum(nl_connect[m - 1])==0:
                    pre_index.append([0])
                # 有入节点
                else:
                    tmp_index = []
                    for index,con in enumerate(nl_connect[m - 1]):
                        if con == 1:
                            tmp_index.append(index+1)
                    pre_index.append(tmp_index)
                    
        # 当这个节点没有被连接，pre_index 处补[]
        else:
            pre_index.append([])

    # 块最终输出边相连的节点list
    out_index = []
    for t in range(nl_num):
        pre_index_ = flatten(pre_index[t+1:])
        # t+1: 表示有1-3编号节点的真实索引
        if nl_active[t] and t+1 not in pre_index_:
            out_index.append(t + 1)
    # 当块中所有节点都没有边，则3个节点并行与输入节点相连，输出为3个节点值
    # 设定第一个
    if sum([1 for act in nl_active if act is not None]) == 0:
        nl_active = [True for _ in range(nl_num)]
        pre_index = [[0] for _ in range(nl_num)]
        out_index = [1,2,3]
        # out_index = [0]
    # print("nl_active,pre_index,out_index",nl_active,pre_index,out_index)
    return nl_active, pre_index, out_index


class UnetConnect(nn.Module):
    def __init__(self, indi, in_channels=3, features=48):
        super(UnetConnect, self).__init__()
        self.concat = Concat_layer(dim = 1)
        self.level_amount = indi.level_amount
        self.encoder_units = indi.encoder_units # indi 的encoder_units
        self.decoder_units = indi.decoder_units 

        self.encoder_conv_blocks = nn.ModuleList([None for _ in range(self.level_amount)]) 
        self.dsl_blocks = nn.ModuleList([None for _ in range(self.level_amount)])

        self.decoder_conv_blocks =  nn.ModuleList([None for _ in range(self.level_amount)])
        self.usl_blocks = nn.ModuleList([None for _ in range(self.level_amount)])
        # self.concat_bn_blocks = nn.ModuleList([None for _ in range(self.level_amount)])
        

        # self.first_conv = nn.Conv2d(in_channels=in_channels, out_channels= features, kernel_size=3, stride=1, padding=1)

        # encoder
        for level in range(self.level_amount):

            # dsl
            if level == 0:
                dsl_in_features = in_channels # in_channels
            else:
                # print("level",level)
                dsl_in_features = self.encoder_units[level-1].features
            # print("level",level)
            self.dsl_blocks[level] = DownBlock(dsl_in_features, self.encoder_units[level].features , self.encoder_units[level].downsample_type)
            self.encoder_conv_blocks[level] = ConvBlock(self.encoder_units[level].block_amount, self.encoder_units[level].features, self.encoder_units[level].features, self.encoder_units[level].conv_type)
           

        for level in range(self.level_amount-1, -1, -1):

            if level == self.level_amount-1:
                conv_in_features = self.encoder_units[level].features
            else:
                conv_in_features = self.encoder_units[level].features + self.decoder_units[level].features
                # self.concat_bn_blocks[level] = OPS['bn'](conv_in_features)
            

            self.decoder_conv_blocks[level] = ConvBlock(self.decoder_units[level].block_amount, conv_in_features, self.decoder_units[level].features, self.decoder_units[level].conv_type)
            self.usl_blocks[level] = UpBlock(self.decoder_units[level].features, self.decoder_units[level].features, self.decoder_units[level].upsample_type)
         
        
        self.conv_last = OPS['conv_block_last'](self.decoder_units[self.level_amount-1].features + in_channels, in_channels)
        self.sig = nn.Sigmoid()

    def forward(self, x):
      
        origin_x =  x
        # x = self.first_conv(x)
        encoder_conv_outputs = [None for _ in range(self.level_amount)] # 存放encoder的ConvBlock部分的结果，之后在decoder中进行残差连接

        # encoder
        for level in range(self.level_amount):
            x = self.dsl_blocks[level](x)
            x = self.encoder_conv_blocks[level](x)
            encoder_conv_outputs[level] = x
        
        # decoder
        for level in range(self.level_amount-1,-1,-1):
            if level != self.level_amount-1:
                x = self.concat([x, encoder_conv_outputs[level]]) # concat
                # x = self.concat_bn_blocks[level](x) # bn
            x = self.decoder_conv_blocks[level](x)
            x = self.usl_blocks[level](x)
        
        # last_conv
        x = self.concat([x, origin_x])
        x = self.conv_last(x)
        out = self.sig(x)

        return out
        
        



            
            

            

            

            




        

# # Unet (concat)
# class Unet(nn.Module):
#     def __init__(self, indi, in_channels = 3, features = 48):
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

#         # self.first_conv =  nn.Conv2d(in_channels=in_channels, out_channels=features, kernel_size=3, padding=1, bias=False)
#         self.first_conv = nn.Sequential(
#             conv(in_channels, features,kernel_size=3, stride=2), bn(features), act("LeakyReLU"),
#             conv(features, features,kernel_size=3, stride=1), bn(features), act("LeakyReLU")
#         )
        
        
#         for level in range(self.level_amount):
            
#             # encoder
#             if level == 0:
#                 dsl_in_features = features # in_channels
#             else:
#                 dsl_in_features = self.encoder_units[level-1].features
            
#             self.dsl_blocks.append(
#                 DownBlock(dsl_in_features, self.encoder_units[level].features , self.encoder_units[level].downsample_type)
#             )
            
#             self.encoder_conv_blocks.append(
#                 ConvBlock(self.encoder_units[level].block_amount, self.encoder_units[level].features, self.encoder_units[level].features, self.encoder_units[level].conv_type)
#                 )
            
            
#             # decoder

#             # if(level != self.level_amount-1):
#             #     usl_in_features = self.decoder_units[level+1].features
#             # else:
#             #     usl_in_features = self.middle_units[self.middle_unit_amount-1].features
#             usl_in_features = self.decoder_units[level].features
#             self.usl_blocks.append(
#                 UpBlock(usl_in_features, self.decoder_units[level].features, self.decoder_units[level].upsample_type)
#             )

#             self.decoder_conv_blocks.append(
#                 ConvBlock(self.encoder_units[level].block_amount, self.decoder_units[level].features + self.encoder_units[level].features,self.decoder_units[level].features, self.decoder_units[level].conv_type)
#             )
            
        
#         # middle
#         for middle_index in range(self.middle_unit_amount):
#             self.middle_conv_blocks.append(
#                 ConvBlock(self.middle_units[middle_index].block_amount, self.middle_units[middle_index].features, self.middle_units[middle_index].features*2, self.middle_units[middle_index].conv_type)
#             )

#         # self.final_conv = nn.Conv2d(in_channels=features, out_channels=in_channels, kernel_size=3, padding=1)
#         self.upsample = nn.Upsample(scale_factor=2)
#         self.conv_last = OPS['conv_block_last'](96+in_channels,in_channels)
#         self.Sig = nn.Sigmoid()

    
#     def forward(self, x):
#         # level_1:
#         orgin_x = x
#         x = self.first_conv(x)
#         encoder_conv_outputs = [None for _ in range(self.level_amount)] # 存放encoder的ConvBlock部分的结果，之后在decoder中进行残差连接

#         # Encoder
#         for level in range(self.level_amount):
#             x = self.dsl_blocks[level](x)
#             x = self.encoder_conv_blocks[level](x)
#             encoder_conv_outputs[level] = x
        
#             # x = self.dsl_blocks[level](encoder_conv_outputs[level])
#             # encoder_conv_outputs[level] = self.encoder_conv_blocks[level](x)
#             # x = self.dsl_blocks[level](encoder_conv_outputs[level])
        
#         for middle_index in range(self.middle_unit_amount):
#             x = self.middle_conv_blocks[middle_index](x)
        
#         for level in range(self.level_amount-1, -1, -1): # 从3到0
#             x = self.usl_blocks[level](x)
#             encoder_conv_output = encoder_conv_outputs[level]
#             # x = torch.cat(encoder_conv_out, x, 1)
#             x = self.concat([encoder_conv_output, x])
#             # x = torch.cat([encoder_conv_out, x], 1)
#             # x = encoder_conv_outputs[len(encoder_conv_outputs)-level-1]+x
#             x = self.decoder_conv_blocks[level](x)
#         x = self.upsample(x)
#         x = self.concat([orgin_x, x])
#         out = self.conv_last(x)
#         # out = self.final_conv(x)
#         out = self.Sig(out)
#         return out


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
    
    