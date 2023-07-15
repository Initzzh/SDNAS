from collections import namedtuple

Genotype = namedtuple('Geotype','normal normal_concat up up_concat down down_concat')

net_final = Genotype(normal=[('conv_3x3_no_bn', 0), ('resb', 1), ('resb', 2), 
                             ('conv_3x3_no_bn', 3), ('sep_conv_5x5', 4), ('skip_connect', 5),
                             ('conv_5x5_no_bn', 6), ('skip_connect', 7), ('resb', 8),
                             ('conv_5x5_no_bn', 9), ('dil_conv_3x3', 10), ('skip_connect', 11), 
                             ('skip_connect', 12), ('dil_conv_5x5', 13), ('skip_connect', 14), 
                             ('resb', 15), ('skip_connect', 16), ('resb', 17), 
                             ('resb', 18), ('resb', 19),('skip_connect', 20)], 
                     normal_concat=[40, 40, 40, 32, 40, 40,40, 40, 40, 48, 48, 48, 40, 40, 40, 32, 32,32, 40, 40,40], 
                     up=[('bilinear_u', 0), ('bilinear_u', 1), ('bilinear_u', 2)], up_concat=[40, 40, 32], 
                     down=[('convolution_d', 0), ('nearest_d', 1), ('nearest_d', 2)], down_concat=[24, 32, 32])



import torch
import torch.nn as nn

from model.operations import *



nl_search_space = ['conv_3x3_no_bn','conv_5x5_no_bn','sep_conv_3x3','sep_conv_5x5','dil_conv_3x3','dil_conv_5x5','resb','rcab']

dsl_search_space = ['convolution_d','bilinear_d','nearest_d','area_d']

usl_search_space = ['deconvolution_u','bilinear_u','nearset_u','area_u']

NL_PRIMITIVES = [
    'conv_3x3_no_bn',
    'conv_5x5_no_bn',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5',
    'resb',
    'rcab'
]


DSL_PRIMITIVES = [
    'convolution_d',
    'bilinear_d',
    'nearest_d',
    'area_d'
]

USL_PRIMITIVES = [
    'deconvolution_u',
    'bilinear_u',
    'nearset_u',
    'area_u',
    # 'sub_pixel_u'
]

class NLBlock(nn.Module):
    def __init__(self, C_in, C_out, block_arichtect):
        super(NLBlock,self).__init__()
        stride = 1
        self.nl_active = block_arichtect['nl_active']
        self.nl_pre_index = block_arichtect['nl_pre_index']
        self.block_out_index  = block_arichtect['block_out_index']
        self.block_nl_types = block_arichtect['block_nl_types']
        
        self.block=nn.ModuleList()
        for i in range(len(self.block_nl_types)):
            if self.nl_active[i]:
                operation = NL_PRIMITIVES[self.block_nl_types[i]]
                self.block.append(OPS[operation](C_in,C_out,stride,True))
                C_in, C_out = C_out,C_out
            else:
                self.block.append(None)
        self.outputs = [None for _ in range(len(self.nl_active) + 1)]
        
    def forward(self, x):
        outputs = self.outputs
        outputs[0] = x
        # i代表节点名即，0，1，2，3，0： 表示上一个block 的输出值，outputs[0]:上个block的输出值即x,outputs[1]:这个block的第一个节点的输出
        # output[index]: 第index个节点的输出值
        # nl_pre_index：第i个节点的前驱节点，block节点序号即i从1开始，0表示上个block的输出值
        for i in range(1,len(self.nl_active)+1):
            if self.nl_active[i-1]:
                for j, index in enumerate(self.nl_pre_index[i-1]):
                    # nl_pre_index[i-1]:
                    if j == 0:
                        input_t = outputs[index]
                    else:
                        input_t = input_t + outputs[index]
                # 第i个节点在nn.ModuleList()中对应的索引为i-1
                outputs[i] = self.block[i-1](input_t)
        for y, o_index in enumerate(self.block_out_index):
            if y == 0:
                out = outputs[o_index]
            else:
                out = out + outputs[o_index]
        
        return out
    


class Decode2Unet(nn.Module):
    def __init__(self, unet_arichtect,input_channels=1,mid_channels=32):
        stride = 1
        # mid_channels=[128,256,512,1024,1024,512,256]
        super(Decode2Unet, self).__init__()
        # first_conv
        self.unet_arichtect = unet_arichtect

        self.init_conv = nn.Conv2d(in_channels=input_channels,out_channels=mid_channels,kernel_size=1, stride=1, padding=0)
        C_in, C_out = mid_channels, mid_channels  # update channels
 
        # encode_block
        self.encode_nl_block = nn.ModuleList()
        self.dsl_block = nn.ModuleList()
        encoding_block_num = 3
        for i in range(encoding_block_num):
            # encoding_block_i
            # block = nn.ModuleList()
            # block_nl_types= self.unet_arichtect[i]['block_nl_types']
            # nl_active = self.unet_arichtect[i]['nl_active']
            # 3 个 NL  
            # update chanenls
            # encoding_block_i: NL_block
            # NL_Block
            C_in, C_out = C_out, C_out
            self.encode_nl_block.append(NLBlock(C_in, C_out, unet_arichtect[i]))
            

            # DSL
            C_in, C_out = C_out, C_out*2  # update channels ,downsample 后Unet 下一层特征图通道数为C_out*2 
            dsl_type = unet_arichtect[i]['dsl_type']
            operation = DSL_PRIMITIVES[dsl_type]
            self.dsl_block.append(OPS[operation](C_in,C_out,stride,True))
            # C_in, C_out = C_out, C_out * 2  
            
        self.decode_nl_block = nn.ModuleList()
        self.usl_block = nn.ModuleList()

        # decoding_block
        self.cat_block = nn.ModuleList()
 
        for i in range(encoding_block_num,len(unet_arichtect)):
            
            # 第一个decode_block 没有cat层，所以C_in 仍为上层的C_out,
            # 之后的decode_block 有cat层，所以C_in变为上层的C_out
            if i-encoding_block_num == 0:
                C_in, C_out = C_out, C_out
            else:
                # cat模块
                C_in, C_out = C_out*2, C_out
                self.cat_block.append(nn.Conv2d(C_in, C_out,kernel_size=1,stride=1,padding=0))
                C_in, C_out = C_out, C_out
            
            self.decode_nl_block.append(NLBlock(C_in, C_out, unet_arichtect[i]))
            
            # USL
            if(i<len(unet_arichtect)-1):
                usl_type = unet_arichtect[i]['usl_type']
                operation = USL_PRIMITIVES[usl_type]
                C_in, C_out = C_out, int(C_out/2)
                self.usl_block.append(OPS[operation](C_in,C_out,stride, True))
                
                # C_in,C_out = C_out*2, C_out # USL 后 encode_nl_block与当前USL 输出值cat,所以下面的层的C_in变为C_out*2
                # C_in, C_out = C_out, int(C_out/2)
        
        # final_Conv
        C_in, C_out = C_out, input_channels
        self.final_conv = nn.Conv2d(C_in, C_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.init_conv(x)
        
        eb_num = len(self.encode_nl_block)
        db_num = len(self.decode_nl_block)
        encode_nl_outputs=[None for _ in range(eb_num)]
        
        # encode_nl_outputs： 没有进行dsl_block 的结果
        # 进行encode_nl_block和dsl_block
        for i in range(eb_num):
            encode_nl_outputs[i] = self.encode_nl_block[i](x)
            x=self.dsl_block[i](encode_nl_outputs[i])
        
        # decode_nl_block和usl_block
        for i in range(db_num):
            # 第一个decode_block 无cat
            if i == 0:
                x = self.decode_nl_block[i](x)
                x = self.usl_block[i](x)
            # 最后一个decode_block 无upsample
            elif i == db_num-1:
                x = self.cat_block[i-1](torch.cat([encode_nl_outputs[eb_num-i], x], 1))
                x = self.decode_nl_block[i](x)
            else:
                # print(torch.cat([encode_nl_outputs[eb_num-i], x], 1).shape)
                # 先cat,channel：32，32-->64,再转换为32，64-->32
                x = self.cat_block[i-1](torch.cat([encode_nl_outputs[eb_num-i], x], 1))
                x = self.decode_nl_block[i](x)
                x = self.usl_block[i](x)
        x = self.final_conv(x)
        return x
            

            




# class Unet(nn.Module):
#     def __init__(self,input_channels):
#         super(Unet, self).__init__()

#         self.


# class DnCNN(NNRegressor):

#     def __init__(self, D, C=64):
#         super(DnCNN, self).__init__()
#         self.D = D

#         # convolution layers
#         self.conv = nn.ModuleList()
#         self.conv.append(nn.Conv2d(3, C, 3, padding=1))
#         self.conv.extend([nn.Conv2d(C, C, 3, padding=1) for _ in range(D)])
#         self.conv.append(nn.Conv2d(C, 3, 3, padding=1))
#         # apply He's initialization
#         for i in range(len(self.conv[:-1])):
#             nn.init.kaiming_normal_(
#                 self.conv[i].weight.data, nonlinearity='relu')

#         # batch normalization
#         self.bn = nn.ModuleList()
#         self.bn.extend([nn.BatchNorm2d(C, C) for _ in range(D)])
#         # initialize the weights of the Batch normalization layers
#         for i in range(D):
#             nn.init.constant_(self.bn[i].weight.data, 1.25 * np.sqrt(C))

#     def forward(self, x):
#         D = self.D
#         h = F.relu(self.conv[0](x))
#         for i in range(D):
#             h = F.relu(self.bn[i](self.conv[i+1](h)))
#         y = self.conv[D+1](h) + x
#         return y


# class UDnCNN(NNRegressor):

#     def __init__(self, D, C=64):
#         super(UDnCNN, self).__init__()
#         self.D = D

#         # convolution layers
#         self.conv = nn.ModuleList()
#         self.conv.append(nn.Conv2d(3, C, 3, padding=1))
#         self.conv.extend([nn.Conv2d(C, C, 3, padding=1) for _ in range(D)])
#         self.conv.append(nn.Conv2d(C, 3, 3, padding=1))
#         # apply He's initialization
#         for i in range(len(self.conv[:-1])):
#             nn.init.kaiming_normal_(
#                 self.conv[i].weight.data, nonlinearity='relu')

#         # batch normalization
#         self.bn = nn.ModuleList()
#         self.bn.extend([nn.BatchNorm2d(C, C) for _ in range(D)])
#         # initialize the weights of the Batch normalization layers
#         for i in range(D):
#             nn.init.constant_(self.bn[i].weight.data, 1.25 * np.sqrt(C))

#     def forward(self, x):
#         D = self.D
#         h = F.relu(self.conv[0](x))
#         h_buff = []
#         idx_buff = []
#         shape_buff = []
#         for i in range(D//2-1):
#             shape_buff.append(h.shape)
#             h, idx = F.max_pool2d(F.relu(self.bn[i](self.conv[i+1](h))),
#                                   kernel_size=(2, 2), return_indices=True)
#             h_buff.append(h)
#             idx_buff.append(idx)
#         for i in range(D//2-1, D//2+1):
#             h = F.relu(self.bn[i](self.conv[i+1](h)))
#         for i in range(D//2+1, D):
#             j = i - (D // 2 + 1) + 1
#             h = F.max_unpool2d(F.relu(self.bn[i](self.conv[i+1]((h+h_buff[-j])/np.sqrt(2)))),
#                                idx_buff[-j], kernel_size=(2, 2), output_size=shape_buff[-j])
#         y = self.conv[D+1](h) + x
#         return y


# class DUDnCNN(NNRegressor):

#     def __init__(self, D, C=64):
#         super(DUDnCNN, self).__init__()
#         self.D = D

#         # compute k(max_pool) and l(max_unpool)
#         k = [0]
#         k.extend([i for i in range(D//2)])
#         k.extend([k[-1] for _ in range(D//2, D+1)])
#         l = [0 for _ in range(D//2+1)]
#         l.extend([i for i in range(D+1-(D//2+1))])
#         l.append(l[-1])

#         # holes and dilations for convolution layers
#         holes = [2**(kl[0]-kl[1])-1 for kl in zip(k, l)]
#         dilations = [i+1 for i in holes]

#         # convolution layers
#         self.conv = nn.ModuleList()
#         self.conv.append(
#             nn.Conv2d(3, C, 3, padding=dilations[0], dilation=dilations[0]))
#         self.conv.extend([nn.Conv2d(C, C, 3, padding=dilations[i+1],
#                                     dilation=dilations[i+1]) for i in range(D)])
#         self.conv.append(
#             nn.Conv2d(C, 3, 3, padding=dilations[-1], dilation=dilations[-1]))
#         # apply He's initialization
#         for i in range(len(self.conv[:-1])):
#             nn.init.kaiming_normal_(
#                 self.conv[i].weight.data, nonlinearity='relu')

#         # batch normalization
#         self.bn = nn.ModuleList()
#         self.bn.extend([nn.BatchNorm2d(C, C) for _ in range(D)])
#         # initialize the weights of the Batch normalization layers
#         for i in range(D):
#             nn.init.constant_(self.bn[i].weight.data, 1.25 * np.sqrt(C))

#     def forward(self, x):
#         D = self.D
#         h = F.relu(self.conv[0](x))
#         h_buff = []

#         for i in range(D//2 - 1):
#             torch.backends.cudnn.benchmark = True
#             h = self.conv[i+1](h)
#             torch.backends.cudnn.benchmark = False
#             h = F.relu(self.bn[i](h))
#             h_buff.append(h)

#         for i in range(D//2 - 1, D//2 + 1):
#             torch.backends.cudnn.benchmark = True
#             h = self.conv[i+1](h)
#             torch.backends.cudnn.benchmark = False
#             h = F.relu(self.bn[i](h))

#         for i in range(D//2 + 1, D):
#             j = i - (D//2 + 1) + 1
#             torch.backends.cudnn.benchmark = True
#             h = self.conv[i+1]((h + h_buff[-j]) / np.sqrt(2))
#             torch.backends.cudnn.benchmark = False
#             h = F.relu(self.bn[i](h))

#         y = self.conv[D+1](h) + x
#         return y

# # 正常的Block: 3
# class Block(nn.Module):
#     def __init__(self, features):
#         super(Block, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, bias=False)
#         self.batch_norm1 = nn.BatchNorm2d(features)
#         self.relu1 = nn.ReLU(inplace=True)

#         self.conv2 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, bias=False)
#         self.batch_norm2 = nn.BatchNorm2d(features)
#         self.relu2 = nn.ReLU(inplace=True)

#         self.conv3 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, bias=False)
#         self.batch_norm3 = nn.BatchNorm2d(features)
#         self.relu3 = nn.ReLU(inplace=True)
        
#         layers = [self.conv1,self.batch_norm1,self.relu1,self.conv2,self.batch_norm2,self.relu2,self.conv3,self.batch_norm3,self.relu3]
#         self.block = nn.Sequential(*layers)
        
    
#     def forward(self, x):
#         out = self.block(x)
#         return out


class ConvRelu(nn.Module):
    def __init__(self, features):
        super(ConvRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels=features, out_channels= features, kernel_size=3, padding=1, bias=False)
        self.batch_norm = nn.BatchNorm2d(features)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.batch_norm(out)
        out = self.relu(out)
        return out + x
    
class Block(nn.Module):
    def __init__(self, features, conv_amount):
        super(Block, self).__init__()
        layers = []
        for _ in range(conv_amount):
            layers.append(ConvRelu(features))
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.block(x)
        return out


# # block_res
# class Block(nn.Module):
#     def __init__(self, features):
#         super(Block, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, bias=False)
#         self.batch_norm1 = nn.BatchNorm2d(features)
#         self.relu1 = nn.ReLU(inplace=True)

#         self.conv2 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, bias=False)
#         self.batch_norm2 = nn.BatchNorm2d(features)
#         self.relu2 = nn.ReLU(inplace=True)

#         self.conv3 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, bias=False)
#         self.batch_norm3 = nn.BatchNorm2d(features)
#         self.relu3 = nn.ReLU(inplace=True)
        
#         # layers = [self.conv1,self.batch_norm1,self.relu1,self.conv2,self.batch_norm2,self.relu2,self.conv3,self.batch_norm3,self.relu3]
#         # self.block = nn.Sequential(*layers)
        
    
#     def forward(self, x):
#         out1 = self.conv1(x)
#         out1 = self.batch_norm1(out1)
#         out1 = self.relu1(out1)

#         out1 = out1 + x

#         out2 = self.conv2(out1)
#         out2 = self.batch_norm2(out2)
#         out2 = self.relu2(out2)

#         out2 = out2+out1

#         out3 = self.conv3(out2)
#         out3 = self.batch_norm3(out3)
#         out3 = self.relu3(out3)

#         out3 = out3+out2

#         # out = self.block(x)
#         return out3
    
# dilated
# class Block(nn.Module):
#     def __init__(self, features, padding, dilation):
#         super(Block, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=padding, dilation=dilation, bias=False)
#         self.batch_norm1 = nn.BatchNorm2d(features)
#         self.relu1 = nn.ReLU(inplace=True)

#         self.conv2 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=padding, dilation=dilation, bias=False)
#         self.batch_norm2 = nn.BatchNorm2d(features)
#         self.relu2 = nn.ReLU(inplace=True)

#         self.conv3 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=padding, dilation=dilation, bias=False)
#         self.batch_norm3 = nn.BatchNorm2d(features)
#         self.relu3 = nn.ReLU(inplace=True)
        
#         layers = [self.conv1,self.batch_norm1,self.relu1,self.conv2,self.batch_norm2,self.relu2,self.conv3,self.batch_norm3,self.relu3]
#         self.block = nn.Sequential(*layers)
        
    
#     def forward(self, x):
#         out = self.block(x)
#         return out


# class DownBlock(nn.Module):
#     def __init__(self, features):
#          # down
#         super(DownBlock,self).__init__()
#         self.maxpool = nn.MaxPool2d(2)
#         self.conv = nn.Conv2d(in_channels=features, out_channels=2*features, kernel_size=1, stride=1, padding=0)
#         # self.batch_norm = nn.BatchNorm2d(features*2)
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self,x):
#         out = self.maxpool(x)
#         out = self.conv(out)
#         # out = self.batch_norm(out)
#         out = self.relu(out)
#         return out

class DownBlock(nn.Module):
    def __init__(self, features):
         # down
        super(DownBlock,self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        # self.conv = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=1, stride=1, padding=0)
        # self.batch_norm = nn.BatchNorm2d(features*2)
        # self.relu = nn.ReLU(inplace=True)

    def forward(self,x):
        out = self.maxpool(x)
        # out = self.conv(out)
        # out = self.batch_norm(out)
        # out = self.relu(out)
        return out
    


# class UpBlock(nn.Module):
#     def __init__(self, features):

#         super(UpBlock, self).__init__()
#         self.conv = nn.Conv2d(in_channels=features,out_channels=features*2, kernel_size=1, stride=1, padding=0)
#         # self.batch_norm = nn.BatchNorm2d(features/2)
#         self.relu = nn.ReLU(inplace=True)
#         self.subpixel = nn.PixelShuffle(2)

#     def forward(self, x):
#         out = self.conv(x)
#         # out = self.batch_norm(out)
#         out = self.relu(out)
#         out = self.subpixel(out)
#         return out
    

class UpBlock(nn.Module):
    def __init__(self, features):

        super(UpBlock, self).__init__()
        # self.conv = nn.Conv2d(in_channels=features,out_channels=features*4, kernel_size=1, stride=1, padding=0)
        # self.batch_norm = nn.BatchNorm2d(features/2)
        # self.relu = nn.ReLU(inplace=True)
        # self.subpixel = nn.PixelShuffle(2)
        self.upsample = nn.ConvTranspose2d(in_channels=features,out_channels=features,kernel_size=3,stride=2,padding=1,output_padding=1)

    def forward(self, x):
        # out = self.conv(x)
        # # out = self.batch_norm(out)
        # out = self.relu(out)
        # out = self.subpixel(out)
        # return out
        out = self.upsample(x)
        return out


class DnCNN(nn.Module):
    def __init__(self, channels, num_of_layers=17):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        # channels=1
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers-2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
    def forward(self, x):
        out = self.dncnn(x)
        return out+x


class DnUnet(nn.Module):
    def __init__(self):
        super(DnUnet, self).__init__()
        in_chanenls = 1
        kernel_size = 3
        padding = 1
        features = 64
        
        self.first_conv = nn.Conv2d(in_channels=1, out_channels=features, kernel_size=3,padding=1, bias=False)
        self.first_relu = nn.ReLU(inplace=True)

        self.block11 = Block(features)
        # # # down
        self.down1 = DownBlock(features)

        features = 128
        self.block21 = Block(features)
        self.down2 = DownBlock(features)

        features = 256
        self.middle_block = Block(features)


        
        self.up2 = UpBlock(features)
        features = 128
        self.cat_after_conv2 = nn.Conv2d(in_channels=features*2, out_channels=features, kernel_size=1, padding=0)
        self.cat_after_relu2 = nn.ReLU(inplace=True)

        self.blcok22 =  Block(features)

        self.up1 = UpBlock(features)
        features = 64

        self.cat_after_conv1 = nn.Conv2d(in_channels=features*2, out_channels=features, kernel_size=1,padding=0)
        self.cat_after_relu1 = nn.ReLU(inplace=True)

        self.block12 = Block(features)

        self.final_conv = nn.Conv2d(in_channels=features,out_channels=1,kernel_size=3,padding=1)
    
    def forward(self, x):



        out = self.first_conv(x)
        out = self.first_relu(out)
        
        out1 = self.block11(out)
        out_down1 = self.down1(out1)

        out2 = self.block21(out_down1)
        out_down2 = self.down2(out2)

        # out3 = self.block31(out_down2)
        # out_down3 = self.down3(out3)

        middle_out = self.middle_block(out_down2)

    
        out_up2 = self.up2(middle_out)

        out_cat2 = torch.cat((out2,out_up2),1)
        out_cat_after2 = self.cat_after_conv2(out_cat2)
        out_cat_after2 = self.cat_after_relu2(out_cat_after2)

        out22 = self.blcok22(out_cat_after2)


        out_up1 = self.up1(out22)

        out_cat1 = torch.cat((out1,out_up1),1)
        out_cat_after1 = self.cat_after_conv1(out_cat1)
        out_cat_after1 = self.cat_after_relu1(out_cat_after1)

        out12 =  self.block12(out_cat_after1)

        final_out = self.final_conv(out12)

        return final_out

# connect:resnet
class DnUnet3(nn.Module):
    def __init__(self):
        super(DnUnet3, self).__init__()
        in_chanenls = 1
        kernel_size = 3
        padding = 1
        features = 64
        
        self.first_conv = nn.Conv2d(in_channels=1, out_channels=features, kernel_size=3,padding=1, bias=False)
        self.first_relu = nn.ReLU(inplace=True)

        self.block11 = Block(features)
        # # # down
        self.down1 = DownBlock(features)

        # features = 64
        self.block21 = Block(features)
        self.down2 = DownBlock(features)

        # features = 64
        self.middle_block = Block(features)


        self.up2 = UpBlock(features)
        # features = 64
        # self.cat_after_conv2 = nn.Conv2d(in_channels=features*2, out_channels=features, kernel_size=1, padding=0)
        # self.cat_after_relu2 = nn.ReLU(inplace=True)

        self.block22 =  Block(features)

        self.up1 = UpBlock(features)
        # features = 64

        # self.cat_after_conv1 = nn.Conv2d(in_channels=features*2, out_channels=features, kernel_size=1,padding=0)
        # self.cat_after_relu1 = nn.ReLU(inplace=True)

        self.block12 = Block(features)

        self.final_conv = nn.Conv2d(in_channels=features,out_channels=1,kernel_size=3,padding=1)
    
    def forward(self, x):

        out = self.first_conv(x)
        out = self.first_relu(out)
        
        out1 = self.block11(out)
        out_down1 = self.down1(out1)

        out2 = self.block21(out_down1)
        out_down2 = self.down2(out2)

        # out3 = self.block31(out_down2)
        # out_down3 = self.down3(out3)

        middle_out = self.middle_block(out_down2)

    
        out_up2 = self.up2(middle_out)
        # print(out_up2.shape,out2.shape)
        out_up2 = out_up2+out2

        # out_cat2 = torch.cat((out2,out_up2),1)
        # out_cat_after2 = self.cat_after_conv2(out_cat2)
        # out_cat_after2 = self.cat_after_relu2(out_cat_after2)

        # out_cat_after2 = nn.functional.dropout(out_cat_after2, 0.3)
        out22 = self.block22(out_up2)
        # out22 = self.blcok22(out_cat_after2)


        out_up1 = self.up1(out22)

        out_up1 =  out_up1+out1

        # out_cat1 = torch.cat((out1,out_up1),1)
        # out_cat_after1 = self.cat_after_conv1(out_cat1)
        # out_cat_after1 = self.cat_after_relu1(out_cat_after1)

        # out_cat_after1 = nn.functional.dropout(out_cat_after1, 0.3)
        out12 = self.block12(out_up1)
        # out12 =  self.block12(out_cat_after1)

        final_out = self.final_conv(out12)

        return final_out
    


# connect:resnet :level:3, block_amount:2
class DnUnet31(nn.Module):
    def __init__(self):
        super(DnUnet31, self).__init__()
        in_chanenls = 1
        kernel_size = 3
        padding = 1
        features = 64
        
        self.first_conv = nn.Conv2d(in_channels=1, out_channels=features, kernel_size=3,padding=1, bias=False)
        self.first_relu = nn.ReLU(inplace=True)

        self.block11 = Block(features,conv_amount=2)
        # # # down
        self.down1 = DownBlock(features)

        # features = 64
        self.block21 = Block(features,conv_amount=2)
        self.down2 = DownBlock(features)

        self.block31 = Block(features, conv_amount=2)
        self.down3 = DownBlock(features)

        # features = 64
        self.middle_block = Block(features,conv_amount=3)


        self.up3 = UpBlock(features)
        self.block32 = Block(features, conv_amount=2)

        self.up2 = UpBlock(features)
        # features = 64
        # self.cat_after_conv2 = nn.Conv2d(in_channels=features*2, out_channels=features, kernel_size=1, padding=0)
        # self.cat_after_relu2 = nn.ReLU(inplace=True)

        self.block22 =  Block(features, conv_amount=2)

        self.up1 = UpBlock(features)
        # features = 64

        # self.cat_after_conv1 = nn.Conv2d(in_channels=features*2, out_channels=features, kernel_size=1,padding=0)
        # self.cat_after_relu1 = nn.ReLU(inplace=True)

        self.block12 = Block(features, conv_amount=2)

        self.final_conv = nn.Conv2d(in_channels=features,out_channels=1,kernel_size=3,padding=1)
    
    def forward(self, x):

        out = self.first_conv(x)
        out = self.first_relu(out)
        
        out1 = self.block11(out)
        out_down1 = self.down1(out1)

        out2 = self.block21(out_down1)
        out_down2 = self.down2(out2)

        out3 = self.block31(out_down2)
        out_down3 = self.down3(out3)

        middle_out = self.middle_block(out_down3)

        
        out_up3  = self.up3(middle_out)
        out_up3 = out_up3 + out3

        out32 = self.block32(out_up3)

        out_up2 = self.up2(out32)

        # print(out_up2.shape,out2.shape)
        out_up2 = out_up2+out2

        # out_cat2 = torch.cat((out2,out_up2),1)
        # out_cat_after2 = self.cat_after_conv2(out_cat2)
        # out_cat_after2 = self.cat_after_relu2(out_cat_after2)

        # out_cat_after2 = nn.functional.dropout(out_cat_after2, 0.3)
        out22 = self.block22(out_up2)
        # out22 = self.blcok22(out_cat_after2)


        out_up1 = self.up1(out22)

        out_up1 =  out_up1+out1

        # out_cat1 = torch.cat((out1,out_up1),1)
        # out_cat_after1 = self.cat_after_conv1(out_cat1)
        # out_cat_after1 = self.cat_after_relu1(out_cat_after1)

        # out_cat_after1 = nn.functional.dropout(out_cat_after1, 0.3)
        out12 = self.block12(out_up1)
        # out12 =  self.block12(out_cat_after1)

        final_out = self.final_conv(out12)

        return final_out
    

class DnUnet4(nn.Module):
    def __init__(self):
        super(DnUnet4, self).__init__()
        in_chanenls = 1
        kernel_size = 3
        padding = 1
        features = 64
        
        self.first_conv = nn.Conv2d(in_channels=1, out_channels=features, kernel_size=3,padding=1, bias=False)
        self.first_relu = nn.ReLU(inplace=True)

        self.block11 = Block(features,padding = 1, dilation = 1)
        # # # down
        # self.down1 = DownBlock(features)

        # features = 64
        self.block21 = Block(features, padding=2, dilation=2)
        # self.down2 = DownBlock(features)

        # features = 64
        self.middle_block = Block(features, padding=4, dilation=4)


        # self.up2 = UpBlock(features)
        # features = 64
        # self.cat_after_conv2 = nn.Conv2d(in_channels=features*2, out_channels=features, kernel_size=1, padding=0)
        # self.cat_after_relu2 = nn.ReLU(inplace=True)

        self.block22 =  Block(features, padding=2, dilation=2)

        # self.up1 = UpBlock(features)
        # features = 64

        # self.cat_after_conv1 = nn.Conv2d(in_channels=features*2, out_channels=features, kernel_size=1,padding=0)
        # self.cat_after_relu1 = nn.ReLU(inplace=True)

        self.block12 = Block(features,padding=1,dilation=1)

        self.final_conv = nn.Conv2d(in_channels=features,out_channels=1,kernel_size=3,padding=1)
    
    def forward(self, x):

        out = self.first_conv(x)
        out = self.first_relu(out)
        
        out1 = self.block11(out)
        # out_down1 = self.down1(out1)

        out2 = self.block21(out1)
        # out_down2 = self.down2(out2)

        middle_out = self.middle_block(out2)
    
        # out_up2 = self.up2(middle_out)

        middle_out = middle_out+out2

        out22 = self.block22(middle_out)

        # out_up1 = self.up1(out22)

        out22 =  out22+out1

        out12 = self.block12(out22)

        final_out = self.final_conv(out12)

        return final_out
    




class DnUnet2(nn.Module):
    def __init__(self):
        super(DnUnet, self).__init__()
        in_chanenls = 1
        kernel_size = 3
        padding = 1
        features = 64
        
        self.first_conv = nn.Conv2d(in_channels=1, out_channels=features, kernel_size=3,padding=1, bias=False)
        self.first_relu = nn.ReLU(inplace=True)

        self.block11 = Block(features)
        # # # down
        # self.down1 = DownBlock(features)

        # features = 128
        self.block21 = Block(features)
        # self.down2 = DownBlock(features)

        # features = 256
        self.middle_block = Block(features)


        
        # self.up2 = UpBlock(features)
        # features = 128
        self.cat_after_conv2 = nn.Conv2d(in_channels=features*2, out_channels=features, kernel_size=1, padding=0)
        self.cat_after_relu2 = nn.ReLU(inplace=True)

        self.blcok22 =  Block(features)

        self.up1 = UpBlock(features)
        # features = 64

        self.cat_after_conv1 = nn.Conv2d(in_channels=features*2, out_channels=features, kernel_size=1,padding=0)
        self.cat_after_relu1 = nn.ReLU(inplace=True)

        self.block12 = Block(features)

        self.final_conv = nn.Conv2d(in_channels=features,out_channels=1,kernel_size=3,padding=1)
    
    def forward(self, x):




        out = self.first_conv(x)
        out = self.first_relu(out)
        
        out1 = self.block11(out)
        # out_down1 = self.down1(out1)

        out2 = self.block21(out1)
        # out2 = self.block21(out_down1)
        # out_down2 = self.down2(out2)

        # out3 = self.block31(out_down2)
        # out_down3 = self.down3(out3)

        # middle_out = self.middle_block(out_down3)
        middle_out = self.middle_block(out2)
        # middle_out = self.middle_block(out_down2)

        # out_up3 = self.up3(middle_out)

        # out_cat3 = torch.cat((out3,out_up3),1)
        # out_cat_after3 = self.cat_after_conv3(out_cat3)
        # out_cat_after3 = self.cat_after_relu3(out_cat_after3)

        # out32 = self.block32(out_cat_after3)

        # out_up2 = self.up2(out32)
        # out_up2 = self.up2(middle_out)
        out_cat2 = torch.cat((out2,middle_out),1)
        # out_cat2 = torch.cat((out2,out_up2),1)
        out_cat_after2 = self.cat_after_conv2(out_cat2)
        out_cat_after2 = self.cat_after_relu2(out_cat_after2)
        out22 = self.blcok22(out_cat_after2)
        # out22 = self.blcok22(out_cat_after2)

        out_up1 = self.up1(out22)
        out_cat1 = torch.cat((out1,out_up1),1)
        out_cat_after1 = self.cat_after_conv1(out_cat1)
        out_cat_after1 = self.cat_after_relu1(out_cat_after1)
        out12 =  self.block12(out_up1)

        final_out = self.final_conv(out12)

        return final_out

        # out = self.first_conv(x)
        # out = self.first_relu(out)
        
        # out1 = self.block11(out)
        # out_down1 = self.down1(out1)

        # out2 = self.block21(out_down1)
        # out_down2 = self.down2(out2)

        # # out3 = self.block31(out_down2)
        # # out_down3 = self.down3(out3)

        # # middle_out = self.middle_block(out_down3)
        # middle_out = self.middle_block(out_down2)

        # # out_up3 = self.up3(middle_out)

        # # out_cat3 = torch.cat((out3,out_up3),1)
        # # out_cat_after3 = self.cat_after_conv3(out_cat3)
        # # out_cat_after3 = self.cat_after_relu3(out_cat_after3)

        # # out32 = self.block32(out_cat_after3)

        # # out_up2 = self.up2(out32)
        # out_up2 = self.up2(middle_out)
        # out_cat2 = torch.cat((out2,out_up2),1)
        # out_cat_after2 = self.cat_after_conv2(out_cat2)
        # out_cat_after2 = self.cat_after_relu2(out_cat_after2)
        # out22 = self.blcok22(out_cat_after2)

        # out_up1 = self.up1(out22)
        # out_cat1 = torch.cat((out1,out_up1),1)
        # out_cat_after1 = self.cat_after_conv1(out_cat1)
        # out_cat_after1 = self.cat_after_relu1(out_cat_after1)
        # out11 =  self.block11(out_cat_after1)

        # final_out = self.final_conv(out11)

        # return final_out
        # out = self.first_conv(x)
        # out = self.first_relu(out)
        
        # out1 = self.block11(out)
        # out_down1 = self.down1(out1)

        # out2 = self.block21(out_down1)
        # out_down2 = self.down2(out2)

        # # out3 = self.block31(out_down2)
        # # out_down3 = self.down3(out3)

        # # middle_out = self.middle_block(out_down3)
        # middle_out = self.middle_block(out_down2)

        # out_up2 = self.up2(middle_out)
        # out22 = self.blcok22(out_up2)

        # out_up1 = self.up1(out22)
        # out12 =  self.block12(out_up1)

        # final_out = self.final_conv(out12)

        # return final_out
        
    



            
class UNet(nn.Module):
    def __init__(self,input_channels):
        super(UNet, self).__init__()

        self.layers_normal = nn.ModuleList()
        self.layers_down = nn.ModuleList()

        self.layers_up = nn.ModuleList()
        self.layers_fuser = nn.ModuleList()

        # 开始时是两个Conv_3x3
        # 
        C_in = input_channels
        C_out = 40
        self.preprocess1 = nn.Sequential(
            ReLUConv(C_in, C_out,kernel_size=3,stride=1, padding=1,affine=False),
            ReLUConv(C_out,C_out,kernel_size=3, stride=1, padding=1, affine=False )
        )

        stride = 1

        # self.normal_block:
        self.normal_block = nn.ModuleList()
        # encoder's downsample
        self.down_block = nn.ModuleList()
        # decoder's upsample
        self.up_block = nn.ModuleList()

        # encode_block与decode_block之间的连接操作的输入，输出通道： encode_C_out+up_C_out,encode_C_out+up_C_out/2
        self.C_F=[40+40,40+40,40+32]

        #encoder_self.normal_block1:
        C_in ,C_out= C_out,40
        self.normal_block += [OPS['conv_3x3_no_bn'](C_in, C_out, stride, True)]
        C_in ,C_out= C_out,40
        self.normal_block += [OPS['resb'](C_in, C_out, stride, True)]
        C_in ,C_out= C_out,40
        self.normal_block += [OPS['resb'](C_in, C_out, stride, True)]

        # encoder's downsample
        
        # self.down_block1:
        C_in = C_out # 40
        C_out = 24 # 下采样的输出通道数
        self.down_block += [OPS['convolution_d'](C_in, C_out, stride, True)]


        # encoder_self.normal_block2:
        C_in, C_out = C_out, 32
        self.normal_block += [OPS['conv_3x3_no_bn'](C_in, C_out, stride, True)]
        C_in, C_out = C_out, 40
        self.normal_block += [OPS['sep_conv_5x5'](C_in, C_out, stride, True)]
        C_in, C_out = C_out, 40
        self.normal_block += [OPS['skip_connect'](C_in, C_out, stride, True)]
        # self.down_block2:
        C_in, C_out= C_out, 32
        self.down_block += [OPS['nearest_d'](C_in, C_out, stride, True)]

        # encoder_self.normal_block3:
        C_in, C_out = C_out, 40
        self.normal_block += [OPS['conv_3x3_no_bn'](C_in, C_out, stride, True)]
        C_in, C_out = C_out, 40
        self.normal_block += [OPS['skip_connect'](C_in, C_out, stride, True)]
        C_in, C_out = C_out, 40
        self.normal_block += [OPS['resb'](C_in, C_out, stride, True)]
        # self.down_block3:
        C_in,C_out = C_out, 32
        self.down_block += [OPS['nearest_d'](C_in, C_out, stride, True)]

        #decoder_self.normal_block1:
        C_in, C_out = C_out, 48
        self.normal_block += [OPS['conv_3x3_no_bn'](C_in, C_out, stride, True)]
        C_in, C_out = C_out, 48
        self.normal_block += [OPS['dil_conv_3x3'](C_in, C_out, stride, True)]
        C_in, C_out = C_out, 48
        self.normal_block += [OPS['skip_connect'](C_in, C_out, stride, True)]

        # self.up_block1:
        C_in,C_out = C_out, 40
        self.up_block += [OPS['bilinear_u'](C_in, C_out, stride, True)]
        C_out = int(self.C_F[0]/2) # 下一个decode 的输入通道数为encode与decode cat的输入通道数/2

        #decoder_self.normal_block2:
        C_in, C_out = C_out, 40
        self.normal_block += [OPS['skip_connect'](C_in, C_out, stride, True)]
        C_in, C_out = C_out, 40
        self.normal_block += [OPS['dil_conv_5x5'](C_in, C_out, stride, True)]
        C_in, C_out = C_out, 40
        self.normal_block += [OPS['skip_connect'](C_in, C_out, stride, True)]

        # self.up_block2:
        C_in,C_out = C_out, 40
        self.up_block += [OPS['bilinear_u'](C_in, C_out, stride, True)]
        C_out = int(self.C_F[1]/2)

        #decoder_self.normal_block3:
        C_in, C_out = C_out, 32
        self.normal_block += [OPS['resb'](C_in, C_out, stride, True)]
        C_in, C_out = C_out, 32
        self.normal_block += [OPS['skip_connect'](C_in, C_out, stride, True)]
        C_in, C_out = C_out, 32
        self.normal_block += [OPS['resb'](C_in, C_out, stride, True)]

        # self.up_block3:
        C_in,C_out = C_out, 32
        self.up_block += [OPS['bilinear_u'](C_in, C_out, stride, True)]
        C_out = int(self.C_F[2]/2)

        #final_self.normal_block2:
        C_in, C_out = C_out, 40
        self.normal_block += [OPS['resb'](C_in, C_out, stride, True)]
        C_in, C_out = C_out, 40
        self.normal_block += [OPS['resb'](C_in, C_out, stride, True)]
        C_in, C_out = C_out, 40
        self.normal_block += [OPS['skip_connect'](C_in, C_out, stride, True)]

        self.fuser_block=nn.ModuleList()
        
        

        # encode 与 decode 的cat操作
        for i in range(3):
            self.fuser_block.append(nn.Conv2d(in_channels=self.C_F[i],out_channels=int(self.C_F[i]/2),kernel_size=3,stride=1, padding=1,bias=False))

        # Unet 结束后 最后一个Conv2d 1x1的卷积只改变通道数
        self.final_layer = nn.Conv2d(in_channels=C_out,out_channels=input_channels,kernel_size=1,stride=1,padding=0,bias=False)
    
    def cat(self, x, map):
        # 计算map与x之间尺寸的差距，将x 扩充到map的尺寸，然后cat 
        diffY = torch.tensor([map.size()[2]-x.size()[2]])
        diffX = torch.tensor([map.size()[3]-x.size()[3]])

        x = F.pad(x,[diffX//2, diffX-diffX//2, diffY//2, diffY-diffY//2])
        return torch.cat([x, map], dim=1) # 按通道数cat



    def forward(self, x):
        x = self.preprocess1(x)
        self.encode_len = 3
        self.nodes=3
        encode_list = []
        for j in range(self.encode_len*2+1):

            # encode 与decode 之间的cat
            if j>self.encode_len:
                k = j -self.encode_len -1 
                x = self.up_block[k](x)
                encode_map = encode_list[self.encode_len-1-k]
                x_catted = self.cat(x,encode_map)
                x = self.fuser_block[k](x_catted)


            for i in  range(self.nodes):
                x = self.normal_block[j*self.nodes+i](x)
            
            if j<self.encode_len:
                encode_list.append(x)
                x = self.down_block[j](x)
        x = self.final_layer(x)
        # print(x.shape)
        return x
    


# op_names_normal, indices_normal = zip(*net_final.normal)
# print(op_names_normal,indices_normal)




    
    


            

        
        

    

        






        
