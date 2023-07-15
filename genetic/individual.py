import numpy as np

import sys
import os

# dir = os.path.abspath("..")
dir = os.getcwd()
# print(dir)
# print(sys.path)
sys.path.append(dir)
# print(sys.path)
# from GA.unit import EncoderUnit, DecoderUnit

from genetic.unit import EncoderUnit, DecoderUnit, MiddleUnit
from genetic.unit import EncoderUnitConnect, DecoderUnitConnect
from setting.config import Config

# params = {
#     "min_level_amount":2,
#     "max_level_amount":4,
#     "min_block_amount":1,
#     "max_block_amount":4,
#     "features":64,
#     "conv_types":["conv_3x3","conv_5x5"],
#     "downsample_types":["maxpool2d"],
#     "upsample_types":["ConvTranspose2d"],

# }

class Individual(object):
    """个体
    """

    def __init__(self, params, indi_id):
        """初始化个体

        Args:
            params(dict): 个体配置参数
            indi_id(str): 个体编号
            level_amount: Unet层级
            encoder_units: encoder单元list
            decoder_units: decoder单元list


        """

        self.id = indi_id
        self.params = params
        self.psnr = -1.0

        self.level_amount = np.random.randint(self.params["min_level_amount"], self.params["max_level_amount"]+1) # Unet网络层级数量
        # self.middle_unit_amount = self.params["middle_unit_amount"] # Unet的中间层数量

        self.encoder_units = []  # Unet网络encoder单元
        self.decoder_units = [] # Unet网络decoder单元
        # self.middle_units = [] # Unet 网络middle_unit 单元
        self.encoder_unit_id=0 # encoder_unit id
        self.decoder_unit_id=0
        # self.middle_unit_id=0
        self.min_level_amount = self.params["min_level_amount"]  # Unet层级的最小数量
        self.max_level_amount = self.params["max_level_amount"]  # Unet层级的最大数量

        

        self.min_block_amount = self.params["min_block_amount"] # encoder_unit/decoder_unit中block的最小数量
        self.max_block_amount = self.params["max_block_amount"] # encoder_unit/decoder_unit中block的最大数量

        self.features = self.params["features"] # block的输入输出通道数

        self.conv_types = self.params["conv_types"]
        self.conv_type_length = len(self.conv_types)

        self.downsample_types = self.params["downsample_types"]
        self.downsample_type_length = len(self.downsample_types)
        
        self.upsample_types = self.params["upsample_types"]
        self.upsample_type_length = len(self.upsample_types)



    def init_encoder(self):
        """encoder 单元初始化

        Args:
        Retuens:
            EncoderUnit: 返回EncoderUnit
        """
        encoder_unit_id = self.encoder_unit_id
        self.encoder_unit_id +=1
        # block_amount = np.random.randint(self.min_block_amount,self.max_block_amount+1) # encoder中block数量，
        block_amount = 1 # 只改变conv_type， dsl_type

        # TODO: 随机生成features
        features = self.features # encoder 中特征图数量即out_channels
        conv_type = self.conv_types[np.random.randint(0,self.conv_type_length)] # 从conv_type列表中随机选取一种conv

        downsample_type = self.downsample_types[np.random.randint(0,self.downsample_type_length)] # 从downsample_type列表中随机选取一种downsample
        
        encoder_unit =EncoderUnit(encoder_unit_id,block_amount,features,conv_type,downsample_type)
        return encoder_unit

    def init_decoder(self, level):
        """decoder 单元初始化

        Args:
        Returns:
            DecoderUnit: 返回DecoderUnit
        
        """
        decoder_unit_id = self.decoder_unit_id
        self.decoder_unit_id += 1

        # block_amount = np.random.randint(self.min_block_amount,self.max_block_amount+1) # decoder中block数量
        
        # 固定block_amount
        if level == self.level_amount-1:
            block_amount = 1
        else:
            block_amount = 2

        features = self.features * 2 # decoder features 96
        conv_type = self.conv_types[np.random.randint(0,self.conv_type_length)]  # 从conv_type列表中随机选取一种conv

        upsample_type = self.upsample_types[np.random.randint(0,self.upsample_type_length)] # 从upsample_type 列表中随机选取一种upsample

        decoder_unit =DecoderUnit(decoder_unit_id,block_amount,features,conv_type, upsample_type)
        
        return decoder_unit

    def init_middle(self):
        """middle_unit 初始化

        Arags:

        Returns:
            MiddleUnit: 返回MiddleUnit

        """
        middle_unit_id = self.middle_unit_id
        self.middle_unit_id += 1

        block_amount = np.random.randint(self.min_block_amount,self.max_block_amount+1) # decoder中block数量

        features = self.features
        conv_type = self.conv_types[np.random.randint(0,self.conv_type_length)]  # 从conv_type列表中随机选取一种conv


        middle_unit = MiddleUnit(middle_unit_id,block_amount,features,conv_type)
        
        return middle_unit
    
    def initialize(self):
        for i in range(self.level_amount):
            self.encoder_units.append(self.init_encoder())
            self.decoder_units.append(self.init_decoder(i))
        
        # middle_units
        # for i in range(self.middle_unit_amount):
        #     self.middle_units.append(self.init_middle())

    def initialize_scale_by_designed(self,dict):
        """根据dict 初始化个体

        Arags:
            dict (dict):
                dict = {
                        "level_amount":3
                        "middle_unit_amount":1
                        "encoder_units": [
                            {'encoder_unit_id': 0, 'block_amount': 1, 'features': 64, 'conv_type': 'conv_3x3', 'downsample_type': 'convolution_d'},
                            {'encoder_unit_id': 1, 'block_amount': 4, 'features': 64, 'conv_type': 'conv_3x3', 'downsample_type': 'maxpool2d'},
                            {'encoder_unit_id': 2, 'block_amount': 2, 'features': 64, 'conv_type': 'conv_5x5', 'downsample_type': 'maxpool2d'}
                        ]

                        "middle_units": [
                            {'middle_unit_id': 0, 'block_amount': 4, 'features': 64, 'conv_type': 'conv_5x5'}
                        ]

                        "decoder_units": [
                            {'decoder_unit_id': 0, 'block_amount': 4, 'features': 64, 'conv_type': 'conv_3x3', 'upsample_type': 'deconv'},
                            {'decoder_unit_id': 1, 'block_amount': 1, 'features': 64, 'conv_type': 'conv_3x3', 'upsample_type': 'deconv'},
                            {'decoder_unit_id': 2, 'block_amount': 2, 'features': 64, 'conv_type': 'conv_5x5', 'upsample_type': 'sub_pixel_u'}
                        ]
                    }
            
        """
        self.scale = dict["scale"]
        self.level_amount = dict["level_amount"]
        # self.middle_unit_amount = dict["middle_unit_amount"]
        encoder_units_dict = dict["encoder_units"]
        decoder_units_dict = dict["decoder_units"]
        # middle_units_dict = dict["middle_units"]

        for level in range(self.level_amount):

            # encoder_unit
            self.encoder_units.append(
                EncoderUnit(encoder_units_dict[level]['encoder_unit_id'], encoder_units_dict[level]["block_amount"], encoder_units_dict[level]["features"],
                            encoder_units_dict[level]["conv_type"], encoder_units_dict[level]["downsample_type"])
                )
            # decoder_unit
            self.decoder_units.append(
                DecoderUnit(decoder_units_dict[level]['decoder_unit_id'], decoder_units_dict[level]["block_amount"], decoder_units_dict[level]["features"],
                            decoder_units_dict[level]["conv_type"], decoder_units_dict[level]["upsample_type"])
            )


    def initialize_by_designed(self,dict):
        """根据dict 初始化个体

        Arags:
            dict (dict):
                dict = {
                        "level_amount":3
                        "middle_unit_amount":1
                        "encoder_units": [
                            {'encoder_unit_id': 0, 'block_amount': 1, 'features': 64, 'conv_type': 'conv_3x3', 'downsample_type': 'convolution_d'},
                            {'encoder_unit_id': 1, 'block_amount': 4, 'features': 64, 'conv_type': 'conv_3x3', 'downsample_type': 'maxpool2d'},
                            {'encoder_unit_id': 2, 'block_amount': 2, 'features': 64, 'conv_type': 'conv_5x5', 'downsample_type': 'maxpool2d'}
                        ]

                        "middle_units": [
                            {'middle_unit_id': 0, 'block_amount': 4, 'features': 64, 'conv_type': 'conv_5x5'}
                        ]

                        "decoder_units": [
                            {'decoder_unit_id': 0, 'block_amount': 4, 'features': 64, 'conv_type': 'conv_3x3', 'upsample_type': 'deconv'},
                            {'decoder_unit_id': 1, 'block_amount': 1, 'features': 64, 'conv_type': 'conv_3x3', 'upsample_type': 'deconv'},
                            {'decoder_unit_id': 2, 'block_amount': 2, 'features': 64, 'conv_type': 'conv_5x5', 'upsample_type': 'sub_pixel_u'}
                        ]
                    }
            
        """
        # self.scale = dict["scale"]
        self.level_amount = dict["level_amount"]
        # self.middle_unit_amount = dict["middle_unit_amount"]
        encoder_units_dict = dict["encoder_units"]
        decoder_units_dict = dict["decoder_units"]
        # middle_units_dict = dict["middle_units"]

        for level in range(self.level_amount):

            # encoder_unit
            self.encoder_units.append(
                EncoderUnit(encoder_units_dict[level]['encoder_unit_id'], encoder_units_dict[level]["block_amount"], encoder_units_dict[level]["features"],
                            encoder_units_dict[level]["conv_type"], encoder_units_dict[level]["downsample_type"])
                )
            # decoder_unit
            self.decoder_units.append(
                DecoderUnit(decoder_units_dict[level]['decoder_unit_id'], decoder_units_dict[level]["block_amount"], decoder_units_dict[level]["features"],
                            decoder_units_dict[level]["conv_type"], decoder_units_dict[level]["upsample_type"])
            )
        
        # for middle_index in range(self.middle_unit_amount):

        #     self.middle_units.append(
        #         MiddleUnit(middle_units_dict[middle_index]["middle_unit_id"], middle_units_dict[middle_index]["block_amount"], middle_units_dict[middle_index]["features"],
        #                    middle_units_dict[middle_index]["conv_type"])
        #     )

        
    
    def reset(self):
        self.psnr = -1.0

    

    def __str__(self):

        encoder_str = ""
        decoder_str = ""
        level_str = "'level_amount': " + str(self.level_amount) +",\n"
        # dict = {
        #     "level_amount": self.level_amount,
        #     "encoder_units": 
            
        # }
        encoder_dicts = [None for _ in range(self.level_amount)]
        decoder_dicts = [None for _ in range(self.level_amount)]
        for i in range(self.level_amount):
            encoder_dicts[i] = self.encoder_units[i].get_dict()
            decoder_dicts[i] = self.decoder_units[i].get_dict()
        
        dict = {
            "level_amount": self.level_amount,
            "encoder_units": encoder_dicts,
            "decoder_units": decoder_dicts
        }
        encoder_str = "'encoder_units':[\n"
        decoder_str = "'decoder_units':[\n"
        for i in range(self.level_amount):
            encoder_str += "\t" + str(encoder_dicts[i])+','
            decoder_str += "\t" + str(decoder_dicts[i])+','
            if i == self.level_amount-1:
                encoder_str += "],"
                decoder_str += "],"
           
            # else:
            encoder_str += "\n"
            decoder_str += "\n"
        
        indi_dict_str = "{\n"+level_str+encoder_str+ decoder_str+"}\n"

        return "\nindi_id:"+str(self.id)+"\n"+indi_dict_str+"psnr:"+str(self.psnr)+'\n'
        # encoder_str += "encoder_units: "
        # for i in range(self.level_amount):
        #     # encoder_str += str(encoder_units[i])+"\n"
        #     encoder_str += str(encoder_dicts[i])+"\n"
        #     # print(self.encoder_units[i])
        
        # # for i in range(self.middle_unit_amount):
        # #     middle_str += str(self.middle_units[i])+"\n"
        # for i in range(self.level_amount):
        #     decoder_str += str(self.decoder_units[i])+"\n"
            # print(self.decoder_units[i])
        # return "\nindi_id:"+str(self.id)+"\n"+"level:"+str(self.level_amount)+"\n"+encoder_str+"\n"+middle_str+"\n"+decoder_str+"psnr:"+str(self.psnr)+'\n'
        # # return "\nindi_id:"+str(self.id)+"\n"+"level:"+str(self.level_amount)+"\n"+encoder_str+"\n"+middle_str+"\n"+decoder_str+'\n'


class IndividualConnect(object):
    """个体
    """

    def __init__(self, params, indi_id):
        """初始化个体

        Args:
            params(dict): 个体配置参数
            indi_id(str): 个体编号
            level_amount: Unet层级
            encoder_units: encoder单元list
            decoder_units: decoder单元list
        """

        self.id = indi_id
        self.params = params
        self.psnr = -1.0

        self.level_amount = np.random.randint(self.params["min_level_amount"], self.params["max_level_amount"]+1) # Unet网络层级数量
        # self.middle_unit_amount = self.params["middle_unit_amount"] # Unet的中间层数量

        self.encoder_units = []  # Unet网络encoder单元
        self.decoder_units = [] # Unet网络decoder单元
        # self.middle_units = [] # Unet 网络middle_unit 单元
        self.encoder_unit_id=0 # encoder_unit id
        self.decoder_unit_id=0
        # self.middle_unit_id=0
        self.min_level_amount = self.params["min_level_amount"]  # Unet层级的最小数量
        self.max_level_amount = self.params["max_level_amount"]  # Unet层级的最大数量

        

        self.min_block_amount = self.params["min_block_amount"] # encoder_unit/decoder_unit中block的最小数量
        self.max_block_amount = self.params["max_block_amount"] # encoder_unit/decoder_unit中block的最大数量

        self.connect_types = self.params["connect_types"] # unit 中block的连接方式

        # self.features_types = self.params["features_types"] # 通道数类型
        self.encoder_features_types = self.params["encoder_features_types"] # encoder 通道数
        self.decoder_features_types = self.params["decoder_features_types"] # decoder 通道数

        self.features = self.params["features"] # block的输入输出通道数
        

        self.conv_types = self.params["conv_types"]
        self.conv_type_length = len(self.conv_types)

        self.downsample_types = self.params["downsample_types"]
        self.downsample_type_length = len(self.downsample_types)
        
        self.upsample_types = self.params["upsample_types"]
        self.upsample_type_length = len(self.upsample_types)



    def init_encoder(self):
        """encoder 单元初始化

        Args:
        Retuens:
            EncoderUnit: 返回EncoderUnit
        """
        encoder_unit_id = self.encoder_unit_id
        self.encoder_unit_id +=1
        # 随机生block_amount
        # block_amount = np.random.randint(self.min_block_amount,self.max_block_amount+1) # encoder中block数量，
        # block_amount = 1 # 只改变conv_type， dsl_type

        # 随机生成block 连接方式
        connect_type = np.random.choice(self.connect_types,size = 1)[0]


        # TODO: 随机生成features
        # features = self.features # encoder 中特征图数量即out_channels
        features = np.random.choice(self.encoder_features_types,size=1)[0] # encoder_features

        conv_type = self.conv_types[np.random.randint(0,self.conv_type_length)] # 从conv_type列表中随机选取一种conv

        downsample_type = self.downsample_types[np.random.randint(0,self.downsample_type_length)] # 从downsample_type列表中随机选取一种downsample
        
        # encoder_unit =EncoderUnit(encoder_unit_id,block_amount,features,conv_type,downsample_type)
        encoder_unit = EncoderUnitConnect(encoder_unit_id, connect_type, features, conv_type, downsample_type)
        return encoder_unit

    def init_decoder(self, level):
        """decoder 单元初始化

        Args:
        Returns:
            DecoderUnit: 返回DecoderUnit
        
        """
        decoder_unit_id = self.decoder_unit_id
        self.decoder_unit_id += 1

        # block_amount = np.random.randint(self.min_block_amount,self.max_block_amount+1) # decoder中block数量
        
        # # 固定block_amount
        # if level == self.level_amount-1:
        #     block_amount = 1
        # else:
        #     block_amount = 2
        
         # 随机生成block 连接方式
        connect_type = np.random.choice(self.connect_types,size = 1)[0]

        features = np.random.choice(self.decoder_features_types, size=1)[0] # decoder_features

        # features = self.features * 2 # decoder features 96
        conv_type = self.conv_types[np.random.randint(0,self.conv_type_length)]  # 从conv_type列表中随机选取一种conv

        upsample_type = self.upsample_types[np.random.randint(0,self.upsample_type_length)] # 从upsample_type 列表中随机选取一种upsample

        # decoder_unit =DecoderUnit(decoder_unit_id,block_amount,features,conv_type, upsample_type)
        decoder_unit = DecoderUnitConnect(decoder_unit_id, connect_type, features, conv_type, upsample_type)
        
        return decoder_unit

    def init_middle(self):
        """middle_unit 初始化

        Arags:

        Returns:
            MiddleUnit: 返回MiddleUnit

        """
        middle_unit_id = self.middle_unit_id
        self.middle_unit_id += 1

        block_amount = np.random.randint(self.min_block_amount,self.max_block_amount+1) # decoder中block数量

        features = self.features
        conv_type = self.conv_types[np.random.randint(0,self.conv_type_length)]  # 从conv_type列表中随机选取一种conv


        middle_unit = MiddleUnit(middle_unit_id,block_amount,features,conv_type)
        
        return middle_unit
    
    def initialize(self):
        for i in range(self.level_amount):
            self.encoder_units.append(self.init_encoder())
            self.decoder_units.append(self.init_decoder(i))
        
        # middle_units
        # for i in range(self.middle_unit_amount):
        #     self.middle_units.append(self.init_middle())

    def initialize_by_designed(self,dict):
        """根据dict 初始化个体

        Arags:
            dict (dict):
                dict = {
                        "level_amount":3
                        "middle_unit_amount":1
                        "encoder_units": [
                            {'encoder_unit_id': 0, 'block_amount': 1, 'features': 64, 'conv_type': 'conv_3x3', 'downsample_type': 'convolution_d'},
                            {'encoder_unit_id': 1, 'block_amount': 4, 'features': 64, 'conv_type': 'conv_3x3', 'downsample_type': 'maxpool2d'},
                            {'encoder_unit_id': 2, 'block_amount': 2, 'features': 64, 'conv_type': 'conv_5x5', 'downsample_type': 'maxpool2d'}
                        ]

                        "middle_units": [
                            {'middle_unit_id': 0, 'block_amount': 4, 'features': 64, 'conv_type': 'conv_5x5'}
                        ]

                        "decoder_units": [
                            {'decoder_unit_id': 0, 'block_amount': 4, 'features': 64, 'conv_type': 'conv_3x3', 'upsample_type': 'deconv'},
                            {'decoder_unit_id': 1, 'block_amount': 1, 'features': 64, 'conv_type': 'conv_3x3', 'upsample_type': 'deconv'},
                            {'decoder_unit_id': 2, 'block_amount': 2, 'features': 64, 'conv_type': 'conv_5x5', 'upsample_type': 'sub_pixel_u'}
                        ]
                    }
            
        """
        self.level_amount = dict["level_amount"]
        # self.middle_unit_amount = dict["middle_unit_amount"]
        encoder_units_dict = dict["encoder_units"]
        decoder_units_dict = dict["decoder_units"]
        # middle_units_dict = dict["middle_units"]

        for level in range(self.level_amount):

            # encoder_unit
            self.encoder_units.append(
                EncoderUnitConnect(encoder_units_dict[level]['encoder_unit_id'],encoder_units_dict[level]['connect_type'],encoder_units_dict[level]["features"], encoder_units_dict[level]["conv_type"], encoder_units_dict[level]["downsample_type"])
                # EncoderUnit(encoder_units_dict[level]['encoder_unit_id'], encoder_units_dict[level]["block_amount"], encoder_units_dict[level]["features"],
                #             encoder_units_dict[level]["conv_type"], encoder_units_dict[level]["downsample_type"])
                )
            # decoder_unit
            self.decoder_units.append(
                DecoderUnitConnect(decoder_units_dict[level]['decoder_unit_id'], decoder_units_dict[level]["connect_type"], decoder_units_dict[level]["features"],decoder_units_dict[level]["conv_type"], decoder_units_dict[level]["upsample_type"])
                # DecoderUnit(decoder_units_dict[level]['decoder_unit_id'], decoder_units_dict[level]["block_amount"], decoder_units_dict[level]["features"],
                #             decoder_units_dict[level]["conv_type"], decoder_units_dict[level]["upsample_type"])
            )
        
        # for middle_index in range(self.middle_unit_amount):

        #     self.middle_units.append(
        #         MiddleUnit(middle_units_dict[middle_index]["middle_unit_id"], middle_units_dict[middle_index]["block_amount"], middle_units_dict[middle_index]["features"],
        #                    middle_units_dict[middle_index]["conv_type"])
        #     )

        
    
    def reset(self):
        self.psnr = -1.0

    

    def __str__(self):

        encoder_str = ""
        decoder_str = ""
        level_str = "'level_amount': " + str(self.level_amount) +",\n"
        # dict = {
        #     "level_amount": self.level_amount,
        #     "encoder_units": 
            
        # }
        encoder_dicts = [None for _ in range(self.level_amount)]
        decoder_dicts = [None for _ in range(self.level_amount)]
        for i in range(self.level_amount):
            encoder_dicts[i] = self.encoder_units[i].get_dict()
            decoder_dicts[i] = self.decoder_units[i].get_dict()
        
        dict = {
            "level_amount": self.level_amount,
            "encoder_units": encoder_dicts,
            "decoder_units": decoder_dicts
        }
        encoder_str = "'encoder_units':[\n"
        decoder_str = "'decoder_units':[\n"
        for i in range(self.level_amount):
            encoder_str += "\t" + str(encoder_dicts[i])+','
            decoder_str += "\t" + str(decoder_dicts[i])+','
            if i == self.level_amount-1:
                encoder_str += "],"
                decoder_str += "],"
           
            # else:
            encoder_str += "\n"
            decoder_str += "\n"
        
        indi_dict_str = "{\n"+level_str+encoder_str+ decoder_str+"}\n"

        return "\nindi_id:"+str(self.id)+"\n"+indi_dict_str+"psnr:"+str(self.psnr)+'\n'
    
if __name__=="__main__":
    config = Config()
    # for _ in range(1):
    #     indi = Individual(config.population_params,_)
    #     indi.initialize()
    #     print(indi)
    
    dict = {
            "level_amount":2,
         
            "encoder_units":[ 
                {'encoder_unit_id': 0, 'block_amount': 3, 'features': 48, 'conv_type': 'conv_3x3', 'downsample_type': 'maxpool2d'},
                {'encoder_unit_id': 1, 'block_amount': 3, 'features': 48, 'conv_type': 'conv_3x3', 'downsample_type': 'maxpool2d'}, ],

            "decoder_units": [
                {'decoder_unit_id': 0, 'block_amount': 3, 'features': 96, 'conv_type': 'conv_3x3', 'upsample_type': 'deconv'},
                {'decoder_unit_id': 1, 'block_amount': 3, 'features': 96, 'conv_type': 'conv_3x3', 'upsample_type': 'deconv'},]
        }
    

    dict = {
       
        "level_amount":3,
        
        "encoder_units":[ 
            {'encoder_unit_id': 0, 'connect_type': 0, 'features': 48, 'conv_type': 'conv_7x7_leakyReLU', 'downsample_type': 'nearest_d'},
            {'encoder_unit_id': 1, 'connect_type': 0, 'features': 48, 'conv_type': 'conv_5x5_leakyReLU', 'downsample_type': 'convolution_d'},
            {'encoder_unit_id': 2, 'connect_type': 4, 'features': 48, 'conv_type': 'conv_5x5_leakyReLU', 'downsample_type': 'bilinear_d'}, ],

        "decoder_units": [
            {'decoder_unit_id': 0, 'connect_type': 6, 'features': 96, 'conv_type': 'conv_3x3_leakyReLU', 'upsample_type': 'sub_pixel_u'},
            {'decoder_unit_id': 1, 'connect_type': 5, 'features': 96, 'conv_type': 'conv_5x5_leakyReLU', 'upsample_type': 'area_u'},
            {'decoder_unit_id': 2, 'connect_type': 3, 'features': 96, 'conv_type': 'conv_3x3_leakyReLU', 'upsample_type': 'bilinear_u'},]
        
    }


    # indi = Individual(config.population_params, indi_id = 0)
    # indi.initialize()

    indi = IndividualConnect(Config.population_params, indi_id=0)
    indi.initialize_by_designed(dict)

    # indi.initialize_by_designed(dict)
    
    print(indi)
        


        
        


        

        
        
        
        

        

