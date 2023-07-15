class EncoderUnit(object):
    def __init__(self,encoder_unit_id, block_amount, features,conv_type, downsample_type):
        """encoder_unit 

        
        """
        self.encoder_unit_id = encoder_unit_id
        self.block_amount = block_amount
        self.features = features
        self.conv_type = conv_type
        self.downsample_type = downsample_type
    
    def get_dict(self):
        dict = {"encoder_unit_id": self.encoder_unit_id,
                "block_amount":self.block_amount,
                 "features":self.features,
                 "conv_type":self.conv_type,
                 "downsample_type":self.downsample_type}
        return dict

    def __str__(self):
        dict = {"encoder_unit_id": self.encoder_unit_id,
                "block_amount":self.block_amount,
                 "features":self.features,
                 "conv_type":self.conv_type,
                 "downsample_type":self.downsample_type}
        return "encoder_unit: "+ str(dict)

class EncoderUnitConnect(object):
    def __init__(self, encoder_unit_id, connect_type, features, conv_type, downsample_type):
        """
            encoder_unit
            encoder_unit_id:
            connect_type: 卷积块之间的连接结构
            features: 输出通道特征图数
            conv_type: 卷积类型
            downsample_type: 下采样类型
        """
        self.encoder_unit_id = encoder_unit_id
        self.connect_type = connect_type
        self.features = features
        self.conv_type = conv_type
        self.downsample_type = downsample_type
    
    def get_dict(self):
        dict = {"encoder_unit_id": self.encoder_unit_id,
                "connect_type":self.connect_type,
                 "features":self.features,
                 "conv_type":self.conv_type,
                 "downsample_type":self.downsample_type}
        return dict
    
    def __str__(self):
        dict = self.get_dict()
        return "encoder_unit: "+ str(dict)

        




class DecoderUnit(object):
    def __init__(self,decoder_unit_id, block_amount, features,conv_type, upsample_type):
        """decoder_unit 

        
        """
        self.decoder_unit_id = decoder_unit_id
        self.block_amount = block_amount
        self.features = features
        self.conv_type = conv_type
        self.upsample_type = upsample_type

    def get_dict(self):
        dict = {"decoder_unit_id": self.decoder_unit_id,
                "block_amount":self.block_amount,
                 "features":self.features,
                 "conv_type":self.conv_type,
                 "upsample_type":self.upsample_type}
        return dict
    
    def __str__(self) -> str:
        dict = {"decoder_unit_id": self.decoder_unit_id,
                "block_amount":self.block_amount,
                 "features":self.features,
                 "conv_type":self.conv_type,
                 "upsample_type":self.upsample_type}
        # print(dict)
        return "decoder_unit: " + str(dict)
    # def __str__(self):
    #     dict = {"decoder_unit_id": self.decoder_unit_id,
    #             "block_amount":self.block_amount,
    #              "features":self.features,
    #              "conv_type":self.conv_type,
    #              "upsample_type":self.upsample_type}
    #     print(dict)
    #     return "decoder_unit: " + str(dict)



class DecoderUnitConnect(object):
    def __init__(self,decoder_unit_id, connect_type, features,conv_type, upsample_type):
        """decoder_unit 

        
        """
        self.decoder_unit_id = decoder_unit_id
        self.connect_type = connect_type
        self.features = features
        self.conv_type = conv_type
        self.upsample_type = upsample_type

    def get_dict(self):
        dict = {"decoder_unit_id": self.decoder_unit_id,
                "connect_type":self.connect_type,
                 "features":self.features,
                 "conv_type":self.conv_type,
                 "upsample_type":self.upsample_type}
        return dict
    
    def __str__(self) -> str:
        dict = self.get_dict()
        # print(dict)
        return "decoder_unit: " + str(dict)



class MiddleUnit(object):
    def __init__(self, middle_unit_id, block_amount, features, conv_type):
        """middle_unit
        """
        self.middle_unit_id = middle_unit_id
        self.block_amount = block_amount
        self.features = features
        self.conv_type = conv_type
    
    def __str__(self):
        dict = {"middle_unit_id": self.middle_unit_id,
                "block_amount":self.block_amount,
                 "features":self.features,
                 "conv_type":self.conv_type,
                }
        # print(dict)
        return "middle_unit: " + str(dict)