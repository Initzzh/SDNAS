import sys
import os
dir = os.getcwd()
sys.path.append(dir)
from genetic.individual import Individual,IndividualConnect
from setting.config import Config
from dip.denoise import denoise

# indi1 = Individual(Config.population_params, 0)
indi_connect = IndividualConnect(Config.population_params, 0)
dict = {
    'level_amount': 3,
    'encoder_units':[
            {'encoder_unit_id': 0, 'connect_type': 5, 'features': 48, 'conv_type': 'conv_7x7_leakyReLU', 'downsample_type': 'maxpool2d'},
            {'encoder_unit_id': 1, 'connect_type': 5, 'features': 48, 'conv_type': 'conv_5x5_leakyReLU', 'downsample_type': 'maxpool2d'},
            {'encoder_unit_id': 2, 'connect_type': 5, 'features': 48, 'conv_type': 'conv_5x5_leakyReLU', 'downsample_type': 'maxpool2d'},],
    'decoder_units':[
            {'decoder_unit_id': 0, 'connect_type': 5, 'features': 96, 'conv_type': 'conv_3x3_leakyReLU', 'upsample_type': 'nearest_u'},
            {'decoder_unit_id': 1, 'connect_type': 5, 'features': 96, 'conv_type': 'conv_5x5_leakyReLU', 'upsample_type': 'nearest_u'},
            {'decoder_unit_id': 2, 'connect_type': 5, 'features': 96, 'conv_type': 'conv_3x3_leakyReLU', 'upsample_type': 'nearest_u'},],
}



dict_list = []
indis = []
dict_name_list = []


for i in range(8):
    connect_type = i
    dict = {}
    dict["level_amount"] = 3
    encoder_units=[None for _ in range(dict["level_amount"])]
    decoder_units= [None for _ in range(dict["level_amount"])]
    for level in range(dict["level_amount"]):
        encoder_unit = {}
        encoder_unit["encoder_unit_id"] = level
        encoder_unit["connect_type"] = connect_type
        encoder_unit["features"] = 48
        encoder_unit["conv_type"] = "conv_3x3_leakyReLU"
        encoder_unit["downsample_type"] = "maxpool2d"
        encoder_units[level] = encoder_unit

        decoder_unit = {}
        decoder_unit["decoder_unit_id"] = level
        decoder_unit["connect_type"] = connect_type
        decoder_unit["features"] = 96
        decoder_unit["conv_type"] = "conv_3x3_leakyReLU"
        decoder_unit["upsample_type"] = "nearest_u"
        decoder_units[level] = decoder_unit
    dict["encoder_units"] = encoder_units
    dict["decoder_units"] = decoder_units
    dict_name = "connect_type_"+str(connect_type)
    dict_name_list.append(dict_name)
    dict_list.append(dict)

for index in range(len(dict_list)):
    indi = IndividualConnect(Config.population_params,dict_name_list[index])
    # indi.initialize_by_designed(dict_list[index])
    indi.initialize_by_designed(dict_list[index])
    indis.append(indi)
    print(indi)

# dict = {
#     "level_amount":3,
#         "encoder_units":[ 
#             {'encoder_unit_id': 0, 'block_amount': 3, 'features': 48, 'conv_type': 'dil_conv_5x5', 'downsample_type': 'maxpool2d'},
#             {'encoder_unit_id': 1, 'block_amount': 3, 'features': 48, 'conv_type': 'dil_conv_5x5', 'downsample_type': 'maxpool2d'},
#             {'encoder_unit_id': 2, 'block_amount': 3, 'features': 48, 'conv_type': 'dil_conv_5x5', 'downsample_type': 'maxpool2d'},  
#         ],

#         # "middle_units": [{'middle_unit_id': 0, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_3x3_leakyReLU'}],

#         "decoder_units": [
#             {'decoder_unit_id': 0, 'block_amount': 3, 'features': 96, 'conv_type': 'dil_conv_5x5', 'upsample_type': 'nearest_u'},
#             {'decoder_unit_id': 1, 'block_amount': 3, 'features': 96, 'conv_type': 'dil_conv_5x5', 'upsample_type': 'nearest_u'},
#             {'decoder_unit_id': 2, 'block_amount': 3, 'features': 96, 'conv_type': 'dil_conv_5x5', 'upsample_type': 'nearest_u'},
           
#             ]
# }

# indi = Individual(Config.population_params, "net_level_3")
# indi.initialize_by_designed(dict)
# indis = [indi]

denoise(None, indis)   
    





#  for i in range(pow(2,level_mount-1 )):
  
#         scale = [0 for _ in range(level_mount)]
#         # scale[0] = 0
#         val = i
#         j = level_mount-1
#         while(val>0):
#             scale[j] = val % 2
#             j -= 1
#             val = val//2
        
#         print(scale)
#         scale_list.append(scale)
#     for scale in scale_list:
#         scale_val = 0
#         for i in range(level_mount):
#             scale_val += pow(2, level_mount-1-i)*scale[i]
#         dict_name = "scale_" + str(scale_val)
#         dict = {
#             "level_amount":5,
#             "encoder_units":[ 
#                 {'encoder_unit_id': 0, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_3x3_leakyReLU', 'downsample_type': 'maxpool2d'},
#                 {'encoder_unit_id': 1, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_3x3_leakyReLU', 'downsample_type': 'maxpool2d'},
#                 {'encoder_unit_id': 2, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_3x3_leakyReLU', 'downsample_type': 'maxpool2d'}, 
#                 {'encoder_unit_id': 3, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_3x3_leakyReLU', 'downsample_type': 'maxpool2d'}, 
#                 {'encoder_unit_id': 4, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_3x3_leakyReLU', 'downsample_type': 'maxpool2d'}, 
#             ],

#                 # "middle_units": [{'middle_unit_id': 0, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_3x3_leakyReLU'}],

#             "decoder_units": [
#                 {'decoder_unit_id': 0, 'block_amount': 2, 'features': 96, 'conv_type': 'conv_3x3_leakyReLU', 'upsample_type': 'nearest_u'},
#                 {'decoder_unit_id': 1, 'block_amount': 2, 'features': 96, 'conv_type': 'conv_3x3_leakyReLU', 'upsample_type': 'nearest_u'},
#                 {'decoder_unit_id': 2, 'block_amount': 2, 'features': 96, 'conv_type': 'conv_3x3_leakyReLU', 'upsample_type': 'nearest_u'},
#                 {'decoder_unit_id': 3, 'block_amount': 2, 'features': 96, 'conv_type': 'conv_3x3_leakyReLU', 'upsample_type': 'nearest_u'},
#                 {'decoder_unit_id': 4, 'block_amount': 1, 'features': 96, 'conv_type': 'conv_3x3_leakyReLU', 'upsample_type': 'nearest_u'},
#                 ]
#         }
#         dict['scale'] = scale
#         # print(dict['scale'])
#         # print(dict_name)
#         dict_list.append(dict)
#         dict_name_list.append(dict_name)
    
#     indis = []
#     for index in range(len(dict_list)):
#         indi = Individual(Config.population_params,dict_name_list[index])
#         # indi.initialize_by_designed(dict_list[index])
#         indi.initialize_scale_by_designed(dict_list[index])
#         indis.append(indi)



# indi_connect.initialize_by_designed(dict)
# indi = [indi_connect]

