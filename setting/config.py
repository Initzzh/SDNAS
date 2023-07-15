class Config(object):
    population_params={

        # 个体参数设置
        "min_level_amount":5,
        "max_level_amount":5,
        "middle_unit_amount":1,
        "min_block_amount":1,
        "max_block_amount":2,

        "connect_types":[0,1,2,3,4,5,6,7], # 000,001,010,011,100,101,110,111
        # "features_types": [32,48,64,80,96],
        "encoder_features_types": [16,32,48],
        "decoder_features_types": [64,80,96],
        # "features_types": [16,32,48,64,80,96],
        "features":48,
        # "conv_types":["conv_3x3", "conv_5x5", "conv_7x7","dil_conv_3x3","dil_conv_5x5","conv_3x3_leakyReLU"],
        "conv_types":["conv_3x3_leakyReLU", "conv_5x5_leakyReLU", "conv_7x7_leakyReLU"],
        "downsample_types":["maxpool2d", "convolution_d", "bilinear_d", "nearest_d", "area_d"],
        "upsample_types":["deconv", "sub_pixel_u", "bilinear_u", "nearest_u","area_u"],

        # 种群设置
        "pop_size" : 30, # 种群大小 30
        "select_num": 10, # 变异选择个体数量 10
        "best_indi_num": 5, # 交叉保留最好个体数量5
        "gen": 50,  # 进化次数 50
        "crossover_prob": 0.7, # 交叉率
        "mutation_prob":0.3, # 变异率
        

        # log
        "log_path" : "logs/v6.27"
        # "record_path":
        
    }

    train_params = {
        # Net train参数
        "epoch": 50,
        "milestone": 30,
        "lr": 1e-3,
        "batch_size":128,
        "noise_mode":"S",
        "noiseL": 25,
        "val_noiseL": 25,
        "partition_degree":1,
        "outf": 'logs/test/save_model',
        "log_path":"logs/test/newUnet.log"
        
    }