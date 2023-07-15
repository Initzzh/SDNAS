if __name__=="__main__":
    from genetic.individual import Individual
    from setting.config import Config
    from evaluate.evaluate_indi import EvaluateIndi

    from dip_denoising.inference import inference
    # inference(indi=None)
    # return 
    
    indi = Individual(Config.population_params,0)
    # dict = {
    #     "level_amount":4,
    #     "middle_unit_amount":1,
    #     "encoder_units":[ 
    #         {'encoder_unit_id': 0, 'block_amount': 2, 'features': 48, 'conv_type': 'conv_5x5', 'downsample_type': 'maxpool2d'},
    #         {'encoder_unit_id': 1, 'block_amount': 2, 'features': 48, 'conv_type': 'conv_5x5', 'downsample_type': 'maxpool2d'},
    #         {'encoder_unit_id': 2, 'block_amount': 2, 'features': 48, 'conv_type': 'conv_5x5', 'downsample_type': 'maxpool2d'}, 
    #         {'encoder_unit_id': 3, 'block_amount': 2, 'features': 48, 'conv_type': 'conv_5x5', 'downsample_type': 'maxpool2d'},],

    #     "middle_units": [{'middle_unit_id': 0, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_5x5'}],

    #     "decoder_units": [
    #         {'decoder_unit_id': 0, 'block_amount': 2, 'features': 48, 'conv_type': 'conv_5x5', 'upsample_type': 'deconv'},
    #         {'decoder_unit_id': 1, 'block_amount': 2, 'features': 48, 'conv_type': 'conv_5x5', 'upsample_type': 'deconv'},
    #         {'decoder_unit_id': 2, 'block_amount': 2, 'features': 48, 'conv_type': 'conv_5x5', 'upsample_type': 'deconv'},
    #         {'decoder_unit_id': 3, 'block_amount': 2, 'features': 48, 'conv_type': 'conv_5x5', 'upsample_type': 'deconv'},]
    # }

    # dict = {

    #     "level_amount":4,
    #     "middle_unit_amount":1,

    #     "encoder_units":[ {'encoder_unit_id': 0, 'block_amount': 1, 'features': 48, 'conv_type': 'sep_conv_5x5', 'downsample_type': 'area_d'},
    #                 {'encoder_unit_id': 1, 'block_amount': 3, 'features': 48, 'conv_type': 'sep_conv_3x3', 'downsample_type': 'area_d'},
    #                 {'encoder_unit_id': 2, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_5x5', 'downsample_type': 'area_d'},
    #                 {'encoder_unit_id': 3, 'block_amount': 2, 'features': 48, 'conv_type': 'conv_3x3', 'downsample_type': 'maxpool2d'},],
    #     "middle_units": [{'middle_unit_id': 0, 'block_amount': 2, 'features': 48, 'conv_type': 'dil_conv_3x3'}],
    #     "decoder_units": [
    #         {'decoder_unit_id': 0, 'block_amount': 1, 'features': 48, 'conv_type': 'sep_conv_3x3', 'upsample_type': 'deconv'},
    #         {'decoder_unit_id': 1, 'block_amount': 1, 'features': 48, 'conv_type': 'dil_conv_5x5', 'upsample_type': 'bilinear_u'},
    #         {'decoder_unit_id': 2, 'block_amount': 1, 'features': 48, 'conv_type': 'sep_conv_7x7', 'upsample_type': 'deconv'},
    #         {'decoder_unit_id': 3, 'block_amount': 3, 'features': 48, 'conv_type': 'sep_conv_5x5', 'upsample_type': 'sub_pixel_u'},]

    # }

    # dict = {

    #     "level_amount":4,
    #     "middle_unit_amount":1,

    #     "encoder_units":[ {'encoder_unit_id': 0, 'block_amount': 3, 'features': 48, 'conv_type': 'conv_3x3', 'downsample_type': 'convolution_d'},
    #                 {'encoder_unit_id': 1, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_3x3', 'downsample_type': 'nearest_d'},
    #                 {'encoder_unit_id': 2, 'block_amount': 2, 'features': 48, 'conv_type': 'sep_conv_3x3', 'downsample_type': 'nearest_d'},
    #                 {'encoder_unit_id': 3, 'block_amount': 2, 'features': 48, 'conv_type': 'conv_5x5', 'downsample_type': 'bilinear_d'},],
    #     "middle_units": [{'middle_unit_id': 0, 'block_amount': 2, 'features': 48, 'conv_type': 'conv_3x3'}],
    #     "decoder_units": [
    #         {'decoder_unit_id': 0, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_5x5', 'upsample_type': 'area_u'},
    #         {'decoder_unit_id': 1, 'block_amount': 3, 'features': 48, 'conv_type': 'sep_conv_7x7', 'upsample_type': 'sub_pixel_u'},
    #         {'decoder_unit_id': 2, 'block_amount': 1, 'features': 48, 'conv_type': 'sep_conv_5x5', 'upsample_type': 'area_u'},
    #         {'decoder_unit_id': 3, 'block_amount': 2, 'features': 48, 'conv_type': 'conv_3x3', 'upsample_type': 'sub_pixel_u'},]

    # }

    # dict = {
    #     "level_amount":5,
    #     "middle_unit_amount":1,
    #     "encoder_units":[ 
    #         {'encoder_unit_id': 0, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_3x3_leakyReLU', 'downsample_type': 'convolution_d'},
    #         {'encoder_unit_id': 1, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_3x3_leakyReLU', 'downsample_type': 'convolution_d'},
    #         {'encoder_unit_id': 2, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_3x3_leakyReLU', 'downsample_type': 'convolution_d'}, 
    #         {'encoder_unit_id': 3, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_3x3_leakyReLU', 'downsample_type': 'convolution_d'},
    #         {'encoder_unit_id': 4, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_3x3_leakyReLU', 'downsample_type': 'convolution_d'}],

    #     # "middle_units": [{'middle_unit_id': 0, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_3x3_leakyReLU'}],

    #     "decoder_units": [
    #         {'decoder_unit_id': 0, 'block_amount': 2, 'features': 96, 'conv_type': 'conv_3x3_leakyReLU', 'upsample_type': 'nearest_u'},
    #         {'decoder_unit_id': 1, 'block_amount': 2, 'features': 96, 'conv_type': 'conv_3x3_leakyReLU', 'upsample_type': 'nearest_u'},
    #         {'decoder_unit_id': 2, 'block_amount': 2, 'features': 96, 'conv_type': 'conv_3x3_leakyReLU', 'upsample_type': 'nearest_u'},
    #         {'decoder_unit_id': 3, 'block_amount': 2, 'features': 96, 'conv_type': 'conv_3x3_leakyReLU', 'upsample_type': 'nearest_u'},
    #         {'decoder_unit_id': 4, 'block_amount': 1, 'features': 96, 'conv_type': 'conv_3x3_leakyReLU', 'upsample_type': 'nearest_u'}]
    # }

    # dict = {
    #     "level_amount":3,
    #     "encoder_units":[ 
    #         {'encoder_unit_id': 0, 'block_amount': 2, 'features': 48, 'conv_type': 'conv_3x3_leakyReLU', 'downsample_type': 'convolution_d'},
    #         {'encoder_unit_id': 1, 'block_amount': 2, 'features': 48, 'conv_type': 'conv_3x3_leakyReLU', 'downsample_type': 'convolution_d'},
    #         {'encoder_unit_id': 2, 'block_amount': 2, 'features': 48, 'conv_type': 'conv_3x3_leakyReLU', 'downsample_type': 'convolution_d'}, 
    #        ],

    #     # "middle_units": [{'middle_unit_id': 0, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_3x3_leakyReLU'}],

    #     "decoder_units": [
    #         {'decoder_unit_id': 0, 'block_amount': 3, 'features': 96, 'conv_type': 'conv_3x3_leakyReLU', 'upsample_type': 'nearest_u'},
    #         {'decoder_unit_id': 1, 'block_amount': 3, 'features': 96, 'conv_type': 'conv_3x3_leakyReLU', 'upsample_type': 'nearest_u'},
    #         {'decoder_unit_id': 2, 'block_amount': 3, 'features': 96, 'conv_type': 'conv_3x3_leakyReLU', 'upsample_type': 'nearest_u'},
    #         ]
    # }

    # dict = {
    #     "level_amount":3,
    #     "encoder_units":[ 
    #         {'encoder_unit_id': 0, 'block_amount': 2, 'features': 48, 'conv_type': 'sep_conv_3x3', 'downsample_type': 'convolution_d'},
    #         {'encoder_unit_id': 1, 'block_amount': 2, 'features': 48, 'conv_type': 'sep_conv_3x3', 'downsample_type': 'convolution_d'},
    #         {'encoder_unit_id': 2, 'block_amount': 2, 'features': 48, 'conv_type': 'sep_conv_3x3', 'downsample_type': 'convolution_d'}, 
    #        ],

    #     # "middle_units": [{'middle_unit_id': 0, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_3x3_leakyReLU'}],

    #     "decoder_units": [
    #         {'decoder_unit_id': 0, 'block_amount': 3, 'features': 96, 'conv_type': 'sep_conv_3x3', 'upsample_type': 'nearest_u'},
    #         {'decoder_unit_id': 1, 'block_amount': 3, 'features': 96, 'conv_type': 'sep_conv_3x3', 'upsample_type': 'nearest_u'},
    #         {'decoder_unit_id': 2, 'block_amount': 3, 'features': 96, 'conv_type': 'sep_conv_3x3', 'upsample_type': 'nearest_u'},
    #         ]
    # }
    # flops:  47.96 GMac params:  262.51 k

    # dict = {
    #     "level_amount":3,
    #     "encoder_units":[ 
    #         {'encoder_unit_id': 0, 'block_amount': 2, 'features': 48, 'conv_type': 'dil_conv_3x3', 'downsample_type': 'convolution_d'},
    #         {'encoder_unit_id': 1, 'block_amount': 2, 'features': 48, 'conv_type': 'dil_conv_3x3', 'downsample_type': 'convolution_d'},
    #         {'encoder_unit_id': 2, 'block_amount': 2, 'features': 48, 'conv_type': 'dil_conv_3x3', 'downsample_type': 'convolution_d'}, 
    #        ],

    #     # "middle_units": [{'middle_unit_id': 0, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_3x3_leakyReLU'}],

    #     "decoder_units": [
    #         {'decoder_unit_id': 0, 'block_amount': 3, 'features': 96, 'conv_type': 'dil_conv_3x3', 'upsample_type': 'nearest_u'},
    #         {'decoder_unit_id': 1, 'block_amount': 3, 'features': 96, 'conv_type': 'dil_conv_3x3', 'upsample_type': 'nearest_u'},
    #         {'decoder_unit_id': 2, 'block_amount': 3, 'features': 96, 'conv_type': 'dil_conv_3x3', 'upsample_type': 'nearest_u'},
    #         ]
    # }
    # #flops:  47.96 GMac params:  262.51 k

    # dict = {
    #     "level_amount":3,
    #     "encoder_units":[ 
    #         {'encoder_unit_id': 0, 'block_amount': 2, 'features': 48, 'conv_type': 'conv_3x3', 'downsample_type': 'convolution_d'},
    #         {'encoder_unit_id': 1, 'block_amount': 2, 'features': 48, 'conv_type': 'conv_3x3', 'downsample_type': 'convolution_d'},
    #         {'encoder_unit_id': 2, 'block_amount': 2, 'features': 48, 'conv_type': 'conv_3x3', 'downsample_type': 'convolution_d'}, 
    #        ],

    #     # "middle_units": [{'middle_unit_id': 0, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_3x3_leakyReLU'}],

    #     "decoder_units": [
    #         {'decoder_unit_id': 0, 'block_amount': 3, 'features': 96, 'conv_type': 'conv_3x3', 'upsample_type': 'nearest_u'},
    #         {'decoder_unit_id': 1, 'block_amount': 3, 'features': 96, 'conv_type': 'conv_3x3', 'upsample_type': 'nearest_u'},
    #         {'decoder_unit_id': 2, 'block_amount': 3, 'features': 96, 'conv_type': 'conv_3x3', 'upsample_type': 'nearest_u'},
    #         ]
    # }
    # # flops:  72.69 GMac params:  1.06 M



    # dict = {
    #     "level_amount":4,
    #     "middle_unit_amount":1,
    #     "encoder_units":[ 
    #         {'encoder_unit_id': 0, 'block_amount': 2, 'features': 48, 'conv_type': 'conv_3x3_leakyReLU', 'downsample_type': 'maxpool2d'},
    #         {'encoder_unit_id': 1, 'block_amount': 2, 'features': 48, 'conv_type': 'conv_3x3_leakyReLU', 'downsample_type': 'maxpool2d'},
    #         {'encoder_unit_id': 2, 'block_amount': 2, 'features': 48, 'conv_type': 'conv_3x3_leakyReLU', 'downsample_type': 'maxpool2d'}, 
    #         {'encoder_unit_id': 3, 'block_amount': 2, 'features': 48, 'conv_type': 'conv_3x3_leakyReLU', 'downsample_type': 'maxpool2d'},],

    #     "middle_units": [{'middle_unit_id': 0, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_3x3'}],

    #     "decoder_units": [
    #         {'decoder_unit_id': 0, 'block_amount': 2, 'features': 48, 'conv_type': 'conv_3x3_leakyReLU', 'upsample_type': 'deconv'},
    #         {'decoder_unit_id': 1, 'block_amount': 2, 'features': 48, 'conv_type': 'conv_3x3_leakyReLU', 'upsample_type': 'deconv'},
    #         {'decoder_unit_id': 2, 'block_amount': 2, 'features': 48, 'conv_type': 'conv_3x3_leakyReLU', 'upsample_type': 'deconv'},
    #         {'decoder_unit_id': 3, 'block_amount': 2, 'features': 48, 'conv_type': 'conv_3x3_leakyReLU', 'upsample_type': 'deconv'},]
    # }

    # dict = {
    #     "level_amount":5,
    #     # "middle_unit_amount":1,
    #     "encoder_units":[ 
    #         {'encoder_unit_id': 0, 'block_amount': 1, 'features': 48, 'conv_type': 'sep_conv_3x3', 'downsample_type': 'convolution_d'},
    #         {'encoder_unit_id': 1, 'block_amount': 1, 'features': 48, 'conv_type': 'sep_conv_3x3', 'downsample_type': 'convolution_d'},
    #         {'encoder_unit_id': 2, 'block_amount': 1, 'features': 48, 'conv_type': 'sep_conv_3x3', 'downsample_type': 'convolution_d'}, 
    #         {'encoder_unit_id': 3, 'block_amount': 1, 'features': 48, 'conv_type': 'sep_conv_3x3', 'downsample_type': 'convolution_d'},
    #         {'encoder_unit_id': 4, 'block_amount': 1, 'features': 48, 'conv_type': 'sep_conv_3x3', 'downsample_type': 'convolution_d'}],

    #     # "middle_units": [{'middle_unit_id': 0, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_3x3_leakyReLU'}],

    #     "decoder_units": [
    #         {'decoder_unit_id': 0, 'block_amount': 2, 'features': 96, 'conv_type': 'sep_conv_3x3', 'upsample_type': 'nearest_u'},
    #         {'decoder_unit_id': 1, 'block_amount': 2, 'features': 96, 'conv_type': 'sep_conv_3x3', 'upsample_type': 'nearest_u'},
    #         {'decoder_unit_id': 2, 'block_amount': 2, 'features': 96, 'conv_type': 'sep_conv_3x3', 'upsample_type': 'nearest_u'},
    #         {'decoder_unit_id': 3, 'block_amount': 2, 'features': 96, 'conv_type': 'sep_conv_3x3', 'upsample_type': 'nearest_u'},
    #         {'decoder_unit_id': 4, 'block_amount': 1, 'features': 96, 'conv_type': 'sep_conv_3x3', 'upsample_type': 'nearest_u'}]
    # }
    # # flops:  46.99 GMac params:  330.34 k  dip_sure_consume_time :21.78665328025818

    # dict = {
    #     "level_amount":5,
    #     # "middle_unit_amount":1,
    #     "encoder_units":[ 
    #         {'encoder_unit_id': 0, 'block_amount': 1, 'features': 48, 'conv_type': 'dil_conv_3x3', 'downsample_type': 'convolution_d'},
    #         {'encoder_unit_id': 1, 'block_amount': 1, 'features': 48, 'conv_type': 'dil_conv_3x3', 'downsample_type': 'convolution_d'},
    #         {'encoder_unit_id': 2, 'block_amount': 1, 'features': 48, 'conv_type': 'dil_conv_3x3', 'downsample_type': 'convolution_d'}, 
    #         {'encoder_unit_id': 3, 'block_amount': 1, 'features': 48, 'conv_type': 'dil_conv_3x3', 'downsample_type': 'convolution_d'},
    #         {'encoder_unit_id': 4, 'block_amount': 1, 'features': 48, 'conv_type': 'dil_conv_3x3', 'downsample_type': 'convolution_d'}],

    #     # "middle_units": [{'middle_unit_id': 0, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_3x3_leakyReLU'}],

    #     "decoder_units": [
    #         {'decoder_unit_id': 0, 'block_amount': 2, 'features': 96, 'conv_type': 'dil_conv_3x3', 'upsample_type': 'nearest_u'},
    #         {'decoder_unit_id': 1, 'block_amount': 2, 'features': 96, 'conv_type': 'dil_conv_3x3', 'upsample_type': 'nearest_u'},
    #         {'decoder_unit_id': 2, 'block_amount': 2, 'features': 96, 'conv_type': 'dil_conv_3x3', 'upsample_type': 'nearest_u'},
    #         {'decoder_unit_id': 3, 'block_amount': 2, 'features': 96, 'conv_type': 'dil_conv_3x3', 'upsample_type': 'nearest_u'},
    #         {'decoder_unit_id': 4, 'block_amount': 1, 'features': 96, 'conv_type': 'dil_conv_3x3', 'upsample_type': 'nearest_u'}]
    # }

    # # 46.99 GMac params:  330.34 k
    dict = {
        "level_amount":5,
        # "middle_unit_amount":1,
        "encoder_units":[ 
            {'encoder_unit_id': 0, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_7x7', 'downsample_type': 'convolution_d'},
            {'encoder_unit_id': 1, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_7x7', 'downsample_type': 'convolution_d'},
            {'encoder_unit_id': 2, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_7x7', 'downsample_type': 'convolution_d'}, 
            {'encoder_unit_id': 3, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_7x7', 'downsample_type': 'convolution_d'},
            {'encoder_unit_id': 4, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_7x7', 'downsample_type': 'convolution_d'}],

        # "middle_units": [{'middle_unit_id': 0, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_3x3_leakyReLU'}],

        "decoder_units": [
            {'decoder_unit_id': 0, 'block_amount': 2, 'features': 96, 'conv_type': 'conv_7x7', 'upsample_type': 'nearest_u'},
            {'decoder_unit_id': 1, 'block_amount': 2, 'features': 96, 'conv_type': 'conv_7x7', 'upsample_type': 'nearest_u'},
            {'decoder_unit_id': 2, 'block_amount': 2, 'features': 96, 'conv_type': 'conv_7x7', 'upsample_type': 'nearest_u'},
            {'decoder_unit_id': 3, 'block_amount': 2, 'features': 96, 'conv_type': 'conv_7x7', 'upsample_type': 'nearest_u'},
            {'decoder_unit_id': 4, 'block_amount': 1, 'features': 96, 'conv_type': 'conv_7x7', 'upsample_type': 'nearest_u'}]
    }

    # dict = {
    #     "level_amount":5,
    #     "middle_unit_amount":1,
    #     "encoder_units":[ 
    #         {'encoder_unit_id': 0, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_3x3_leakyReLU', 'downsample_type': 'convolution_d'},
    #         {'encoder_unit_id': 1, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_3x3_leakyReLU', 'downsample_type': 'convolution_d'},
    #         {'encoder_unit_id': 2, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_3x3_leakyReLU', 'downsample_type': 'convolution_d'}, 
    #         {'encoder_unit_id': 3, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_3x3_leakyReLU', 'downsample_type': 'convolution_d'},
    #         {'encoder_unit_id': 4, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_3x3_leakyReLU', 'downsample_type': 'convolution_d'}],

    #     # "middle_units": [{'middle_unit_id': 0, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_3x3_leakyReLU'}],

    #     "decoder_units": [
    #         {'decoder_unit_id': 0, 'block_amount': 2, 'features': 96, 'conv_type': 'conv_3x3_leakyReLU', 'upsample_type': 'nearest_u'},
    #         {'decoder_unit_id': 1, 'block_amount': 2, 'features': 96, 'conv_type': 'conv_3x3_leakyReLU', 'upsample_type': 'nearest_u'},
    #         {'decoder_unit_id': 2, 'block_amount': 2, 'features': 96, 'conv_type': 'conv_3x3_leakyReLU', 'upsample_type': 'nearest_u'},
    #         {'decoder_unit_id': 3, 'block_amount': 2, 'features': 96, 'conv_type': 'conv_3x3_leakyReLU', 'upsample_type': 'nearest_u'},
    #         {'decoder_unit_id': 4, 'block_amount': 1, 'features': 96, 'conv_type': 'conv_3x3_leakyReLU', 'upsample_type': 'nearest_u'}]
    # }
    # flops:  84.32 GMac params:  1.19 M 95.57958316802979  dip_sure_consume_time: 30.318413496017456 dip_sure: max_psnr:29.544 eSURE_unifom: 29.855423, 30.065315 

    # dict = {
    #     "level_amount":5,
    #     "middle_unit_amount":1,
    #     "encoder_units":[ 
    #         {'encoder_unit_id': 0, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_3x3', 'downsample_type': 'convolution_d'},
    #         {'encoder_unit_id': 1, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_3x3', 'downsample_type': 'convolution_d'},
    #         {'encoder_unit_id': 2, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_3x3', 'downsample_type': 'convolution_d'}, 
    #         {'encoder_unit_id': 3, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_3x3', 'downsample_type': 'convolution_d'},
    #         {'encoder_unit_id': 4, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_3x3', 'downsample_type': 'convolution_d'}],

    #     # "middle_units": [{'middle_unit_id': 0, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_3x3_leakyReLU'}],

    #     "decoder_units": [
    #         {'decoder_unit_id': 0, 'block_amount': 2, 'features': 96, 'conv_type': 'conv_3x3', 'upsample_type': 'nearest_u'},
    #         {'decoder_unit_id': 1, 'block_amount': 2, 'features': 96, 'conv_type': 'conv_3x3', 'upsample_type': 'nearest_u'},
    #         {'decoder_unit_id': 2, 'block_amount': 2, 'features': 96, 'conv_type': 'conv_3x3', 'upsample_type': 'nearest_u'},
    #         {'decoder_unit_id': 3, 'block_amount': 2, 'features': 96, 'conv_type': 'conv_3x3', 'upsample_type': 'nearest_u'},
    #         {'decoder_unit_id': 4, 'block_amount': 1, 'features': 96, 'conv_type': 'conv_3x3', 'upsample_type': 'nearest_u'}]
    # }
    # # flops:  64.43 GMac params:  1.19 M    time_consume: 92.27867150306702

    # best_indi
    # dict = {
    #     'level_amount': 5,
    #     'encoder_units':[
    #         {'encoder_unit_id': 0, 'block_amount': 1, 'features': 48, 'conv_type': 'sep_conv_5x5', 'downsample_type': 'area_d'},
    #         {'encoder_unit_id': 1, 'block_amount': 1, 'features': 48, 'conv_type': 'sep_conv_5x5', 'downsample_type': 'area_d'},
    #         {'encoder_unit_id': 2, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_3x3', 'downsample_type': 'bilinear_d'},
    #         {'encoder_unit_id': 3, 'block_amount': 1, 'features': 48, 'conv_type': 'dil_conv_5x5', 'downsample_type': 'convolution_d'},
    #         {'encoder_unit_id': 4, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_5x5', 'downsample_type': 'convolution_d'}],
    #     'decoder_units':[
    #         {'decoder_unit_id': 0, 'block_amount': 2, 'features': 96, 'conv_type': 'conv_3x3', 'upsample_type': 'area_u'},
    #         {'decoder_unit_id': 1, 'block_amount': 2, 'features': 96, 'conv_type': 'sep_conv_5x5', 'upsample_type': 'bilinear_u'},
    #         {'decoder_unit_id': 2, 'block_amount': 2, 'features': 96, 'conv_type': 'conv_5x5', 'upsample_type': 'sub_pixel_u'},
    #         {'decoder_unit_id': 3, 'block_amount': 2, 'features': 96, 'conv_type': 'dil_conv_5x5', 'upsample_type': 'nearest_u'},
    #         {'decoder_unit_id': 4, 'block_amount': 1, 'features': 96, 'conv_type': 'dil_conv_5x5', 'upsample_type': 'bilinear_u'}]
    # }
    # # dip_sure:29.90


    best_indi_dict = {
    'level_amount': 5,
    'encoder_units':[
        {'encoder_unit_id': 0, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_3x3', 'downsample_type': 'area_d'},
        {'encoder_unit_id': 1, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_3x3_leakyReLU', 'downsample_type': 'bilinear_d'},
        {'encoder_unit_id': 2, 'block_amount': 1, 'features': 48, 'conv_type': 'dil_conv_5x5', 'downsample_type': 'bilinear_d'},
        {'encoder_unit_id': 3, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_7x7', 'downsample_type': 'nearest_d'},
        {'encoder_unit_id': 4, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_7x7', 'downsample_type': 'maxpool2d'},],
    'decoder_units':[
        {'decoder_unit_id': 0, 'block_amount': 2, 'features': 96, 'conv_type': 'conv_5x5', 'upsample_type': 'bilinear_u'},
        {'decoder_unit_id': 1, 'block_amount': 2, 'features': 96, 'conv_type': 'conv_5x5', 'upsample_type': 'bilinear_u'},
        {'decoder_unit_id': 2, 'block_amount': 2, 'features': 96, 'conv_type': 'conv_3x3', 'upsample_type': 'sub_pixel_u'},
        {'decoder_unit_id': 3, 'block_amount': 2, 'features': 96, 'conv_type': 'dil_conv_5x5', 'upsample_type': 'bilinear_u'},
        {'decoder_unit_id': 4, 'block_amount': 1, 'features': 96, 'conv_type': 'dil_conv_5x5', 'upsample_type': 'nearest_u'},],
    }

    indi.initialize_by_designed(dict)
    evaluate_indi =  EvaluateIndi(Config.train_params)
    # evaluate_indi.evaluate_indi_psnr(indi)
    indi2  = Individual(Config.population_params,"best_indi")
    indi2.initialize_by_designed(best_indi_dict)
    evaluate_indi.evaluate_indi_psnr(indi2)
    
    # inference(indi)

    
    # evaluate_indi =  EvaluateIndi(Config.train_params)

    # # evaluate_indi.evaluate_indi_psnr_test(indi)

    # evaluate_indi.evaluate_indi_psnr(indi)
    # print("indi.psnr",indi.psnr)