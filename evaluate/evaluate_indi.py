import numpy as np
import logging
import time
import os
import sys

dir  = os.getcwd()
sys.path.append(dir)

from genetic.individual import Individual


# from evaluate.DnCNN.dataset import Dataset
# from evaluate.DnCNN.model import Unet
# from evaluate.DnCNN.train import net_train
from setting.config import Config
from dip_denoising.inference import inference
# from evaluate.dip_denoising.inference import inference

class EvaluateIndi(object):
    def __init__(self, train_params):
        self.train_params = train_params

    def evaluate_indi_psnr_test(self, indi):
        
        # for indi in self.indis:
        indi.psnr = np.random.uniform(25.0, 30.4)
    
    def evaluate_indi_psnr(self, indi):
        indi.psnr =  inference(indi)
        # indi.psnr = np.random.uniform(25.0, 30.4)
        # print("psnr:",indi.psnr)
        # indi.psnr = net_train(indi, self.train_params)
       

if __name__=="__main__":
    # Config.train_params
    indi = Individual(Config.population_params,0)
    dict = {
        "level_amount":2,
        "middle_unit_amount":1,
        "encoder_units":[ 
            {'encoder_unit_id': 0, 'block_amount': 3, 'features': 64, 'conv_type': 'conv_3x3', 'downsample_type': 'maxpool2d'},
            {'encoder_unit_id': 1, 'block_amount': 3, 'features': 64, 'conv_type': 'conv_3x3', 'downsample_type': 'maxpool2d'}, ],

        "middle_units": [{'middle_unit_id': 0, 'block_amount': 3, 'features': 64, 'conv_type': 'conv_3x3'}],

        "decoder_units": [
            {'decoder_unit_id': 0, 'block_amount': 3, 'features': 64, 'conv_type': 'conv_3x3', 'upsample_type': 'deconv'},
            {'decoder_unit_id': 1, 'block_amount': 3, 'features': 64, 'conv_type': 'conv_3x3', 'upsample_type': 'deconv'},]
    }
    indi.initialize_by_designed(dict)

    
    evaluate_indi =  EvaluateIndi(Config.train_params)

    # evaluate_indi.evaluate_indi_psnr_test(indi)

    evaluate_indi.evaluate_indi_psnr(indi)