

import numpy as np

import copy
import os
import sys
dir = os.getcwd()
sys.path.append(dir)

from genetic_op.selection import Selection
from genetic_op.doCrossover import doCrossover
from evaluate.evaluate_indi import EvaluateIndi
# from log import Log


from setting.config import Config

params = Config.population_params
train_params = Config.train_params
class Crossover(object):
    def __init__(self, log, indis, gen_epoch) :
        """ 交叉操作

        Arags:
            log(Log): 记录日志
            indis (list[Individual]): 种群中个体
            gen_epoch: 种群进化代次
        """
        self.log = log
        self.indis = indis
        self.gen_epoch = gen_epoch
        self.best_indi_num = params["best_indi_num"]
        self.crossover_prob = params["crossover_prob"]
    
    
    
    def evaluate_indi_psnr_test(self, indi):
        # for indi in self.indis:
        indi.psnr = np.random.uniform(29.0, 30.4)
    
    def process(self):
        #  遍历pop_size/2, 每次通过二元锦标赛选择2个个体作为parent,然后进行corssover
        # 1. 交叉
        # self.log.info("交叉前种群个体：")
        # for indi in self.indis:
        #     self.log.info(str(indi))
      
        self.log.info("-"*30+"[交叉]"+'-'*30)
        docrossover = doCrossover(self.indis, self.crossover_prob, self.log) 
        offsprings = docrossover.do_crossover()
         
        # for offspring in offsprings:
        #     print(offspring.psnr)

        # 计算子代适应度
        # TODO: 替换成正确的evaluate

        evaluate_indi =EvaluateIndi(train_params)
        # 对进行交叉操作后产生的子代进行适应度计算
        for indi in offsprings:

            if indi.psnr == -1.0:
                evaluate_indi.evaluate_indi_psnr(indi)
        
        self.log.info("-"*30+"offsprings"+"-"*30)
        for indi in offsprings:
            self.log.info(str(indi))
                
            # self.log.info(indi.psnr)
        
        # for offspring in offsprings:
        #     self.log.info(str(offspring))
        # log: 交叉后产生的子代的个体

        # 2. 加入到indis中并重新编号
        offsprings = offsprings + self.indis
        # log psnr
        self.log.info("-"*30+"offspring+indi"+"-"*30)
        for indi in offsprings:
            # self.evaluate_indi_psnr_test(indi)
            self.log.info(indi.psnr)

    
        new_offsprings = []
        indi_id = 0
        for offspring in offsprings:
            offspring.id = "indi%02d%02d" % (self.gen_epoch, indi_id)
            indi_id += 1
        
        # 3. 精英策略，将最好的n（5）个个体放入下一代中，通过二元锦标赛进行选择pop_size - n个个体
        for _ in range(self.best_indi_num):
            best_indi = max(offsprings, key = lambda indi: indi.psnr)
            new_offsprings.append(best_indi)
            offsprings.remove(best_indi)

        # 二元锦标赛选择pop_size - n 个个体
        selection = Selection()
        # new_offsprings += selection.binary_tournament_selection(offsprings, len(self.indis)-self.best_indi_num)
        new_offsprings += copy.deepcopy(selection.binary_tournament_selection(offsprings, len(self.indis)-self.best_indi_num))
        
        self.indis.clear() # 清空indis_list
        for offspring in new_offsprings:
            self.indis.append(offspring)

        self.log.info("-"*30 +"[交叉结束]"+'-'*30)

        # log
        self.log.info('-'*30+'交叉选择后的indi'+'-'*30)
        for indi in self.indis:
            self.log.info(str(indi))
        self.log.info('-'*30 + 'psnr_list'+'-'*30)
        for indi in self.indis:
            self.log.info(indi.psnr)
        # self.log.info("交叉后种群个体：")
        # for indi in self.indis:
        #     self.log.info(str(indi))
        

        # print(new_offsprings)
        # selection.binary_tournament_selection()
        # for _ in range(len(self.indis) - self.best_indi_num ):
            # selection.binary_tournament_selection(offsprings, 2) 

if __name__ =="__main__":
    from log_utils.log import Log
    from genetic.individual import Individual
    logger = Log()
    indis = []
    for i in range(0,30):
        indi = Individual(params,i)
        indi.initialize()
        indi.psnr = np.random.uniform(28.0, 30.40)
        indis.append(indi)
        print(indi.psnr)
        # print(indi)

    crossover = Crossover(logger, indis, 0)
    crossover.process()
    for indi in indis:
        print(indi.psnr)
    print("-----")
        

