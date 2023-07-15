import numpy as np
import copy 
import os
import sys
dir = os.getcwd()
sys.path.append(dir)
from genetic_op.doMutation import doMutation
from evaluate.evaluate_indi import EvaluateIndi

from setting.config import Config

params =  Config.population_params
train_params = Config.train_params

class Mutation(object):

    def __init__(self, log, indis, gen_epoch):
        """变异操作

        Args:
            log(Log): 记录变异操作
            indis (list[Individual])
            gen_epoch (int): 当前进化的代次
        """
        self.log =log
        self.indis = indis
        self.gen_epoch = gen_epoch
        self.select_num = params["select_num"]

        
    def binary_tournament_selection(self, indis, select_num):
        """ 二元锦标赛选择个体

        Args:
            indis (list[Individuals]): individuals,种群个体list
            select_num (int): 选择个体数量
        """

        selected_indis = []
        for _ in range(select_num):
            individuals = np.random.choice(indis,2,replace=False)
            if individuals[0].psnr > individuals[1].psnr:
                selected_indis.append(individuals[0])
            else:
                selected_indis.append(individuals[1])
        return selected_indis
    

    def evaluate_indi_psnr_test(self, indi):
        # for indi in self.indis:
        indi.psnr = np.random.uniform(29.0, 30.4)
        

    def process(self):
        np.random.seed(10)
        # self.log.info("变异前种群个体:")
        # for indi in self.indis:
        #     self.log.info(str(indi))
        # 1. 二元锦标赛选取select_num个个体
        selected_indis = self.binary_tournament_selection(self.indis, self.select_num)
        selected_indis = [copy.deepcopy(indi) for indi in selected_indis]
        
        # self.log.info("要变异的个体:")
        # for indi in selected_indis:
        #     self.log.info(str(indi))

        # 2. 对select_num个个体进行变异
        self.log.info("-"*30+"[变异]"+"-"*30)
        domutation = doMutation(self.log, selected_indis)
        self.offsprings = domutation.do_mutation() # 变异操作


        # 赋予临时id
        for indi in selected_indis:
            indi.id = "off_spring_{}".format(indi.id)
            indi.reset()

        # 3. 重新计算个体的适应度（后续可以计算MAC,FLOPS）
        evaluate_indi = EvaluateIndi(train_params)
        for offspring in self.offsprings:
            if offspring.psnr == -1.0:
                evaluate_indi.evaluate_indi_psnr(offspring)

            # self.evaluate_indi_psnr_test(offspring)
        
        # self.log.info("变异后的个体:")
        for indi in self.offsprings:
            self.log.info(str(indi))

        # for indi in self.indis:
        #     self.log.info(indi.psnr)

        # 4. 子代加入种群
        for offspring in self.offsprings:
            self.indis.append(offspring)
        
        self.log.info("-"*30+"[offspings+indis]"+'-'*30)
        for indi in self.indis:
            self.log.info(indi.psnr)

        # 所有个体重新编号：
        indi_id = 0
        for indi in self.indis:
            indi.id = "indi%02d%02d"%(self.gen_epoch,indi_id)
            indi_id += 1

        # 5. 删除select_num个psnr最差的个体
        for _ in range(self.select_num):
            worst_indi = min(self.indis, key = lambda indi: indi.psnr) 
            # print(worst_indi.psnr)
            self.indis.remove(worst_indi)
        self.log.info("-"*30+"[变异结束]"+"-"*30)
        self.log.info("-"*30+"[变异后indi]"+"-"*30)
        for indi in self.indis:
            self.log.info(str(indi))
        for indi in self.indis:
            self.log.info(indi.psnr)
            # self.log.info(indi.psnr)
        # self.log.info("变异后的种群个体")
        # for indi in self.indis:
        #     self.log.info(indi)
        # self.indis.sort(key = lambda indi: indi.psnr, reverse=True)
        # for _ in range(self.select_num):
        #     self.indis.pop(-1) # 删除末尾元素

        del domutation

if __name__=="__main__":
    from utils.log import Log
    from genetic.individual import Individual
    from setting.config import Config
    params = Config.population_params
    logger = Log()
    indis = []
    for i in range(0,30):
        indi = Individual(params, i)
        indi.initialize()
        indi.psnr = np.random.uniform(28.0, 30.40)
        indis.append(indi)
        print(indi)
    
    mutation = Mutation(logger, indis, 0)
    mutation.process()
    
        



        



    
