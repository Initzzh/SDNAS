
import time
from genetic.population import Population
from setting.config import Config
from evaluate.evaluate_indi import EvaluateIndi
from genetic_op.crossover import Crossover
from genetic_op.mutation import Mutation
from log_utils.log import Log
# from utils.log import Log
population_params = Config.population_params
train_params = Config.train_params

# 1. 创建种群
# 2. 计算适应度
# 3. 进行交叉操作 crossover_p:
# 4. 进行变异 mutation_p:
# 5. 迭代演化

class Evolve(object):
    def __init__(self):
        # 创建种群
        self.population = Population(population_params, gen_no=0)
        self.evaluate_indi = EvaluateIndi(train_params)
        self.total_gen = population_params["gen"]
        self.logger = Log()

        self.record_path = population_params["log_path"]+"/evolve_6_25.log"

    
    def do_evlove(self):
        # 1.种群初始化
        # self.population.initialize()
        # 1.1 精英初始化
        start_time = time.time()
        dict = {
        "level_amount":5,
        "encoder_units":[ 
            {'encoder_unit_id': 0, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_3x3_leakyReLU', 'downsample_type': 'convolution_d'},
            {'encoder_unit_id': 1, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_3x3_leakyReLU', 'downsample_type': 'convolution_d'},
            {'encoder_unit_id': 2, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_3x3_leakyReLU', 'downsample_type': 'convolution_d'}, 
            {'encoder_unit_id': 3, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_3x3_leakyReLU', 'downsample_type': 'convolution_d'},
            {'encoder_unit_id': 4, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_3x3_leakyReLU', 'downsample_type': 'convolution_d'}],

       

        "decoder_units": [
            {'decoder_unit_id': 0, 'block_amount': 2, 'features': 96, 'conv_type': 'conv_3x3_leakyReLU', 'upsample_type': 'nearest_u'},
            {'decoder_unit_id': 1, 'block_amount': 2, 'features': 96, 'conv_type': 'conv_3x3_leakyReLU', 'upsample_type': 'nearest_u'},
            {'decoder_unit_id': 2, 'block_amount': 2, 'features': 96, 'conv_type': 'conv_3x3_leakyReLU', 'upsample_type': 'nearest_u'},
            {'decoder_unit_id': 3, 'block_amount': 2, 'features': 96, 'conv_type': 'conv_3x3_leakyReLU', 'upsample_type': 'nearest_u'},
            {'decoder_unit_id': 4, 'block_amount': 1, 'features': 96, 'conv_type': 'conv_3x3_leakyReLU', 'upsample_type': 'nearest_u'}]
        }
        dicts = [dict]
        self.population.initialize_by_elite(dicts)


        record = open(self.record_path, encoding="utf-8", mode='a' ) # 记录每代种群中最好的个体
        record.write("-"*50+str(time.strftime("%Y%m%d-%H%M%S"))+"-"*50+'\n')
        record.flush()
        time.sleep(1)
        self.logger.info("-"*50+str(time.strftime("%Y%m%d-%H%M%S"))+"-"*50+'\n')

        # log 
        for indi in self.population.individuals:
            self.logger.info(str(indi))

        # 2.计算种群适应度
        for indi in self.population.individuals:
            self.evaluate_indi.evaluate_indi_psnr(indi)
        
        # log, record 
        for indi in self.population.individuals:
            self.logger.info(str(indi))
            # record.write(str(indi))
            
            
        
        # for indi in self.population.individuals:
        #     self.logger.info(indi.psnr)
        #     self.logger.info(indi.psnr)
        #     # print(indi.psnr)
        best_indi = max(self.population.individuals, key = lambda indi: indi.psnr)
        self.logger.info("gen:{} best_indi:".format(0)+str(best_indi))
        record.write("gen:{} best_indi:".format(0)+str(best_indi))
        record.flush()
        time.sleep(1)
        # crossover =  Crossover()
        # muatation = Mutation()
        # 3. 种群进化
        

        for gen in range(self.total_gen):
            self.logger.info("-"*30+"第{}代进化".format(gen+1)+"-"*30)
            #  3.1 进行交叉
            # self.logger.info("-"*20+"[交叉]"+"-"*20)
            crossover =  Crossover(self.logger,self.population.individuals,gen+1)
            crossover.process()

            # for indi in self.population.individuals:
            #     self.logger.info(indi.psnr)
            # 3.2 进行变异
            # self.logger.info("-"*20+"[变异]"+"-"*20)
            mutation = Mutation(self.logger,self.population.individuals, gen+1)
            mutation.process()
            self.logger.info("="*35+"第{}代进化结果个体".format(gen+1)+"="*35)
            for indi in self.population.individuals:
                self.logger.info(indi)
            
            for indi in self.population.individuals:
                self.logger.info(indi.psnr)
            best_indi = max(self.population.individuals, key = lambda indi: indi.psnr)
            self.logger.info("="*20+"gen:{} best_indi:".format(gen+1)+"="*20+str(best_indi))
            record.write("gen:{} best_indi".format(gen+1)+str(best_indi))
            record.flush()
            time.sleep(1)
        consume_time = time.time()-start_time
        record.write("evolve consume_time:" + str(consume_time))
        record.close()



if __name__=="__main__":
    
    evolve = Evolve()
    print('-'*40+"[演化]"+'-'*40)
    evolve.do_evlove()
    print("-"*40+"[结束]"+"-"*40)


