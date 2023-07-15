import copy

from genetic.individual import Individual
from setting.config import Config
class Population(object):
    def __init__(self, params, gen_no):
        """ population 参数初始化

        Args: 
            params(dict): 参数设置
            gen_id(int): 种群当前进化次数
        """
        self.gen_no = gen_no # 种群当前经历的进化代次
        self.indi_no = 0 # 种群中个体的数量
        self.params = params
        self.pop_size = params["pop_size"]
        self.individuals = []


    def initialize_by_elite(self, elite_indi_dicts):

        """
        精英初始化, 将已知较好的个体直接放入种群中。
        """
        elite_indi_len = len(elite_indi_dicts)
        for i in range(elite_indi_len):
            indi_id = "indi%02d%02d" % (self.gen_no, self.indi_no)
            indi = Individual(self.params, indi_id)
            indi.initialize_by_designed(elite_indi_dicts[i])
            self.indi_no += 1
            self.individuals.append(indi)
        
        for _ in range(elite_indi_len,self.pop_size):
            indi_id= "indi%02d%02d"%(self.gen_no,self.indi_no)
            indi = Individual(self.params,indi_id)
            indi.initialize()
            self.indi_no += 1
            self.individuals.append(indi)




    def initialize(self):
        """ 种群初始化
        """
        
        for _ in range(self.pop_size):
            indi_id= "indi%02d%02d"%(self.gen_no,self.indi_no)
            indi = Individual(self.params,indi_id)
            indi.initialize()
            self.indi_no += 1
            self.individuals.append(indi)
    
    def create_from_offspring(self, offsprings):
        for offspring in offsprings:
            indi = copy.deepcopy(offspring)
            indi_id = "indi%02d%02d"%(self.gen_no,self.indi_no)
            indi.indi_id = indi_id
            self.indi_no += 1
            self.individuals.append(indi)
    
    def __str__(self):
        _str =  []
        for ind in self.individuals:
            _str.append(str(ind)) 
            
            _str.append('-'*100)
        return '\n'.join(_str)


if __name__=="__main__":
    config = Config()
    population = Population(config.params,gen_id=0)
    population.initialize()
    print(population)






    

