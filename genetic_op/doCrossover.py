import numpy as np
import copy

from genetic_op.selection import Selection
class doCrossover(object):
    def __init__(self, indis, crossover_prob,log):
        """交叉操作初始化

        Arags:
            parents (list(Individual), length:2): 2个个体，作为交叉的父代
        """

        self.indis = indis
        self.log = log
        self.indis_length = len(indis)
        self.crossover_prob = crossover_prob
        
        # self.level_amount = indis[0].level_amount
        # self.parent1_level_amount = parents[0]
        # self.parent2_level_amount = parents[1]

    def change_unit(self, parents, unit_pos, unit_type):
        # change

        # unit_type: 0: encoder_unit,1: decoder_unit
        self.log.info("{}与{}进行交叉".format(parents[0].id, parents[1].id))

        # encoder_unit
        if unit_type == 0:
            # block_amount 交叉
            self.log.info("encoder_unit:{}进行互换".format(unit_pos))
            if parents[0].encoder_units[unit_pos].block_amount != parents[1].encoder_units[unit_pos].block_amount:
                self.log.info("变换前两个体的 block_amount: {},{}".format(parents[0].encoder_units[unit_pos].block_amount,parents[1].encoder_units[unit_pos].block_amount ))
                tmp_block_amount = parents[0].encoder_units[unit_pos].block_amount 
                parents[0].encoder_units[unit_pos].block_amount = parents[1].encoder_units[unit_pos].block_amount
                parents[1].encoder_units[unit_pos].block_amount = tmp_block_amount
                self.log.info("变换后两个体的 block_amount: {},{}".format(parents[0].encoder_units[unit_pos].block_amount,parents[1].encoder_units[unit_pos].block_amount ))
            # features 交叉
            if parents[0].encoder_units[unit_pos].features != parents[1].encoder_units[unit_pos].features:
                self.log.info("变换前两个体的 features: {},{}".format(parents[0].encoder_units[unit_pos].features,parents[1].encoder_units[unit_pos].features ))
                tmp_features = parents[0].encoder_units[unit_pos].features
                parents[0].encoder_units[unit_pos].features = parents[1].encoder_units[unit_pos].features
                parents[1].encoder_units[unit_pos].features = tmp_features
                self.log.info("变换后两个体的 features: {},{}".format(parents[0].encoder_units[unit_pos].features,parents[1].encoder_units[unit_pos].features ))
            
            # conv_type 交叉
            if parents[0].encoder_units[unit_pos].conv_type != parents[1].encoder_units[unit_pos].conv_type:
                self.log.info("变换前两个体的 conv_type: {},{}".format(parents[0].encoder_units[unit_pos].conv_type,parents[1].encoder_units[unit_pos].conv_type ))
                tmp_conv_type = parents[0].encoder_units[unit_pos].conv_type
                parents[0].encoder_units[unit_pos].conv_type = parents[1].encoder_units[unit_pos].conv_type
                parents[1].encoder_units[unit_pos].conv_type = tmp_conv_type
                self.log.info("变换后两个体的 conv_type: {},{}".format(parents[0].encoder_units[unit_pos].conv_type,parents[1].encoder_units[unit_pos].conv_type ))
            # downsample_type 交叉
            if parents[0].encoder_units[unit_pos].downsample_type != parents[1].encoder_units[unit_pos].downsample_type:
                self.log.info("变换前两个体的 downsample_type: {},{}".format(parents[0].encoder_units[unit_pos].downsample_type, parents[1].encoder_units[unit_pos].downsample_type ))
                tmp_downsample_type = parents[0].encoder_units[unit_pos].downsample_type
                parents[0].encoder_units[unit_pos].downsample_type = parents[1].encoder_units[unit_pos].downsample_type
                parents[1].encoder_units[unit_pos].downsample_type = tmp_downsample_type
                self.log.info("变换后两个体的 downsample_type: {},{}".format(parents[0].encoder_units[unit_pos].downsample_type, parents[1].encoder_units[unit_pos].downsample_type ))
        # decoder_unit
        elif unit_type==1:
            # block_amount 交叉
            self.log.info("decoder_unit:{}进行互换".format(unit_pos))
            if parents[0].decoder_units[unit_pos].block_amount != parents[1].decoder_units[unit_pos].block_amount:
                self.log.info("变换前两个体的 block_amount: {},{}".format(parents[0].decoder_units[unit_pos].block_amount,parents[1].decoder_units[unit_pos].block_amount ))
                tmp_block_amount = parents[0].decoder_units[unit_pos].block_amount 
                parents[0].decoder_units[unit_pos].block_amount = parents[1].decoder_units[unit_pos].block_amount
                parents[1].decoder_units[unit_pos].block_amount = tmp_block_amount
                self.log.info("变换后两个体的 block_amount: {},{}".format(parents[0].decoder_units[unit_pos].block_amount,parents[1].decoder_units[unit_pos].block_amount ))
            # features 交叉
            if parents[0].decoder_units[unit_pos].features != parents[1].decoder_units[unit_pos].features:
                self.log.info("变换前两个体的 features: {},{}".format(parents[0].decoder_units[unit_pos].features,parents[1].decoder_units[unit_pos].features ))
                tmp_features = parents[0].decoder_units[unit_pos].features
                parents[0].decoder_units[unit_pos].features = parents[1].decoder_units[unit_pos].features
                parents[1].decoder_units[unit_pos].features = tmp_features
                self.log.info("变换后两个体的 features: {},{}".format(parents[0].decoder_units[unit_pos].features,parents[1].decoder_units[unit_pos].features ))
            
            # conv_type 交叉
            if parents[0].decoder_units[unit_pos].conv_type != parents[1].decoder_units[unit_pos].conv_type:
                self.log.info("变换前两个体的 conv_type: {},{}".format(parents[0].decoder_units[unit_pos].conv_type,parents[1].decoder_units[unit_pos].conv_type ))
                tmp_conv_type = parents[0].decoder_units[unit_pos].conv_type
                parents[0].decoder_units[unit_pos].conv_type = parents[1].decoder_units[unit_pos].conv_type
                parents[1].decoder_units[unit_pos].conv_type = tmp_conv_type
                self.log.info("变换后两个体的 conv_type: {},{}".format(parents[0].decoder_units[unit_pos].conv_type,parents[1].decoder_units[unit_pos].conv_type ))
            # upsample_type 交叉
            if parents[0].decoder_units[unit_pos].upsample_type != parents[1].decoder_units[unit_pos].upsample_type:
                self.log.info("变换前两个体的 upsample: {},{}".format(parents[0].decoder_units[unit_pos].upsample_type, parents[1].decoder_units[unit_pos].upsample_type ))
                tmp_upsample_type = parents[0].decoder_units[unit_pos].upsample_type
                parents[0].decoder_units[unit_pos].upsample_type = parents[1].decoder_units[unit_pos].upsample_type
                parents[1].decoder_units[unit_pos].upsample_type = tmp_upsample_type
                self.log.info("变换后两个体的 upsample: {},{}".format(parents[0].decoder_units[unit_pos].upsample_type, parents[1].decoder_units[unit_pos].upsample_type ))
        
        # middle_unit
        elif unit_type ==2:
            # block_amount 交叉
            self.log.info("middle_unit:{}进行互换".format(unit_pos))
            if parents[0].middle_units[unit_pos].block_amount != parents[1].middle_units[unit_pos].block_amount:
                self.log.info("变换前两个体的 block_amount: {},{}".format(parents[0].middle_units[unit_pos].block_amount,parents[1].middle_units[unit_pos].block_amount ))
                tmp_block_amount = parents[0].middle_units[unit_pos].block_amount 
                parents[0].middle_units[unit_pos].block_amount = parents[1].middle_units[unit_pos].block_amount
                parents[1].middle_units[unit_pos].block_amount = tmp_block_amount
                self.log.info("变换后两个体的 block_amount: {},{}".format(parents[0].middle_units[unit_pos].block_amount,parents[1].middle_units[unit_pos].block_amount ))
            # features 交叉
            if parents[0].middle_units[unit_pos].features != parents[1].middle_units[unit_pos].features:
                self.log.info("变换前两个体的 features: {},{}".format(parents[0].middle_units[unit_pos].features,parents[1].middle_units[unit_pos].features ))
                tmp_features = parents[0].middle_units[unit_pos].features
                parents[0].middle_units[unit_pos].features = parents[1].middle_units[unit_pos].features
                parents[1].middle_units[unit_pos].features = tmp_features
                self.log.info("变换后两个体的 features: {},{}".format(parents[0].middle_units[unit_pos].features,parents[1].middle_units[unit_pos].features ))
            
            # conv_type 交叉
            if parents[0].middle_units[unit_pos].conv_type != parents[1].middle_units[unit_pos].conv_type:
                self.log.info("变换前两个体的 conv_type: {},{}".format(parents[0].middle_units[unit_pos].conv_type,parents[1].middle_units[unit_pos].conv_type ))
                tmp_conv_type = parents[0].middle_units[unit_pos].conv_type
                parents[0].middle_units[unit_pos].conv_type = parents[1].middle_units[unit_pos].conv_type
                parents[1].middle_units[unit_pos].conv_type = tmp_conv_type
                self.log.info("变换后两个体的 conv_type: {},{}".format(parents[0].middle_units[unit_pos].conv_type,parents[1].middle_units[unit_pos].conv_type ))

        # psnr 置初始值
        parents[0].psnr = -1.0
        parents[1].psnr = -1.0
    
    def get_unit_pos(self,parents):
        """随机产生符合要求的unit位置

        Arags:
            parents (list[Individual], len:2):  2个父代个体
            unit_type (int 0/1): unit类型 0:encoder_unit , 1:decoder_unit
        Returns: 
            unit_pos (int): 返回产生合理的单元位置

        """

        unequal_pos = []

        # encoder中unit不同的unit_pos
        for unit_pos in range(parents[0].level_amount):
            if self.unit_is_equal(parents[0].encoder_units[unit_pos], parents[1].encoder_units[unit_pos], unit_type=0) != True:
                unequal_pos.append(unit_pos)
        for unit_pos in range(parents[0].level_amount):
            if self.unit_is_equal(parents[0].decoder_units[unit_pos],parents[1].decoder_units[unit_pos], unit_type=1) != True:
                unequal_pos.append(parents[0].level_amount + unit_pos)
        
        # for unit_pos in range(parents[0].middle_unit_amount):
        #     if self.unit_is_equal(parents[0].middle_units[unit_pos], parents[1].middle_units[unit_pos], unit_type=2) !=True:
        #         unequal_pos.append(parents[0].level_amount+unit_pos)
        # for unit_pos in range(parents[0].level_amount):
        #     if self.unit_is_equal(parents[0].decoder_units[unit_pos],parents[1].decoder_units[unit_pos], unit_type=1) != True:
        #         unequal_pos.append(parents[0].level_amount+ parents[0].middle_unit_amount+ unit_pos)
        
        unit_pos = np.random.choice(unequal_pos)
        unit_type = 0
        # unit_pos 对应的是encoder_unit
        if unit_pos < parents[0].level_amount:
            unit_type = 0
            
        # # unit_pos 对应的是middle_unit
        # elif unit_pos < parents[0].level_amount+parents[0].middle_unit_amount:
        #     unit_type = 2
        #     unit_pos = unit_pos- parents[0].level_amount
        # # unit_pos 对应的是decoder_unit
        else:
            unit_type = 1
            # unit_pos =unit_pos- parents[0].level_amount- parents[0].middle_unit_amount
            unit_pos = unit_pos-parents[0].level_amount
        
        return unit_pos, unit_type


        

        # # 这是parents的level_amount相同
        # unit_pos = np.random.randint(0, parents[0].level_amount)
        # if unit_type == 0:
        #     while self.unit_is_equal(parents[0].encoder_units[unit_pos], parents[1].encoder_units[unit_pos], unit_type):
        #         unit_pos = np.random.randint(0,parents[0].level_amount)
        # elif unit_type == 1:
        #     while self.unit_is_equal(parents[0].decoder_units[unit_pos], parents[1].decoder_units[unit_pos], unit_type):
        #         unit_pos = np.random.randint(0, parents[0].level_amount)
        # return unit_pos


    def indi_is_equal(self,indi1,indi2):
        is_equal = True
        if indi1.level_amount != indi2.level_amount:
            is_equal = False
        else:
            for unit_index in range(indi1.level_amount):
                is_equal = self.unit_is_equal(indi1.encoder_units[unit_index], indi2.encoder_units[unit_index], unit_type=0)
                if is_equal == False:
                    break
                is_equal = self.unit_is_equal(indi1.decoder_units[unit_index], indi2.decoder_units[unit_index], unit_type=1)
                if is_equal == False:
                    break
        return is_equal
                
    def unit_is_equal(self, unit1, unit2 , unit_type  ):
        """判断两个unit 是否一样

        Arags:
            unit1 (EncoderUnit/DecoderUnit) : 单元块
            unit2 (EncoderUnit/DecoderUnit) : 单元块
            unit_type (int 0,1) : 单元块类型, Encoder还是Decoder
        
        Returns:
            is_equal (bool): 返回unit1,unit2是否相同
        """
        
        is_equal = True
        if unit1.block_amount != unit2.block_amount:
            is_equal = False
        elif unit1.features != unit2.features:
            is_equal = False
        elif unit1.conv_type != unit2.conv_type:
            is_equal = False
        else:
            if unit_type ==0:
                if unit1.downsample_type != unit2.downsample_type:
                    is_equal = False
            elif unit_type == 1:
                if unit1.upsample_type != unit2.upsample_type:
                    is_equal = False
        return is_equal
    



    def do_crossover(self):
        selection = Selection()
        offsprings = []
        for crossover_index in range(self.indis_length//2):
            
            # 选择2个父代个体
            self.log.info("-"*25+"第{}次进行交叉".format(crossover_index)+"-"*25)
            # 随机选择个体,判断个体结构是否相同，相同则再继续随机选择2个个体。
            # 这里假定level_amount都相同,
            # TODO：level_amount 不同情况代处理
            parents = selection.binary_tournament_selection(self.indis, 2)
            while self.indi_is_equal(parents[0],parents[1]):
                parents = selection.binary_tournament_selection(self.indis, 2)
            parents = copy.deepcopy(parents)
            # print("parents\n",parents[0],"\n",parents[1])
            
            # 在满足交叉率时，进行交叉
            if np.random.rand() < self.crossover_prob:
                # self.log.info("进行交叉，子代通过父代交叉得到")
                # # # 判断parents的level_amount是否一致
                # # corssover_unit_pos = np.random.randint(0, self.level_amount)
                # # while self.unit_is_equal()
                # # corssover_unit_type = np.random.randint(0, 2)
                # crossover_unit_type = np.random.randint(0, 2)
                # # 判断encoder_units中unit是否都相同, 若都行同，则交叉点在decoder_units中
                
                # if crossover_unit_type == 0:
                #     flag = False # 
                #     for unit_index in range(parents[0].level_amount):
                #         if self.unit_is_equal(parents[0].encoder_units[unit_index],parents[1].encoder_units[unit_index], unit_type=0) == False:
                #             # unit不同
                #             flag = True
                #             break
                #     if flag == False:
                #         crossover_unit_type = 1

                # # 判断encoder_units中unit是否都相同, 若都想同，则交叉点在decoder_units中
                # elif crossover_unit_type == 1:
                #     flag = False
                #     for unit_index in range(parents[0].level_amount):
                #         if self.unit_is_equal(parents[0].decoder_units[unit_index], parents[1].decoder_units[unit_index], unit_type=1) == False:
                #             flag = True
                #     if flag == False:
                #         crossover_unit_type = 0
                crossover_unit_pos, crossover_unit_type = self.get_unit_pos(parents)
                # crossover_unit_pos = self.get_unit_pos(parents,crossover_unit_type)
                # while self.unit_is_equal(parents[0].)
                self.change_unit(parents,crossover_unit_pos, crossover_unit_type)
                offsprings += parents
            else:
                self.log.info("未进行交叉，子代直接复制父代")
            # print("parents\n",parents[0],"\n",parents[1])
            # offsprings += parents
        return offsprings
        

