from setting.config import Config

import numpy as np
params = Config.population_params 

class doMutation(object):
    """个体之间进行变异
    
    """
    def __init__(self, log, indis):
        """初始化

        Arags:
            inds (list[Individuals]) : 种群中的个体
        
        """
        self.offsprings = indis
        self.log = log
        self.conv_types = params["conv_types"]
        self.downsample_types = params["downsample_types"]
        self.upsample_types = params["upsample_types"]
        self.conv_types_index = list(range(len(self.conv_types)))
        self.downsample_types_index = list(range(len(self.downsample_types)))
        self.upsample_types_index = list(range(len(self.upsample_types)))
        
    def get_new_muation_unit_pos(self, old_unit_pos, indi):
        """ 重新获取待变异block的unit位置

        Arags:
            old_unit_pos (int): 旧的unit位置
            indi (Individual): 待变异的个体
        
        Returns:
            [int]: 新的unit 位置
        
        """
        new_pos = np.random.randint(0,indi.level_amount)
        while(old_unit_pos == new_pos):
            new_pos = np.random.randint(0, indi.level_amount)
        return new_pos

    def alter_unit(self, indi, unit_pos, unit_type):
        # TODO: 随机feature
        if unit_type == 0:
            # TODO: 随机feature
            # TODO: 随机conv类型
            self.log.info("修改第{}个encoder_unit".format(unit_pos))
            cur_conv_type_index = self.conv_types.index(indi.encoder_units[unit_pos].conv_type)
            tmp_conv_types_index  = [ val for val in self.conv_types_index if val != cur_conv_type_index]
            new_conv_type_index = np.random.choice(tmp_conv_types_index) 
            indi.encoder_units[unit_pos].conv_type = self.conv_types[new_conv_type_index]
            self.log.info("conv_type变化{}-->{}".format(self.conv_types[cur_conv_type_index],self.conv_types[new_conv_type_index]))
            
            # TODO: 随机downsample 类型
            cur_downsample_type_index = self.downsample_types.index(indi.encoder_units[unit_pos].downsample_type)
            tmp_downsample_types_index = [val for val in self.downsample_types_index if val != cur_downsample_type_index]
            new_downsample_type_index = np.random.choice(tmp_downsample_types_index)
            indi.encoder_units[unit_pos].downsmple_type = self.downsample_types[new_downsample_type_index]
            self.log.info("downsample_type变化{}-->{}".format(self.downsample_types[cur_downsample_type_index],self.downsample_types[new_downsample_type_index]))
            

        elif unit_type == 1:

            self.log.info("修改第{}个decoder_unit".format(unit_pos))
            # TODO: 随机feature
            # TODO: 随机conv类型
            
            cur_conv_type_index = self.conv_types.index(indi.decoder_units[unit_pos].conv_type)
            tmp_conv_types_index  = [ val for val in self.conv_types_index if val != cur_conv_type_index]
            new_conv_type_index = np.random.choice(tmp_conv_types_index) 
            indi.decoder_units[unit_pos].conv_type = self.conv_types[new_conv_type_index]
            self.log.info("conv_type变化:{}-->{}".format(self.conv_types[cur_conv_type_index],self.conv_types[new_conv_type_index]))
            
            # TODO: 随机downsample 类型
            cur_upsample_type_index = self.upsample_types.index(indi.decoder_units[unit_pos].upsample_type)
            tmp_upsample_types_index = [val for val in self.upsample_types_index if val != cur_upsample_type_index]
            new_upsample_type_index = np.random.choice(tmp_upsample_types_index)
            indi.decoder_units[unit_pos].upsmple_type = self.upsample_types[new_upsample_type_index]
            self.log.info("upsample_type变化:{}-->{}".format(self.upsample_types[cur_upsample_type_index],self.upsample_types[new_upsample_type_index]))

        indi.psnr = -1.0
    def add_block(self,indi, unit_pos, unit_type):
        if unit_type == 0:
            self.log.info("增加第{}个encoder_unit中的block数量".format(unit_pos))
            indi.encoder_units[unit_pos].block_amount += 1
            self.log.info("block_amount变化:{}-->{}".format(indi.encoder_units[unit_pos].block_amount-1, indi.encoder_units[unit_pos].block_amount))
        elif unit_type == 1:
            self.log.info("增加第{}个decoder_unit中的block数量".format(unit_pos))
            indi.decoder_units[unit_pos].block_amount += 1
            self.log.info("block_amount变化:{}-->{}".format(indi.decoder_units[unit_pos].block_amount-1, indi.decoder_units[unit_pos].block_amount))
        elif unit_type == 2:
            self.log.info("增加第{}个middle_unit中的block数量".format(unit_pos))
            indi.middle_units[unit_pos].block_amount += 1
            self.log.info("block_amount变化:{}-->{}".format(indi.middle_units[unit_pos].block_amount-1, indi.middle_units[unit_pos].block_amount))
        indi.psnr = -1.0

    def remove_block(self,indi, unit_pos, unit_type):
        if unit_type == 0:
            self.log.info("减少第{}个encoder_unit中的block数量".format(unit_pos))
            indi.encoder_units[unit_pos].block_amount -= 1
            self.log.info("block_amount变化:{}-->{}".format(indi.encoder_units[unit_pos].block_amount+1, indi.encoder_units[unit_pos].block_amount))
        elif unit_type == 1:
            self.log.info("减少第{}个decoder_unit中的block数量".format(unit_pos))
            indi.decoder_units[unit_pos].block_amount -= 1
            self.log.info("block_amount变化:{}-->{}".format(indi.decoder_units[unit_pos].block_amount+1, indi.decoder_units[unit_pos].block_amount))
        elif unit_type == 2:
            self.log.info("增加第{}个middle_unit中的block数量".format(unit_pos))
            indi.middle_units[unit_pos].block_amount -= 1
            self.log.info("block_amount变化:{}-->{}".format(indi.middle_units[unit_pos].block_amount+1, indi.middle_units[unit_pos].block_amount))

        indi.psnr = -1.0

    def get_unit_pos(self,indi):
        unit_pos= np.random.randint(0,indi.level_amount+indi.level_amount)
        if unit_pos < indi.level_amount:
            unit_type = 0
        # elif unit_pos < indi.level_amount+indi.middle_unit_amount:
        #     unit_type = 2
        #     unit_pos = unit_pos- indi.level_amount
        else:
            unit_type = 1
            unit_pos = unit_pos - indi.level_amount
        return unit_pos, unit_type

        

    def do_mutation(self):
        """进行变异操作

        Returns:
            offsprings list[Individuals]:返回生成的个体
        """
        # self.log.info("开始变异")
        
        for index, offspring in enumerate(self.offsprings):
            self.log.info("-"*25+"{}开始变异".format(offspring.id)+"-"*25)
            mutation_op = np.random.randint(-1, 2) # 变异操作： -1：删除block; 0: 修改block参数； 1：增加block
            mutation_op = 0
            # 随机选择变异的block位置
            
            # mutation_unit_type = np.random.randint(0,2) # 0: 在encoder上进行变异，1: 在decoder上进行变异
            # mutation_unit_pos = np.random.randint(0,offspring.level_amount) # 要变异的unit位置

            mutation_unit_pos, mutation_unit_type = self.get_unit_pos(offspring)

            if mutation_op == 1: 
                valid = True # 选择的unit的block_amount 达到最大，则随机选择（删除，修改block参数）
                if mutation_unit_type ==0 :
                    if offspring.encoder_units[mutation_unit_pos].block_amount == offspring.max_block_amount:
                        valid = False
                elif mutation_unit_type == 1:
                    if offspring.decoder_units[mutation_unit_pos].block_amount == offspring.max_block_amount:
                        valid = False
                # else:
                #     if offspring.middle_units[mutation_unit_pos].block_amount == offspring.max_block_amount:
                #         valid = False
                # 满足添加block的条件
                if valid:
                    self.add_block(offspring, mutation_unit_pos, mutation_unit_type)
                # 不满足条件，则随机选择删除操作，修改操作
                else:
                    mutation_op = np.random.randint(-1,1)
                    if mutation_op == -1:
                        self.remove_block(offspring,mutation_unit_pos, mutation_unit_type)
                    elif mutation_op == 0:
                        self.alter_unit(offspring, mutation_unit_pos, mutation_unit_type)
            
            elif mutation_op == -1:
                valid = True # 选择的unit的block_amount 达到最小，则随机选择（删除，修改block参数）
                if mutation_unit_type ==0 :
                    if offspring.encoder_units[mutation_unit_pos].block_amount == offspring.min_block_amount:
                        valid = False
                elif mutation_unit_type == 1:
                    if offspring.decoder_units[mutation_unit_pos].block_amount == offspring.min_block_amount:
                        valid = False
                # else:
                #     if offspring.middle_units[mutation_unit_pos].block_amount == offspring.min_block_amount:
                #         valid = False
                
                if valid:
                    self.remove_block(offspring, mutation_unit_pos, mutation_unit_type)
                
                # 不满足条件， 则随机选择添加操作，修改操作
                else:
                    mutation_op = np.random.randint(0,2)
                    if mutation_op == 0:
                        self.alter_unit(offspring, mutation_unit_pos, mutation_unit_type)
                    elif mutation_op == 1:
                        self.add_block(offspring, mutation_unit_pos, mutation_unit_type)
                    
            elif mutation_op == 0:
                self.alter_unit(offspring, mutation_unit_pos, mutation_unit_type)




            # # 添加block
            # if mutation_op==1:
            #     # 如果当前待增加的unit中的block数量已达到上限，则选择其他unit添加，如果均达到上限，则选择其他两种变异操作
            #     valid = False
            #     if mutation_unit_type == 0:
            #         # 判断encoder_units中所有unit的block是否都达到上限，如果达到了，则在decoder_units中进行添加操作,如果encoder_units,decoder_units的block都达到上限，则换变异操作
            #         for encoder_unit in offspring.encoder_units:
            #             if encoder_unit.block_amount < offspring.max_block_amount:
            #                 valid = True
            #                 break
            #         if valid == False:
            #             for decoder_unit in offspring.decoder_units:
            #                 if decoder_unit.block_amount < offspring.max_block_amount:
            #                     mutation_unit_type = 1
            #                     valid = True
            #                     break
            #     elif mutation_unit_type == 1:
            #         for decoder_unit in offspring.decoder_units:
            #             if decoder_unit.block_amount < offspring.max_block_amount:
            #                 valid = True
            #                 break
            #         if valid == False:
            #             for encoder_unit in offspring.encoder_units:
            #                 if encoder_unit.block_amount < offspring.max_block_amount:
            #                     mutation_unit_type = 0
            #                     valid = True
            #                     break

            #     if valid:
            #         # encoder 中进行变异
            #         if mutation_unit_type == 0:
            #             # 当unit的block达到了上限，重新随机选择unit_pos 进行增加block数量
            #             while offspring.encoder_units[mutation_unit_pos].block_amount ==  offspring.max_block_amount:
            #                 mutation_unit_pos = self.get_new_muation_unit_pos(mutation_unit_pos, offspring)

            #             # offspring.encoder_units[mutation_unit_pos].block_amount += 1 # block数量增加1
            #             self.add_block(offspring, mutation_unit_pos, mutation_unit_type)
                     
            #         elif mutation_unit_type == 1:
            #             while offspring.decoder_units[mutation_unit_pos].block_amount == offspring.max_block_amount:
            #                 mutation_unit_pos = self.get_new_muation_unit_pos(mutation_unit_pos, offspring)
            #             # offspring.decoder_units[mutation_unit_pos].block_amount +=1 
            #             self.add_block(offspring, mutation_unit_pos,mutation_unit_type)
            #     else:
            #         mutation_op = np.random.randint(-1,1) # 删除block/ 修改block
            #         mutation_unit_type = np.random.randint(0,1)  # 随机变异的unit 类型/ encoder,decoder
            #         mutation_unit_pos = np.random.randint(0,offspring.level_amount) # unit 位置
            #         # 删除block
            #         if mutation_op == -1:
            #             if mutation_unit_type == 0:
            #                 self.remove_block(offspring, mutation_unit_pos, mutation_unit_type)
            #                 # offspring.encoder_units[mutation_unit_pos].block_amount -= 1
            #             elif mutation_unit_type == 1:
            #                 self.remove_block(offspring, mutation_unit_pos, mutation_unit_type)
            #                 # offspring.decoder_units[mutation_unit_pos].block_amount -= 1
            #         # 修改block
            #         elif mutation_op == 0:
            #             self.alter_unit(offspring, mutation_unit_pos, mutation_unit_type)
            # # 删除block
            # elif mutation_op == -1:
            #     # 如果当前待删除的unit的block待到了下限，则选择其他unit的block进行删除，如果均到达下限，则选择其他两种变异操作
            #     valid = False
            #     if mutation_unit_type == 0:
            #         for encoder_unit in offspring.encoder_units:
            #             if encoder_unit.block_amount > offspring.min_block_amount:
            #                 valid = True
            #                 break
            #         if valid == False:
            #             for decoder_unit in offspring.decoder_units:
            #                 if decoder_unit.block_amount > offspring.min_block_amount:
            #                     mutation_unit_type = 1
            #                     valid = True
            #                     break
            #     else:
            #         for decoder_unit in offspring.decoder_units:
            #             if decoder_unit.block_amount > offspring.min_block_amount:
            #                 valid = True
            #                 break
            #         if valid == False:
            #             for encoder_unit in offspring.encoder_units:
            #                 if encoder_unit.block_amount > offspring.min_block_amount:
            #                     mutation_unit_type = 0
            #                     valid = True
            #                     break
            #     if valid:
            #         if mutation_unit_type == 0:
            #             while offspring.encoder_units[mutation_unit_pos].block_amount == offspring.min_block_amount:
            #                 mutation_unit_pos = self.get_new_muation_unit_pos(mutation_unit_pos, offspring)
            #             # 删除block
            #             # offspring.encoder_units[mutation_unit_pos].block_amount -= 1
            #             self.remove_block(offspring, mutation_unit_pos, mutation_unit_type)
            #         elif mutation_unit_type == 1:
            #             while offspring.decoder_units[mutation_unit_pos].block_amount == offspring.min_block_amount:
            #                 mutation_unit_pos = self.get_new_muation_unit_pos(mutation_unit_pos, offspring)
            #             # 删除block
            #             # offspring.decoder_units[mutation_unit_pos].block_amount -= 1
            #             self.remove_block(offspring, mutation_unit_pos, mutation_unit_type)
            #     else:
            #         mutation_op = np.random.randint(0,2)
            #         mutation_unit_type = np.random.randint(0,1)  # 随机变异的unit 类型/ encoder,decoder
            #         mutation_unit_pos = np.random.randint(0,offspring.level_amount) # unit 位置
            #         # 添加block
            #         if mutation_op == 1:
            #             if mutation_unit_type == 0:
            #                 self.add_block(offspring, mutation_unit_pos, mutation_unit_type)
            #                 # offspring.encoder_units[mutation_unit_pos].block_amount += 1
            #             elif mutation_unit_type == 1:
            #                 # offspring.decoder_units[mutation_unit_pos].block_amount += 1
            #                 self.add_block(offspring, mutation_unit_pos, mutation_unit_type)
            #         # 修改block
            #         elif mutation_op == 0:
            #             self.alter_unit(offspring, mutation_unit_pos, mutation_unit_type)

            # # 修改block
            # elif mutation_op == 0:
            #     self.alter_unit(offspring, mutation_unit_pos, mutation_unit_type)

        return self.offsprings         



                        
                
                        
                    
                

                



            




