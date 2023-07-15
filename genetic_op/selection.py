import numpy as np
import copy
class Selection(object):
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