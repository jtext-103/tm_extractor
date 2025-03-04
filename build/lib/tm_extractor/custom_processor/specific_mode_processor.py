# -*-  coding:utf-8  -*-
import numpy as np
from typing import Tuple
from jddb.processor import Signal, BaseProcessor
from copy import deepcopy

class SpecificModeExtractProcessor(BaseProcessor):
    "提取给定模式m/n的幅值、频率，是否大于2Gs，是否与其他模式出现了耦合情况"
    def __init__(self,m,n,bigger_threshold):
        super().__init__(m=m,n=n, bigger_threshold=bigger_threshold)
        self.m=m
        self.n=n
        self.bigger_threshold=bigger_threshold

    def find_index(self, even_odd_th_array, fre):
        # 获取最后一列
        last_column = even_odd_th_array[:, -1]
        # 计算与 fre 的绝对误差
        error = np.abs(last_column - fre)
        # 找到误差小于 1000 的索引
        valid_indices = np.where(error < 1000)[0]
        if len(valid_indices) == 0:
            return None  # 如果没有满足条件的行，返回 None
        # 在满足条件的行中，找到误差最小的索引
        min_error_index = valid_indices[np.argmin(error[valid_indices])]
        return min_error_index

    def transform(self, *signal: Signal) -> Tuple[Signal, ...]:
        #"\\new_B_th_nm_most_judge", [0]
        # "\\new_B_th_nm_sec_judge", [1]
        # "\\new_B_th_nm_third_judge", [2]
        #"\\new_B_th_nm_forth_judge", [3]
        # "\\new_B_even_n_most_th", [4]
        # "\\new_B_even_n_sec_th", [5]
        # "\\new_B_even_n_third_th", [6]
        # "\\new_B_even_n_forth_th", [7]
        # "\\new_B_odd_n_most_th", [8]
        # "\\new_B_odd_n_sec_th", [9]
        # "\\new_B_odd_n_third_th", [10]
        # "\\new_B_odd_n_forth_th", [11]

        most_signal = deepcopy(signal.__getitem__(0))
        # sec_signal = deepcopy(signal.__getitem__(1))
        # third_signal = deepcopy(signal.__getitem__(2))
        # forth_signal = deepcopy(signal.__getitem__(3))
        mn_is_couple = deepcopy(signal.__getitem__(0))
        mn_is_exist = deepcopy(signal.__getitem__(0))
        mn_bigger_than_2_Gs = deepcopy(signal.__getitem__(0))
        mn_amp_fre = deepcopy(signal.__getitem__(0))
        mn_is_couple_data = np.zeros((len(most_signal.data)), dtype=int)
        mn_is_exist_data = np.zeros((len(most_signal.data)), dtype=int)
        mn_bigger_than_2_Gs_data = np.zeros((len(most_signal.data)), dtype=int)
        mn_amp_fre_data = np.zeros((len(most_signal.data), 2), dtype=float)
        # mn是否与其他模数耦合
        # mn是否存在
        # mn是否大于2Gs
        # mn的幅值、频率（不存在的时候给0）
        if self.m % 2 == 0:
            index_m = -4
            index_n = -3
            other_m = -2
            other_n = -1
        else:
            other_m = -4
            other_n = -3
            index_m = -2
            index_n = -1
        for i in range(len(most_signal.data)):
            data_th_array = np.array([signal.__getitem__(th).data[i] for th in range(4)])
            index_even_odd_th = None
            if self.m % 2 == 0:
                even_odd_th_array = np.array([signal.__getitem__(th).data[i] for th in range(4, 8)])
            else:
                even_odd_th_array = np.array([signal.__getitem__(th).data[i] for th in range(8, 12)])
            for index_th in range(len(data_th_array)):
                if data_th_array[index_th, index_m] == self.m and data_th_array[index_th, index_n] == self.n:
                    mn_is_exist_data[i] = 1
                    # if data_th_array[index_th, 2]>=2:
                    #     mn_bigger_than_2_Gs_data[i]=1
                    if data_th_array[index_th, other_m] > 0 and data_th_array[index_th, other_n] > 0:  # 判断耦合的
                        mn_is_couple_data[i] = 1
                        index_even_odd_th = self.find_index(even_odd_th_array, data_th_array[index_th, 3])
                        if index_even_odd_th is not None:
                            mn_amp_fre_data[i, 0] = even_odd_th_array[index_even_odd_th, 1]
                            mn_amp_fre_data[i, 1] = even_odd_th_array[index_even_odd_th, 2]
                        elif data_th_array[index_th, 1] == 1 or index_even_odd_th is None:  # 耦合但是找不到对应的
                            mn_amp_fre_data[i, 0] = data_th_array[index_th, 2]
                            mn_amp_fre_data[i, 1] = data_th_array[index_th, 3]
                    else:
                        mn_amp_fre_data[i, 0] = data_th_array[index_th, 2]
                        mn_amp_fre_data[i, 1] = data_th_array[index_th, 3]
                    if mn_amp_fre_data[i, 0] > 2:
                        mn_bigger_than_2_Gs_data[i] = 1
                    # if mn_bigger_than_2_Gs_data[i]!=1 or index_even_odd_th is None:
                    #     mn_amp_fre_data[i, 0] = data_th_array[index_th, 2]
                    #     mn_amp_fre_data[i, 1] = data_th_array[index_th, 3]
                    break
        # mn_is_couple.data=mn_is_couple_data
        mn_is_exist.data = mn_is_exist_data
        mn_is_couple.data = mn_is_couple_data
        mn_bigger_than_2_Gs.data = mn_bigger_than_2_Gs_data
        mn_amp_fre.data = mn_amp_fre_data
        return mn_amp_fre, mn_is_couple, mn_is_exist, mn_bigger_than_2_Gs

