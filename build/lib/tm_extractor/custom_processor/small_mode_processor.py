# -*-  coding:utf-8  -*-
import numpy as np
import math
from typing import Tuple, Union
from jddb.processor import Signal, BaseProcessor
from copy import deepcopy


class SmallModeProcessor(BaseProcessor):
    "在幅值为0.5-2之间的模式，根据qa，给出模数"
    def __init__(self, threshold_down=0.5,threshold_upp=2):
        super().__init__(threshold_down=threshold_down,threshold_upp=threshold_upp)
        self.threshold_down = threshold_down
        self.threshold_upp = threshold_upp
    def transform(self, *signal: Signal) -> Union[Signal, Tuple[Signal, ...]]:
        # "\\new_B_LFS_nm_most_judge", signal[0]
        # "\\new_B_LFS_nm_sec_judge", signal[1]
        # "\\new_B_LFS_nm_third_judge",signal[2]
        # "\\new_B_LFS_nm_forth_judge", signal[3]
        # "\\qa_1k"signal[4]
        most_signal = deepcopy(signal.__getitem__(0))
        sec_signal = deepcopy(signal.__getitem__(1))
        third_signal = deepcopy(signal.__getitem__(2))
        forth_signal = deepcopy(signal.__getitem__(3))
        qa_1k_signal = deepcopy(signal.__getitem__(4))
        for i in range(len(most_signal.data)):
            data_arrays = np.array([most_signal.data[i], sec_signal.data[i], third_signal.data[i], forth_signal.data[i]])
            # for i, arr in enumerate(new_arrays):
            #     print(f"Array {i + 1}: {arr}")
            data_arrays = self.small_mode(data_arrays, qa_1k_signal.data[i])
            most_signal.data[i] = data_arrays[0]
            sec_signal.data[i] = data_arrays[1]
            third_signal.data[i] = data_arrays[2]
            forth_signal.data[i] = data_arrays[3]
        return most_signal, sec_signal, third_signal, forth_signal
    def round_number(self, number):
        if number < 1.5:
            return 1
        else:
            integer_part = int(number)  # 获取整数部分
            decimal_part = number - integer_part  # 获取小数部分
            if decimal_part >= 0.55:
                return integer_part + 1  # 小数部分 >= 0.55，向上取整
            else:
                return integer_part  # 小数部分 < 0.55，舍去小数部分
    def qa_condition(self,m_csd_phase, n_number, qa):
        #如果qa与m/n相差1，且m/n>qa 那么m应该没有取对
        #但如果m/n=2/1，那么qa不设置限制
        if m_csd_phase/n_number-qa<=1 and m_csd_phase/n_number!=1:
            return True
        else:
            if round(m_csd_phase/n_number)==2:
                return True
            else:
                return False
    def check_even_after_gcd(self, m, n):
        m=int(m)
        n=int(n)
        # 计算最大公因数
        gcd = math.gcd(m, n)

        # 检查 m 除以 GCD 的结果是否为偶数
        if (m // gcd) % 2 == 0:
            return int(m/gcd),int(n/gcd),True  # m 被认为是偶数
        else:
            return int(m/gcd),int(n/gcd),False  # m 被认为是奇数

    def mode_m_n_identical(self, m, n,n_csd):
        if m == n :
            return m,self.round_number(n_csd)
        else:
            return m,n

    def check_conditions(self, arr):
        # 检查最后两列是否全小于等于 0
        condition1 = np.all(arr[-2:] > 0)

        # 检查倒数第三列和倒数第四列是否全小于 0
        condition2 = np.all(arr[-3:-1] > 0)

        # 两个条件同时成立
        return condition1 and condition2

    def small_mode(self,data_arrays,qa):
        mode_state = 0
        new_data_arrays = deepcopy(data_arrays)

        if all(value <= 2 for value in new_data_arrays[:, 2]):
            for index in range(len(data_arrays)):
                if not self.check_conditions(data_arrays[index]):
                    if self.threshold_down <= data_arrays[index][2] < self.threshold_upp and data_arrays[index][3]>1600:
                        new_data_arrays[index, -4:] = -1
                        #先确定n的值
                        if new_data_arrays[index][8]== 0 :
                            n_number = self.round_number(new_data_arrays[index][0])
                        else:
                            n_number = new_data_arrays[index][8]
                        #m_phase不为0，再根据qa确定m的值 以及m/n是否为偶数
                        if new_data_arrays[index][7] != 0 and self.qa_condition(new_data_arrays[index][7], n_number, qa):
                            m,n,is_even=self.check_even_after_gcd(new_data_arrays[index][7], n_number)
                            m,n = self.mode_m_n_identical(m, n, new_data_arrays[index][0])
                            if is_even:
                                new_data_arrays[index][9]=m
                                new_data_arrays[index][10]=n
                            else:
                                new_data_arrays[index][11]=m
                                new_data_arrays[index][12]=n
                            mode_state = 1
                        else:
                            m_csd=self.round_number(new_data_arrays[index][6])
                            if not(self.qa_condition(m_csd, n_number, qa)):
                                m_csd=self.round_number(qa*n_number)
                            m, n, is_even = self.check_even_after_gcd(m_csd, n_number)
                            m, n = self.mode_m_n_identical(m, n, new_data_arrays[index][0])
                            if is_even:
                                new_data_arrays[index][9]=m
                                new_data_arrays[index][10]=n
                            else:
                                new_data_arrays[index][11]=m
                                new_data_arrays[index][12]=n
                            mode_state = 1
                if mode_state == 1:
                    break

        return new_data_arrays
