from copy import deepcopy
from typing import Tuple, List

import numpy as np
from jddb.processor import Signal
from scipy import signal as sig
from jddb.processor import *


class ModeFreNoisyCheck(BaseProcessor):
    """
           -------
            fre_th类型float 默认值4000
           -------
    """

    def __init__(self,fre_th: float = 4000):
        # var_th=1e-13, coherence_th=0.95, real_angle=7.5
        # var_th=1e-20, coherence_th=0.9, real_angle=15
        # var_th=1e-13, coherence_th=0.9, real_angle=22.5
        ## var_th=1e-14, coherence_th=0.95, real_angle=22.5
        # super().__init__(fre_th=fre_th)
        super().__init__()
        self.fre_th = fre_th

    def find_noisy_indices(self,arr):
        indices = []
        skipped_indices = set()  # 用于记录已经处理的索引
        arr_indices = []
        # 处理前两行
        for i in range(2):  # 第一行和第二行
            for j in range(len(arr[i])):  # 遍历每行的每个元素
                if arr[i][j] < self.fre_th and arr[i][j] != 0:
                    arr_indices.append((i, j))  # 将索引加入结果
                    # 找到按照行最靠后然后列最靠前的第一个符合条件的数a
                    a = arr[i][j]
                    # 检查后两行的元素
                    if i + 2 < len(arr):
                        # 判断后两行元素是否满足条件
                        if arr[i + 1].sum() == 0 and arr[i + 2].sum() == 0:  # 后两行全为0
                            indices.append((i, j))  # 将索引加入结果
                        else:
                            # 检查后两行是否所有非0元素与a的差大于4000
                            valid_elements = True
                            for k in range(i + 1, i + 3):  # 后两行
                                for l in range(len(arr[k])):
                                    if arr[k][l] != 0 and abs(arr[k][l] - a) <= self.fre_th:
                                        valid_elements = False
                                        break
                                if not valid_elements:
                                    break
                            if valid_elements:
                                for i, j in arr_indices:
                                    indices.append((i, j))
                                # indices.append((i, j))  # 将索引加入结果
        # 遍历数组的每一行，从第三行开始
        for i in range(3, len(arr)):  # 从第三行开始，因为我们需要前两行数据
            for j in range(len(arr[i])):  # 遍历每行的每个元素
                if (i, j) in skipped_indices:  # 如果该索引已处理，跳过
                    continue
                if arr[i][j] < self.fre_th and arr[i][j] != 0:
                    # 检查前后行中是否有连续的3个小于4000且不为0的数
                    valid_elements = 0
                    for k in range(i - 2, i + 1):  # 前后3行的检查
                        if k >= 0 and k < len(arr):  # 确保行号有效
                            if arr[k][j] < self.fre_th and arr[k][j] != 0:
                                valid_elements += 1
                    # 如果该元素是第一个符合条件的小于4000且不为0的元素
                    if valid_elements > 0:
                        # 判断前两行所有不为0的元素减去数a的结果是否都大于4000
                        a = arr[i][j]
                        prev_elements_check = True
                        for k in range(i - 2, i):  # 检查前两行
                            for l in range(len(arr[k])):
                                if arr[k][l] != 0 and (k, l) not in skipped_indices:  # 非0元素进行检查
                                    if abs(arr[k][l] - a) <= self.fre_th:
                                        prev_elements_check = False
                                        break
                            if not prev_elements_check:
                                break
                        # 如果条件满足，返回索引
                        if prev_elements_check or (arr[i - 1].sum() == 0 and arr[i - 2].sum() == 0):  # 前两行全是0
                            # 找到连续的符合条件的索引
                            consecutive_indices = [(i, j)]
                            # 检查下一个连续的符合条件的元素
                            next_i = i + 1
                            while next_i < len(arr) and arr[next_i][j] < self.fre_th and arr[next_i][j] != 0:
                                consecutive_indices.append((next_i, j))
                                skipped_indices.add((next_i, j))  # 跳过已处理的索引
                                next_i += 1
                            # 将这些索引添加到结果中
                            indices.extend(consecutive_indices)
                            skipped_indices.add((i, j))  # 标记当前索引为已处理
                            break  # 跳过这些已经处理过的索引
        return indices

    def find_noisy_indices_from_m(self, n_fre_array: np.ndarray, m_fre_array: np.ndarray):
        indices_from_m = []
        # 遍历n_fre_array的每一行
        for i in range(len(n_fre_array)):
            for j in range(len(n_fre_array[i])):
                if n_fre_array[i][j] < self.fre_th:  # 检查是否小于4000
                    # 检查与m_fre_array对应行的所有元素做差的绝对值是否都大于4000
                    valid = True
                    for k in range(len(m_fre_array[i])):
                        if (m_fre_array[i][k]!=0 and abs(n_fre_array[i][j] - m_fre_array[i][k]) <= self.fre_th) or np.all(m_fre_array[i] == 0):
                        # if abs(n_fre_array[i][j] - m_fre_array[i][k]) <= self.fre_th:
                            valid = False
                            break
                    if valid:
                        indices_from_m.append((i, j))  # 返回符合条件的索引
        return indices_from_m

    def transform(self, *signal: Signal)-> Tuple[Signal]:
        # "\\n_most_max_th", "\\n_sec_max_th", "\\n_third_max_th", "\\n_forth_max_th",
        #signal[0]: "\\n_most_max_th"
        #signal[1]: "\\n_sec_max_th"
        #signal[2]: "\\n_third_max_th"
        #signal[3]: "\\n_forth_max_th"
        #signal[4]: "\\m_most_max_th"
        #signal[5]: "\\m_sec_max_th"
        #signal[6]: "\\m_third_max_th"
        #signal[7]: "\\m_forth_max_th"
        n_signal = [deepcopy(signal.__getitem__(i_signal)) for i_signal in range(4)]
        n_fre_array=np.zeros((signal.__getitem__(0).data.shape[0],4))
        n_data=np.zeros((4,signal.__getitem__(0).data.shape[0],signal.__getitem__(0).data.shape[1]))
        for i_signal in range(4):
            n_data[i_signal]=n_signal[i_signal].data
        for i_signal in range(4):
            n_fre_array[:,i_signal]=n_signal[i_signal].data[:,3]
        indices = self.find_noisy_indices(n_fre_array)
        for i,j in indices:#j表示第几个元素，i表示第几行
            n_data[j,i] =np.zeros(signal.__getitem__(0).data.shape[1])
        for i in range(4):
            n_signal[i].data=n_data[i]

        m_signal = [deepcopy(signal.__getitem__(i_signal)) for i_signal in range(4,8)]
        m_fre_array = np.zeros((signal.__getitem__(0).data.shape[0], 4))
        for i_signal in range(4):
            m_fre_array[:, i_signal] = m_signal[i_signal].data[:, 3]
        indices_from_m = self.find_noisy_indices_from_m(n_fre_array, m_fre_array)
        for i, j in indices_from_m:
            m_signal[j].data[i] = np.zeros(signal.__getitem__(0).data.shape[1])

        return n_signal[0], n_signal[1], n_signal[2], n_signal[3]