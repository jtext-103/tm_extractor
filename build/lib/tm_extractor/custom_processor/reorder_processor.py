# -*-  coding:utf-8  -*-
import numpy as np
from scipy.fftpack import fft
import math
from typing import Tuple, Union
from jddb.processor import Signal, BaseProcessor
from copy import deepcopy

class ReOrderProcessor(BaseProcessor):
    def __init__(self):
        super().__init__()

    def reorder(self, data_arrays, is_tearing_arrays):
        # 生成 (索引, 第二个元素值) 列表
        data_with_indices = [(i, arr[1]) for i, arr in enumerate(data_arrays)]
        # 按照第二个元素值从大到小排序
        sorted_data = sorted(data_with_indices, key=lambda x: x[1], reverse=True)
        # 处理相邻元素，使其满足“数值差不大于1则保持原顺序”
        grouped_data = []
        temp_group = [sorted_data[0]]  # 初始化第一个分组

        for i in range(1, len(sorted_data)):
            idx, value = sorted_data[i]
            prev_idx, prev_value = sorted_data[i - 1]
            if prev_value - value <= 2:  # 差值不大于 1，属于同一组
                temp_group.append((idx, value))
            else:  # 开始新的分组
                grouped_data.extend(sorted(temp_group, key=lambda x: x[0]))  # 按原索引顺序
                temp_group = [(idx, value)]
        # 处理最后一组
        grouped_data.extend(sorted(temp_group, key=lambda x: x[0]))
        # 生成新的排列
        new_arrays = [[data_arrays[idx][0], data_arrays[idx][1], data_arrays[idx][2]] for idx, _ in grouped_data]
        is_tearing_new_arrays = [is_tearing_arrays[idx] for idx, _ in grouped_data]
        return new_arrays, is_tearing_new_arrays

    def transform(self, *signal: Signal) -> Tuple[Signal, ...]:
        most_signal = deepcopy(signal.__getitem__(0))
        sec_signal = deepcopy(signal.__getitem__(1))
        third_signal = deepcopy(signal.__getitem__(2))
        forth_signal = deepcopy(signal.__getitem__(3))
        is_tearing_most_signal = deepcopy(signal.__getitem__(4))
        is_tearing_sec_signal = deepcopy(signal.__getitem__(5))
        is_tearing_third_signal = deepcopy(signal.__getitem__(6))
        is_tearing_forth_signal = deepcopy(signal.__getitem__(7))
        for i in range(len(most_signal.data)):
            data_arrays = [most_signal.data[i], sec_signal.data[i], third_signal.data[i], forth_signal.data[i]]
            is_tearing_arrays=[is_tearing_most_signal.data[i], is_tearing_sec_signal.data[i], is_tearing_third_signal.data[i], is_tearing_forth_signal.data[i]]
            # for i, arr in enumerate(new_arrays):
            #     print(f"Array {i + 1}: {arr}")
            data_arrays, is_tearing_arrays = self.reorder(data_arrays,is_tearing_arrays)
            most_signal.data[i]=data_arrays[0]
            sec_signal.data[i]=data_arrays[1]
            third_signal.data[i]=data_arrays[2]
            forth_signal.data[i]=data_arrays[3]
            is_tearing_most_signal.data[i]=is_tearing_arrays[0]
            is_tearing_sec_signal.data[i]=is_tearing_arrays[1]
            is_tearing_third_signal.data[i]=is_tearing_arrays[2]
            is_tearing_forth_signal.data[i]=is_tearing_arrays[3]

        return most_signal,sec_signal,third_signal,forth_signal,is_tearing_most_signal,is_tearing_sec_signal,is_tearing_third_signal,is_tearing_forth_signal



#     def reorder(self, data_arrays,is_tearing_arrays):
#         new_data = np.empty(shape=(0, 4), dtype=float)
#         # for i in range(len(most_data)):
#         #         #     new_data = np.vstack([new_data, [most_data[i], sec_data[i], third_data[i], forth_data[i]]])
#
#         # 将每个数组的第二个元素和索引结合起来
#
#         data_with_indices = [(i, arr[1]) for i, arr in enumerate(data_arrays)]
#
#         # 按照第二个元素从大到小排序
#         sorted_data = sorted(data_with_indices, key=lambda x: x[1], reverse=True)
#
#         # 生成新数组
#         new_arrays = [
#             [data_arrays[idx][0], data_arrays[idx][1], data_arrays[idx][2]]  # 取第一个元素、第二个元素，以及原位置
#             for idx, _ in sorted_data
#         ]
#         # 输出结果
#         is_tearing_new_arrays=[
#             is_tearing_arrays[idx]  # 取第一个元素、第二个元素，以及原位置
#             for idx, _ in sorted_data
#         ]
#         return new_arrays, is_tearing_new_arrays
# #