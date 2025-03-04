# -*-  coding:utf-8  -*-
import numpy as np
from jddb.processor import Signal, BaseProcessor
from copy import deepcopy

class DoubleJudgeProcessor(BaseProcessor):
    """
       判断是否是倍频/非倍频模式
           # 判断是否为倍频模式 # 那么只要被放入了 mode_double_fre就不可以再跟其他比较了
    # A B C D四种频率
    # A B为倍频，则只取A mode_undouble_fre(1, 0, *, *)  mode_double_fre(0, 1, *, *)
    # A B不为倍频 则取 A B 到mode_undouble_fre(1, 1, *, *) mode_double_fre(0, 0, *, *)
    # B为倍频
    # A C为倍频，则只取A mode_undouble_fre(1, 0, 0, *) B mode_double_fre(0, 1, 1, *)
    # A C不为倍频 则取 A C 到mode_undouble_fre(1, 0, 1, *) mode_double_fre(0, 1, 0, *)
    # B 不为倍频
    # A C为倍频，则只取A mode_undouble_fre(1, 0, 0, *) B mode_double_fre(0, 1, 1, *)
    # A C不为倍频 则取 A C 到mode_undouble_fre(1, 0, 1, *) mode_double_fre(0, 1, 0, *)
    """

    def __init__(self, fre_diff_threshold=1000, dpi=100):  # （fre +-400/fre_base） =有一个是整数就认为是倍频
        super().__init__(fre_diff_threshold=fre_diff_threshold, dpi=dpi)
        self.fre_diff_threshold=fre_diff_threshold
        self.dpi = dpi
    def transform(self, signal: Signal):
        mode_fre_signal = deepcopy(signal)
        double_fre_data = np.zeros(shape=(len(mode_fre_signal.data), 4), dtype=float)
        undouble_fre_data = np.zeros(shape=(len(mode_fre_signal.data), 4), dtype=float)
        index=0
        for slice_index in  range(len(mode_fre_signal.data)):#412,426
            mode_fre_signal.data[slice_index][mode_fre_signal.data[slice_index] <=0] = np.inf
            double_fre_data[index], undouble_fre_data[index] = self.judge_fre_double(mode_fre_signal.data[slice_index])
            index+=1
        mode_fre_signal.data = undouble_fre_data
        return mode_fre_signal

    def sort_with_indices(self, arr):
        # 对数组按从小到大排序，同时返回排序后的元素和它们的原始索引
        return sorted(enumerate(arr), key=lambda x: x[1])

    # 对数组进行排序并返回排序元素及原始索引
    def double_judge(self, m_amp, m_amp_later):  # 根据幅值判断是否计入撕裂模
        is_double = False
        # if m_amp_later==np.inf:
        m_amp_arr = self.generate_sequence(m_amp=m_amp, step=self.dpi,
                                      count=int(self.fre_diff_threshold / self.dpi) *2 + 1)
        m_amp_later_arr = self.generate_sequence(m_amp=m_amp_later, step=self.dpi,
                                            count=int(self.fre_diff_threshold / self.dpi) * 2 + 1)
        if self.check_divisible(arr1=m_amp_arr, arr2=m_amp_later_arr):
            is_double = True

        return is_double

    def check_divisible(self, arr1, arr2):
        # 遍历 arr1 和 arr2 中的每个元素，检查是否存在相除为整数的情况
        for a in arr1:
            for b in arr2:
                if b != 0 and b % a == 0:  # 避免除以 0
                    return True  # 一旦找到，立即返回 True
        return False  # 如果遍历结束仍未找到，返回 False

    def generate_sequence(self, m_amp, step, count):
        # 生成以 m_amp 为中心，间隔为 step 的等差数列
        start = m_amp - step * (count // 2)  # 计算起始值
        return [start + i * step for i in range(count)]

    def process_array(self, undouble_fre,double_fre,mode_fre_i):
        # 找到值为 1 的元素的索引
        undouble_index = np.argmax(undouble_fre)  # 适合仅一个 1 的情况

        # 如果索引不为 0，修改数组
        if undouble_index != 0 and mode_fre_i[0]!=0 and mode_fre_i[0]!=np.inf:
            undouble_fre[undouble_index] = 0
            undouble_fre[0] = 1
            double_fre[undouble_index] = 1
            double_fre[0] = 0

        # 返回索引和修改后的数组
        return undouble_fre,double_fre

    def judge_fre_double(self, mode_fre_data_slice):
        processed_double_fre = np.zeros(4, dtype=float)
        processed_undouble_fre = np.zeros(4, dtype=float)
        inf_fre = np.zeros(4, dtype=float)
        inf_fre[mode_fre_data_slice == np.inf] = 1#mode fre为-1的即<2Gs的情况 为mode_fre_data_slice == np.inf
        processed_undouble_fre[0] = 1
        sorted_with_indices = self.sort_with_indices(mode_fre_data_slice)
        sorted_indices = [index for index, value in sorted_with_indices]
        sorted_mode_fre = np.array([mode_fre_data_slice[index] for index in sorted_indices])
        sorted_inf_fre = np.array([inf_fre[index] for index in sorted_indices])
        for i in range(4):
            if processed_double_fre[i] != 1:
                for j in range(4):
                    if processed_double_fre[j] != 1 and i != j and sorted_inf_fre[i]!=1 and sorted_inf_fre[j]!=1:#mode fre为-1的即<2Gs的情况 为1
                        is_double = self.double_judge(sorted_mode_fre[i], sorted_mode_fre[j])
                        if is_double:
                            processed_double_fre[j] = 1
                            processed_undouble_fre[j] = 0
                        else:
                            processed_double_fre[j] = 0
                            processed_undouble_fre[j] = 1
                        #如果只有一个频率，但是基频不是the_most,则将the_most的processed_undouble_fre置为[1,0,0,0]processed_double_fre=[0,1,1,1],-1都还是保留
        processed_undouble_fre[ sorted_inf_fre == 1] = -1#mode fre为-1的即<2Gs的情况 ，等同于sorted_inf_fre == 1 ，在最高后的处理时processed_undouble_fre取-1
        # 恢复原始顺序
        # restore_indices = np.argsort(sorted_indices)
        double_fre = processed_double_fre[np.argsort(sorted_indices)]
        undouble_fre =processed_undouble_fre[np.argsort(sorted_indices)]
        if np.sum(processed_undouble_fre == 1) == 1:#如果只有一个频率，但是基频不是the_most,则将the_most的processed_undouble_fre置为[1,0,0,0]processed_double_fre=[0,1,1,1],-1都还是保留
            undouble_fre, double_fre = self.process_array(undouble_fre, double_fre,mode_fre_data_slice)
        return double_fre, undouble_fre