# -*-  coding:utf-8  -*-
import numpy as np
from scipy.fftpack import fft
import math
from typing import Tuple, Union
from jddb.processor import Signal, BaseProcessor
from typing import Optional, List
from scipy.fftpack import fft
from copy import deepcopy

class AmpJudgeProcessor(BaseProcessor):
    """
       给定频率,判断B_tehta是否大于阈值,如果大于阈值,则认为是撕裂模,如果不是撕裂模，其频率标注为-1
    """

    def __init__(self, threshold=2):  # pad 表示f精度，精度为 1/(fs*pad),f 的长度为 pad*fs/2
        super().__init__(threshold=threshold)
        self.threshold=threshold
    def transform(self, *signal: Signal):
        # #signal[0]:m_most_max_all
        # #signal[1]:m_sec_max_all
        # #signal[2]:m_third_max_all
        # #signal[3]:m_forth_max_all
        # #signal[4]:even_most_max_all
        # #signal[5]:even_sec_max_all
        # #signal[6]:even_third_max_all
        # #signal[7]:even_forth_max_all
        # #signal[8]:odd_most_max_all
        # #signal[9]:odd_sec_max_all
        # #signal[10]:odd_third_max_all
        # #signal[11]:odd_forth_max_all
        mode_fre_signal = deepcopy((signal.__getitem__(0)))
        m_most_all_signal = deepcopy((signal.__getitem__(0)))
        m_sec_all_signal = deepcopy((signal.__getitem__(1)))
        m_third_all_signal = deepcopy((signal.__getitem__(2)))
        m_forth_all_signal = deepcopy((signal.__getitem__(3)))
        even_signals=[deepcopy(signal.__getitem__(i)) for i in range(4,8)]
        odd_signals=[deepcopy(signal.__getitem__(i)) for i in range(8,11)]
        # n_m_th_signal = [deepcopy((signal.__getitem__(i)) for i in range(len(signal)))]
        mode_fre = np.zeros(shape=(len(mode_fre_signal.data), 4), dtype=float)
        mode_fre[:,0], m_most_all_signal.data,  = self.amp_judge(m_most_all_signal.data,even_signals,odd_signals)
        mode_fre[:,1], m_sec_all_signal.data = self.amp_judge(m_sec_all_signal.data,even_signals,odd_signals)
        mode_fre[:,2], m_third_all_signal.data = self.amp_judge(m_third_all_signal.data,even_signals,odd_signals)
        mode_fre[:,3], m_forth_all_signal.data = self.amp_judge(m_forth_all_signal.data,even_signals,odd_signals)
        mode_fre_signal.data=mode_fre
        return mode_fre_signal, m_most_all_signal, m_sec_all_signal, m_third_all_signal, m_forth_all_signal

    def odd_even_judge(self, odd_even_arr, target, diff_freq_threshold=1000):
        # 获取第三列数据
        third_col = odd_even_arr[:, 2]
        # 计算绝对差
        abs_diff = np.abs(third_col - target)
        # 找到最小差值的索引
        row_idx = abs_diff.argmin()
        # 检查差值是否超过阈值
        if abs_diff[row_idx] > diff_freq_threshold:
            return False  # 或者返回其他默认值
        else:
            # 返回该行第二列的值
            if odd_even_arr[row_idx, 1]>=self.threshold:
                return True
            else:
                return False

    def amp_judge(self, m_data,even_signals,odd_signals):#根据幅值判断是否计入撕裂模
        lfs_amp_fre_cover = np.zeros(shape=(len(m_data), 8), dtype=float)
        mode_fre = np.zeros(shape=(len(m_data)), dtype=float)

        for i in range(len(m_data)):
            even_data_i = np.array([even_signals.__getitem__(signal_i).data[i] for signal_i in range(len(even_signals))])
            odd_data_i = np.array([odd_signals.__getitem__(signal_i).data[i] for signal_i in range(len(odd_signals))])
            if m_data[i][1]>=self.threshold:
                lfs_amp_fre_cover[i] = np.concatenate([np.array([m_data[i][0]]),np.array([1]),m_data[i][1:]])
                mode_fre[i]=m_data[i][2]
            elif 0<m_data[i][1]<self.threshold:#这个地方其实不会到0，到0.5这个阈值0.5<=data<2Gs
                if self.odd_even_judge(even_data_i, m_data[i][2]) and self.odd_even_judge(odd_data_i, m_data[i][2]):
                    lfs_amp_fre_cover[i] = np.concatenate([np.array([m_data[i][0]]), np.array([1]), m_data[i][1:]])
                    mode_fre[i] = m_data[i][2]
                else:
                    lfs_amp_fre_cover[i] = np.concatenate([np.array([m_data[i][0]]),np.array([0]),m_data[i][1:]] )
                # mode_fre[i]=0
            else:
                mode_fre[i]=-1#
                lfs_amp_fre_cover[i][1] = -1
        return mode_fre, lfs_amp_fre_cover