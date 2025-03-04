import os
from collections import Counter
import math
from typing import Tuple
from copy import deepcopy
import numpy as np
from scipy import signal as sig
from jddb.processor import *
class MNModeUnionProcessor(BaseProcessor):
    def __init__(self,pol_samperate: int,real_angle:float,pad=1):
        super().__init__(pol_samperate=pol_samperate,real_angle=real_angle,pad=pad)
        self.pad = pad
        self.pol_samperate = pol_samperate
        self.real_angle = real_angle
        self.n_pad=250

    def transform(self, *signal: Signal) -> Tuple[Signal, ...]:
        n_most_signal = deepcopy(signal.__getitem__(0))
        n_sec_signal = deepcopy(signal.__getitem__(1))
        n_third_signal = deepcopy(signal.__getitem__(2))
        n_forth_signal = deepcopy(signal.__getitem__(3))
        m_four_signal_list = [deepcopy(signal.__getitem__(i)) for i in range(4, 8)]
        # m_most_signal = deepcopy(signal.__getitem__(4))
        # m_sec_signal = deepcopy(signal.__getitem__(5))
        # m_third_signal = deepcopy(signal.__getitem__(6))
        # m_forth_signal = deepcopy(signal.__getitem__(7))
        m_phase_four_signal = deepcopy(signal.__getitem__(8))
        pol03_signal = deepcopy(signal.__getitem__(9))
        pol04_signal = deepcopy(signal.__getitem__(10))
        n_data = np.array([n_most_signal.data, n_sec_signal.data, n_third_signal.data, n_forth_signal.data])
        n_data_new = np.zeros( (n_data.shape[0], n_data.shape[1], n_data.shape[2] + 4))
        #new_B_LFS_n_most_mode_number
        self.n_pad=len(pol03_signal.data[0])*self.pad
        self.pol_samperate = pol03_signal.attributes["OriginalSampleRate"]
        # self.f=np.fft.fftfreq(self.n_pad, 1 / pol03_signal.attributes["OriginalSampleRate"]) [:int(self.n_pad / 2)]
        for i in range(len(n_most_signal.data)):

            m_4fre_i = [m_four_signal.data[i,3] for m_four_signal in m_four_signal_list]
            m_4mode_i =[m_four_signal.data[i,0] for m_four_signal in m_four_signal_list]
            for j in range(len(n_data)):
                if not np.all(n_data[j][i] == 0):
                    use_ampd = 0
                    #找到对应的m_four_signal_list的模数和频率
                    m_fre_closest_index,m_fre_closest_value=self.find_fre_mode(m_4fre_i,n_data[j][i][2])
                    if m_fre_closest_index!=-1:
                        m_mode_number=m_4mode_i[m_fre_closest_index]
                        use_ampd=1
                    else:
                        #重新做一下csd然后找到对应的值去填充
                        m_mode_number,m_fre_closest_value=self.csd_resolve(n_data[j][i][2],pol03_signal.data[i],pol04_signal.data[i])
                    n_data_new[j][i] = np.hstack((n_data[j][i], use_ampd, m_fre_closest_value, m_mode_number, m_phase_four_signal.data[i, j]))
        n_most_signal.data = n_data_new[0]
        n_sec_signal.data = n_data_new[1]
        n_third_signal.data = n_data_new[2]
        n_forth_signal.data = n_data_new[3]
        return n_most_signal,n_sec_signal,n_third_signal,n_forth_signal

    def csd_resolve(self, n_j_data_i, pol03_data_i, pol04_data_i):
        """
        重新做一下csd然后找到对应的值去填充
        """
        f,csd_resolve  = sig.csd( pol03_data_i,pol04_data_i, fs=self.pol_samperate, window='hann', nperseg=self.n_pad,
                            scaling='density')
        # 找到 CSD 幅值最大处的索引
        max_idx_03 = np.argmax(np.abs(csd_resolve))
        # 获取最大幅值对应的频率
        m_fre_closest_index = np.abs(np.array(f) - n_j_data_i).argmin()
        # 使用索引找到最接近的数值
        m_fre_closest_value = f[m_fre_closest_index]
        phase_csd = (np.angle(csd_resolve) * 180 / np.pi)[m_fre_closest_index]
        mode_number=phase_csd/self.real_angle
        return mode_number,m_fre_closest_value

    def find_fre_mode(self, m_fre_i, n_j_data_i):
        """
        找到对应的m_j_data_i的模数和频率
        """
        # 找到最接近值的索引
        m_fre_closest_index = np.abs(np.array(m_fre_i) - n_j_data_i).argmin()
        # 使用索引找到最接近的数值
        m_fre_closest_value = m_fre_i[m_fre_closest_index]
        # 检查一下误差值不能大于等于1K，如果超过，那么需要重新做一下csd然后找到对应的值去填充
        if np.abs(m_fre_closest_value - n_j_data_i) >= 600:
            m_fre_closest_index = -1
            m_fre_closest_value = -1
        return m_fre_closest_index, m_fre_closest_value