# -*-  coding:utf-8  -*-
import numpy as np
from jddb.processor import Signal, BaseProcessor
from copy import deepcopy


class AmpIsTearingProcessor(BaseProcessor):
    def __init__(self, amp_threshold_ratio=0.5, f_upper_threshold=800):
        super().__init__(amp_threshold_ratio=amp_threshold_ratio, f_upper_threshold=f_upper_threshold)
        self.threshold_ratio =  amp_threshold_ratio
        self.f_upper_threshold= f_upper_threshold
    def transform(self, signal: Signal):
        # signal[1]: m_most_max_all
        B_LFS_Inte_slice_signal = deepcopy(signal)
        is_tearing_signal = deepcopy(signal)
        recover, is_tearing = self.is_tearing_threshold(B_LFS_Inte_slice_signal.data)
        # recover_data,is_tearing_near_time_data = self.is_tearing_near_time( recover_data, f_upper=800)
        B_LFS_Inte_slice_signal.data=recover
        is_tearing_signal.data=is_tearing
        return B_LFS_Inte_slice_signal,is_tearing_signal

    def is_tearing_threshold(self, B_LFS_Inte_slice_data):
        """
        判断撕裂是否发生
        :param B_LFS_Inte_slice: B_LFS_Inte_slice
        :return: is_tearing: 是否发生撕裂
        """
        is_tearing = np.zeros(len(B_LFS_Inte_slice_data))
        recover = deepcopy(B_LFS_Inte_slice_data)
        for i in range(len(B_LFS_Inte_slice_data)):
            if B_LFS_Inte_slice_data[i][1] >= self.threshold_ratio and B_LFS_Inte_slice_data[i][2] > self.f_upper_threshold :
                is_tearing[i] = int(1)
            elif B_LFS_Inte_slice_data[i][1] ==0 and B_LFS_Inte_slice_data[i][2] == 0:
                is_tearing[i] = -1
            else:
                is_tearing[i] = int(0)
                recover[i] = np.array([0,0,0])

        return recover,is_tearing