#  -*-  coding: utf-8  -*-
from jddb.processor import Signal, Shot, ShotSet, BaseProcessor
import numpy as np
from copy import deepcopy

class DivertorSafetyFactorProcessor(BaseProcessor):
    "适用于Jtext的绘制撕裂模幅值,在23秋季与24春季minor_radius_a=0.22, major_radius_R=1.05"
    def __init__(self,minor_radius_a=0.22, major_radius_R=1.05):
        super().__init__(minor_radius_a=minor_radius_a, major_radius_a=major_radius_R)
        self.MU_0 = 4 * np.pi * 10 ** (-7)
        self.minor_radius_a=minor_radius_a
        self.major_radius_R =major_radius_R
    def transform(self,  *signal: Signal):
        # #signal[0]:Ip
        # #signal[1]:qa#在这一步计算一次qa_raw
        ip_signal = deepcopy(signal.__getitem__(0))
        bt_signal = deepcopy(signal.__getitem__(1))
        qa_signal = deepcopy(signal.__getitem__(1))
        K = 2 * np.pi * self.minor_radius_a ** 2 / (self.MU_0 * 1000 * self.major_radius_R)
        qa_signal.data=K * bt_signal.data / ip_signal.data
        # if mir_raw_signal.parent.labels["ERTUsed"]==1:
        return qa_signal

