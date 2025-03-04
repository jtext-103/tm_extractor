#  -*-  coding: utf-8  -*-
import copy
import math
import os
from jddb.processor import Signal, Shot, ShotSet, BaseProcessor
from typing import Optional, List
from scipy.fftpack import fft
from copy import deepcopy
import numpy as np
from copy import deepcopy

class InteProcessor(BaseProcessor):
    """
    给定积分范围，返回积分数值以及求B的数值，（250k的采样）
    降采样，做时间切片
    傅里叶分解

    """
    def __init__(self,NS=0.0689):# cm^2
        super().__init__(NS=NS)

        self.NS = NS

    def transform(self, *signal: Signal):
        ##signal[0]:LFS MA_POLA_P06
        # signal[1]:HFS MA_POLB_P06
        hfs_signal = deepcopy(signal.__getitem__(0))
        lfs_signal = deepcopy(signal.__getitem__(1))
        NS=self.NS# cm^2

        # dt=1/lfs_signal.attributes["OriginalSampleRate"]
        dt = 1 / lfs_signal.attributes["SampleRate"]
        lfs_signal.data = 1e4 * dt * np.cumsum(lfs_signal.data) / NS#Gs
        hfs_signal.data = 1e4 * dt * np.cumsum(hfs_signal.data) / NS#Gs

        return hfs_signal, lfs_signal
