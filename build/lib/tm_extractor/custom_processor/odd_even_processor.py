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

class EvenOddProcessor(BaseProcessor):
    """
    给定积分范围，返回积分数值以及求B的数值，（250k的采样）
    降采样，做时间切片
    傅里叶分解

    """
    def __init__(self):
        super().__init__()


    def transform(self, *signal: Signal):
        ##signal[0]:LFS MA_POLA_P06
        # signal[1]:HFS MA_POLB_P06
        hfs_signal = deepcopy(signal.__getitem__(0))
        lfs_signal = deepcopy(signal.__getitem__(1))
        even_signal = deepcopy(signal.__getitem__(0))
        odd_signal = deepcopy(signal.__getitem__(0))
        odd_signal.data =( lfs_signal.data-hfs_signal.data)/2
        even_signal.data = (lfs_signal.data + hfs_signal.data) / 2
        return odd_signal, even_signal
