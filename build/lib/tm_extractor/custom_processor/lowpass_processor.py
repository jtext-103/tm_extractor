#  -*-  coding: utf-8  -*-
import math
from typing import Tuple
from jddb.processor import Signal, Shot, ShotSet, BaseProcessor
from typing import Optional, List
from scipy.fftpack import fft
from copy import deepcopy
from scipy.interpolate import interp1d
import pandas as pd
import numpy as np
from scipy import signal as sig
from copy import deepcopy

import matplotlib.pyplot as plt

class LowPassProcessor(BaseProcessor):
    def __init__(self, cutoff_freq: float):
        super().__init__(cutoff_freq=cutoff_freq)
        self.cutoff_freq = cutoff_freq # 截止频率

    def transform(self, signal: Signal) -> Signal:

        fs = signal.attributes["SampleRate"]
        # 设计低通滤波器

        nyquist_freq = 0.5 * fs
        normal_cutoff = self.cutoff_freq / nyquist_freq

        b, a = sig.butter(4, normal_cutoff, btype='low', analog=False)
        # 设计倒相滤波器

        # 应用滤波器

        filtered_signal_data = sig.lfilter(b, a, signal.data)
        # filtered_signal = signal_input.lfilter(b, a, smoothed_signal)
        filtered_signal_inverse_data = sig.lfilter(b, a, np.flip(filtered_signal_data))
        # 对倒相滤波器的结果进行倒序，以抵消相位延迟
        filtered_signal_phase_compensated_data= np.flip(filtered_signal_inverse_data)
        filtered_signal_phase_compensated_data = np.round(filtered_signal_phase_compensated_data, 6)


        return Signal(data=filtered_signal_phase_compensated_data, attributes=signal.attributes)
