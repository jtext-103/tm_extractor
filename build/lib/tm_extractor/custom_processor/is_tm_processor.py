# -*-  coding:utf-8  -*-
import numpy as np
from typing import Tuple
from jddb.processor import Signal, BaseProcessor
from copy import deepcopy
class IsTMProcessor(BaseProcessor):
    def __init__(self):
        super().__init__()

    def transform(self, *signal: Signal) -> Signal:
        most_signal = deepcopy(signal.__getitem__(0))
        is_tm_signal = deepcopy(signal.__getitem__(0))
        is_tm_data = np.zeros((len(most_signal.data)),dtype=int)
        for i in range(len(most_signal.data)):
            data_th_array =np.array([signal.__getitem__(th).data[i] for th in range(len(signal))])
            for index_th in range(len(data_th_array)):
                if  data_th_array[index_th, 2]>0:
                    is_tm_data[i] = 1
                    break
        is_tm_signal.data = is_tm_data
        return is_tm_signal
