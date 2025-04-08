from scipy.fftpack import fft
from copy import deepcopy
from jddb.processor import Signal, BaseProcessor
import numpy as np
class IsTearingJudgeProcessor(BaseProcessor):
    """
       给定频率,判断B_tehta是否大于阈值,如果大于阈值,则认为是撕裂模,如果不是撕裂模，其频率标注为-1
    """

    def __init__(self):  # pad 表示f精度，精度为 1/(fs*pad),f 的长度为 pad*fs/2
        super().__init__()

    def transform(self, *signal: Signal):
        # #signal[0]:m_most_max_all
        # #signal[1]:m_sec_max_all
        # #signal[2]:m_third_max_all
        # #signal[3]:m_forth_max_all
        is_tearing_signal=deepcopy(signal.__getitem__(0))
        m_most_all_signal = deepcopy((signal.__getitem__(0)))
        # m_sec_all_signal = deepcopy((signal.__getitem__(1)))
        # m_third_all_signal = deepcopy((signal.__getitem__(2)))
        # m_forth_all_signal = deepcopy((signal.__getitem__(3)))
        is_tearing_data=np.zeros((len(m_most_all_signal.data)))
        for i in range(len(m_most_all_signal.data)):
            for i_th in range(len(signal)):
                if signal.__getitem__(i_th).data[i,0] > 0:
                    is_tearing_data[i]=1
                    break
        is_tearing_signal.data=is_tearing_data
        return is_tearing_signal
