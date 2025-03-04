from copy import deepcopy
from typing import Tuple, List
import numpy as np
from jddb.processor import Signal
from scipy import signal as sig
from jddb.processor import *

class ModeFreCheck(BaseProcessor):
    """
    A processor for checking the mode frequencies using cross-spectral density (CSD) analysis.
    This class performs operations such as cross-power spectral density calculation,
    coherence checking, and identifying missing frequencies in the data.

    Args:
        real_angle (float): The real angle for mode checking. Default is 7.5.
        down_number (int, optional): The down-sampling factor for the data. Default is 1.
        coherence_th (float, optional): The coherence threshold for mode validation. Default is 0.95.
    """

    def __init__(self, real_angle: float, down_number=1, coherence_th=0.95):
        """
        Initializes the ModeFreCheck processor with the provided real angle, down-sampling factor,
        and coherence threshold.

        Args:
            real_angle (float): The real angle for mode checking.
            down_number (int, optional): The down-sampling factor for the data.
            coherence_th (float, optional): The coherence threshold for mode validation.
        """
        super().__init__(down_number=down_number, real_angle=real_angle, coherence_th=coherence_th)
        self.down_number = down_number
        self.real_angle = real_angle
        self.coherence_th = coherence_th

    def csd_check_mode(self, chip_data1, chip_data2, fre_missing, sample_rate):
        """
        Calculates the cross-spectral density (CSD) and coherence between two signals, and checks
        if the mode at the specified frequency is valid.

        Args:
            chip_data1 (np.ndarray): The first signal data.
            chip_data2 (np.ndarray): The second signal data.
            fre_missing (float): The frequency to check.
            sample_rate (float): The sample rate of the signals.

        Returns:
            bool: True if the mode is valid, False otherwise.
        """
        # Calculate the cross-spectral density (CSD) and coherence
        number_of_a_window = len(chip_data1)
        (f, csd) = sig.csd(chip_data1, chip_data2, fs=sample_rate, window='hann',
                           nperseg=int(number_of_a_window / self.down_number), scaling='density')
        (f_coherence, coherence) = sig.coherence(chip_data1, chip_data2, fs=sample_rate,
                                                 window='hann', nperseg=int(number_of_a_window / self.down_number))

        # Find the index corresponding to the missing frequency
        indices = np.where(f == fre_missing)
        phase_csd = np.angle(csd[indices[0][0]]) * 180 / np.pi
        mode = phase_csd / self.real_angle


    def transform(self, *signal: Signal) -> Tuple[Signal]:

        # Deepcopy the input signals to preserve original data
        n_signal = [deepcopy(signal.__getitem__(i_signal)) for i_signal in range(4)]
        sample_rate = signal.__getitem__(8).attributes['OriginalSampleRate']
        m_signal = [deepcopy(signal.__getitem__(i_signal)) for i_signal in range(8, 10)]

        # Check modes for each time point in the signal data
        for i in range(len(signal.__getitem__(0).data)):  # For each time point
            n_signal_fre = [signal.__getitem__(i_signal).data[i, 3] for i_signal in range(4)]
            m_signal_fre = [signal.__getitem__(i_signal).data[i, 3] for i_signal in range(4, 8)]

            # Find frequencies that are in n_signal but missing in m_signal
            m_missing_fre = self.missing_in_fre(m_signal_fre, n_signal_fre)
            error_fre = []

            # Check the missing frequencies in m_signal for validity
            if len(m_missing_fre) != 0:
                for fre_missing in m_missing_fre:
                    is_mode = self.csd_check_mode(m_csd_signal.__getitem__(0).data[i, :],
                                                  m_csd_signal.__getitem__(1).data[i, :], fre_missing, sample_rate)
                    if not is_mode:
                        error_fre.append(fre_missing)
            # If any error frequencies are found, set the corresponding data in n_signal to zero
            if len(error_fre) != 0:
                for element in error_fre:
                    index = n_signal_fre.index(element)
                    n_signal[index].data[i, :] = np.zeros(len(n_signal[index].data[i, :]))
        return n_signal[0], n_signal[1], n_signal[2], n_signal[3]


    # 如果跨越比较大，查看一下m中连续的前两个、后两个频率是否没有跳跃
    # 如果没有跳跃，找一下n中有没有频率是相近的，与m误差大约为1000Hz以内
    # 误差不大的就把m所有项往前移动
    # 误差大的就对n中的频率进行重置，按照对应频率计算出其他值
