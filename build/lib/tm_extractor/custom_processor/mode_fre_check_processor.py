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

    def __init__(self, n_real_angle: float, m_real_angle: float, down_number=1, coherence_th=0.95):
        """
        Initializes the ModeFreCheck processor with the provided real angle, down-sampling factor,
        and coherence threshold.

        Args:
            real_angle (float): The real angle for mode checking.
            down_number (int, optional): The down-sampling factor for the data.
            coherence_th (float, optional): The coherence threshold for mode validation.
        """
        super().__init__(n_real_angle=n_real_angle, m_real_angle=m_real_angle, down_number=down_number, coherence_th=coherence_th)
        self.down_number = down_number
        self.n_real_angle = n_real_angle
        self.m_real_angle = m_real_angle
        self.coherence_th = coherence_th

    def csd_compute_n(self, chip_data1, chip_data2, fre_missing, sample_rate):
        number_of_a_window = len(chip_data1)
        (f, csd) = sig.csd(chip_data1, chip_data2, fs=sample_rate, window='hann',
                           nperseg=int(number_of_a_window / self.down_number), scaling='density')

        # Find the index corresponding to the missing frequency
        indices = np.where(f == fre_missing)
        log_abs_csd= np.log(np.abs(csd[indices[0][0]]))
        phase_csd = np.angle(csd[indices[0][0]]) * 180 / np.pi
        mode = phase_csd / self.n_real_angle
        return np.array([mode,
                        1000 * log_abs_csd/ fre_missing,
                        log_abs_csd,
                        fre_missing,
                        phase_csd,
                        1
                        ]).reshape(1, 6)

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
        mode = phase_csd / self.m_real_angle

        # Check if the mode is valid based on coherence threshold
        if mode > 0 and coherence[indices[0][0]] > self.coherence_th:
            return True
        else:
            return False

    def missing_in_fre(self, check_fre, complete_fre):
        """
        Identifies the frequencies present in `complete_fre` but missing in `check_fre`.

        Args:
            check_fre (list): The list of frequencies to check.
            complete_fre (list): The list of all frequencies.

        Returns:
            list: A list of missing frequencies.
        """
        missing_fre = []
        for fre in complete_fre:
            if fre not in check_fre:
                missing_fre.append(fre)
        return missing_fre


    def is_approx_multiple(self,i, n_fre, m_fre):
        # Set the error margin based on the value of A
        if m_fre <= 6000:
            error_margin = 600
        else:
            error_margin = 1000

        # Calculate the ratio of B to A
        ratio = n_fre / m_fre

        # Check if B is approximately a multiple of A (at least 2 times A) and within the error margin
        # print(f"i {i}, r: {ratio}")
        if ratio >= 2 and abs(n_fre - m_fre * round(ratio)) <= error_margin:
            return True
        else:
            return False


    def transform(self, *signal: Signal) -> Tuple[Signal]:
        """
            Transforms the input signals by checking the validity of modes at specific frequencies
            using cross-spectral density (CSD) analysis. If an invalid mode is detected, the corresponding
            signal is set to zero.

            Args:
                *signal (Signal): A tuple of Signal objects representing different modes and their data.

            Returns:
                Tuple[Signal]: A tuple of transformed signals where invalid modes are set to zero.
        """
        # Deepcopy the input signals to preserve original data
        n_signal = [deepcopy(signal.__getitem__(i_signal)) for i_signal in range(4)]
        m_signal= [deepcopy(signal.__getitem__(i_signal)) for i_signal in range(4, 8)]
        sample_rate = signal.__getitem__(8).attributes['OriginalSampleRate']
        n_csd_signal = [deepcopy(signal.__getitem__(i_signal)) for i_signal in range(8, 10)]
        m_csd_signal = [deepcopy(signal.__getitem__(i_signal)) for i_signal in range(10, 12)]

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

        for i in range(1,len(signal.__getitem__(0).data)):  # For each time point#1
            # 如果是第一个频率 跟后面一个频率比较
            #  只要不是全零，检查一下第一项中频率跨越比较大的，
            if not np.all(n_signal[0].data[i] == 0) and not np.all(n_signal[0].data[i - 1] == 0):
                if self.is_approx_multiple(i,n_fre=n_signal[0].data[i , 3], m_fre=n_signal[0].data[i- 1, 3]):
                    # 如果跨越比较大，查看一下m中连续的前两个、后两个频率是否没有跳跃
                    if not np.all(n_signal[0].data[i] == 0) and not np.all(m_signal[0].data[i] == 0):
                        if self.is_approx_multiple(i,n_fre=n_signal[0].data[i, 3], m_fre=m_signal[0].data[i, 3]):
                            if abs(n_signal[1].data[i- 1, 3]-m_signal[0].data[i, 3])<=1000:
                                n_signal[0].data[i]=n_signal[1].data[i]
                                n_signal[1].data[i] = n_signal[2].data[i]
                                n_signal[2].data[i] = n_signal[3].data[i]
                                n_signal[3].data[i] = (np.zeros(len(n_signal[0].data[i])))
                            else:
                                n_signal[0].data[i]=self.csd_compute_n(n_csd_signal[0].data[i], n_csd_signal[1].data[i], m_signal[0].data[i, 3], sample_rate)

        return n_signal[0], n_signal[1], n_signal[2], n_signal[3]
