#  -*-  coding: utf-8  -*-
import os
from matplotlib import gridspec
from jddb.processor import Signal, BaseProcessor
import numpy as np
from copy import deepcopy
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')  # 以下绘制的图形不会在窗口显示
class PlotShotProcessor(BaseProcessor):
    "适用于Jtext的绘制撕裂模幅值,在23秋季与24春季minor_radius_a=0.22, major_radius_R=1.05"
    def __init__(self,plt_save:str,minor_radius_a=0.22, major_radius_a=1.05):
        super().__init__(plt_save=plt_save,minor_radius_a=minor_radius_a, major_radius_a=major_radius_a)
        self.plt_save=plt_save
        self.MU_0 = 4 * np.pi * 10 ** (-7)
        self.minor_radius_a=minor_radius_a
        self.major_radius_R =major_radius_a
    def transform(self,  *signal: Signal):
        # #signal[0]:MA_POLB_P06
        # #signal[1]:B_LFS_Inte
        # #signal[2]:Ip
        # #signal[3]:qa#在这一步计算一次qa_raw
        mir_raw_signal = deepcopy(signal.__getitem__(0))
        shot_no = mir_raw_signal.parent.shot_no

        mir_inte_signal = deepcopy(signal.__getitem__(1))
        ip_signal = deepcopy(signal.__getitem__(2))
        bt_signal = deepcopy(signal.__getitem__(3))
        qa_signal = deepcopy(signal.__getitem__(3))
        most_signal = deepcopy(signal.__getitem__(4))
        qa_signal=deepcopy(signal.__getitem__(5))
        # K = 2 * np.pi * self.minor_radius_a ** 2 / (self.MU_0 * 1000 * self.major_radius_R)
        # qa_signal.data=K * bt_signal.data / ip_signal.data
        # if mir_raw_signal.parent.labels["ERTUsed"]==1:
        if self.is_plot:
            self.plot_shot(mir_raw_signal,mir_inte_signal, ip_signal, qa_signal,most_signal, shot_no)
        return qa_signal
        # return Signal(data=smoothed_signal_data, attributes=signal.attributes)
    def plot_shot(self,mir_raw_signal,mir_inte_signal, ip_signal, qa_signal,most_signal, shot_no):
            downtime = max([len(mir_raw_signal.time), len(mir_inte_signal.time), len(ip_signal.time), len(qa_signal.time)])
            # 绘制图形并存入指定路径
            # 图片保存路径
            dir_name = os.path.join(self.plt_save, f'shot_{shot_no} ' + '.jpg')
            tags = ["mir_raw", "mir_Inte", "ip", "qa", "most"]
            fig = plt.figure(figsize=(10, 10))
            gs = gridspec.GridSpec(5, 2, width_ratios=[1, 0.05])  # 3 rows, 2 columns (second column for colorbars)

            # Create the first subplot (spectrogram of \\MA_POLB_P06 signal)
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.specgram(
                # shot.get_signal("\\MA_POLB_P06").data,
                # Fs=shot.get_signal("\\MA_POLB_P06").attributes["SampleRate"],
                mir_raw_signal.data,
                Fs=mir_raw_signal.attributes["SampleRate"],
                cmap='hsv')
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Frequency (Hz)')
            ax1.set_title('Spectrogram of \\MA_POLB_P06 Signal')

            # Create the second subplot (spectrogram of \\B_LFS_Inte signal)
            ax2 = fig.add_subplot(gs[1, 0])
            ax2.specgram(
                # shot.get_signal("\\B_LFS_Inte").data,
                #          Fs=shot.get_signal("\\B_LFS_Inte").attributes["SampleRate"],
                mir_inte_signal.data,
                Fs=mir_inte_signal.attributes["SampleRate"],
                cmap='hsv')
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Frequency (Hz)')
            ax2.set_title('Spectrogram of B_LFS_Inte Signal')

            # Create the third subplot (time series of qa_signal)
            ax3 = fig.add_subplot(gs[2, 0])
            ax3.plot(most_signal.time,
                     most_signal.data[:, 1], color='blue', label=f'max amp {shot_no}')
            ax3.set_ylim(0, max( most_signal.data[:, 1])/0.7)
            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('Amplitude')
            ax3.set_title('Time Series of max amp Signal')
            ax3.axhline(y=20, color='red', linestyle='--', label='y=20')
            ax3.axhline(y=10, color='green', linestyle='--', label='y=10')
            ax3.legend()

            ax4 = fig.add_subplot(gs[3, 0])
            ax4.plot(ip_signal.time,
                     ip_signal.data, color='blue', label='Even Mode')
            # ax4.set_ylim(0, 30)
            ax4.set_xlabel('Time (s)')
            ax4.set_ylabel('Amplitude')
            ax4.set_title('Time Series of IP Signal')
            ax4.legend()

            # Create the third subplot (time series of qa_signal)
            ax5 = fig.add_subplot(gs[4, 0])
            ax5.plot(most_signal.time,most_signal.data[:, 0], color='blue', label='max amp Mode')
            ax5.plot(qa_signal.time,qa_signal.data, color='darkred', label='Qa')
            ax5.set_ylim(0, max(max(qa_signal.data),max(most_signal.data[:, 0]))/0.8)
            ax5.set_xlabel('Time (s)')
            ax5.set_ylabel('Amplitude')
            ax5.set_title('Time Series of QA Signal')
            ax5.axhline(y=1, color='red', linestyle='--', label='y=1')
            ax5.axhline(y=2, color='green', linestyle='--', label='y=2')
            ax5.axhline(y=3, color='red', linestyle='--', label='y=3')
            ax5.axhline(y=4, color='green', linestyle='--', label='y=4')
            ax5.legend()

            # Create new subplots for the colorbars
            ax6 = fig.add_subplot(gs[0, 1])  # Colorbar for the first spectrogram
            ax7 = fig.add_subplot(gs[1, 1])  # Colorbar for the second spectrogram

            # Create the colorbars and assign them to the new axes
            cbar1 = fig.colorbar(ax1.images[0], cax=ax6, label='Intensity (dB)')
            cbar2 = fig.colorbar(ax2.images[0], cax=ax7, label='Intensity (dB)')

            # Adjust layout to align x-axes vertically and avoid overlap
            plt.tight_layout()
            plt.savefig(dir_name)
            # Show the plot
            plt.close()
