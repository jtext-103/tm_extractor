import os
from copy import deepcopy
from typing import Tuple, List
import numpy as np
from jddb.processor import Signal
from matplotlib import pyplot as plt, gridspec
from scipy import signal as sig
from jddb.processor import *

class PlotModeAmpProcessor(BaseProcessor):
    """
           -------
    Args:   绘制撕裂模式的幅值图
           -------
    """

    def __init__(self,mode_plot_save_file:str, mn_list:List[int]):
        super().__init__(mode_plot_save_file=mode_plot_save_file, mn_list=mn_list)
        self.mode_plot_save_file = mode_plot_save_file
        self.mn_list = mn_list
        self.ensure_directory_exists()
        self.shot_no = 0

    def ensure_directory_exists(self):
        os.makedirs(self.mode_plot_save_file, exist_ok=True)  # exist_ok=True 避免已存在时报错

    def plot_shot_amp_fre(self, mir_signal_raw, mir_signal_inte, mir_signal_even, mir_signal_odd, mir_signal_lfs, qa_signal,
                  mn_amp_signal, mn_fre_signal):
        colot_list = ['blue', 'darkred', 'green', 'black', 'yellow', 'purple', 'orange', 'pink', 'brown', 'gray']
        # Create a figure with a customized gridspec for positioning
        fig = plt.figure(figsize=(20, 14))
        gs = gridspec.GridSpec(6, 2, width_ratios=[1, 0.05])  # 3 rows, 2 columns (second column for colorbars)
        # 设置全局字体大小
        plt.rcParams.update({'font.size': 12})  # 设置为您希望的字体大小
        # Create the first subplot (spectrogram of \\MA_POLB_P06 signal)
        time1 = mir_signal_raw.time
        data1 = mir_signal_raw.data
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.specgram(
            data1,
            Fs=mir_signal_raw.attributes["SampleRate"],
            cmap='hsv')
        # ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Frequency (Hz)')
        ax1.set_title('Spectrogram of LFS Raw Signal')

        # Create the second subplot (spectrogram of \\B_LFS_Inte signal)
        ax2 = fig.add_subplot(gs[1, 0])
        data2 = mir_signal_inte.data
        ax2.specgram(
            data2,
            Fs=mir_signal_inte.attributes["SampleRate"],
            cmap='hsv')
        # ax2.set_xlabel('Time (s)')
        # ax2.set_ylim(0, int(mir_signal_raw.attributes["SampleRate"]/2))
        ax2.set_ylabel('Frequency (Hz)')
        ax2.set_title('Spectrogram of \\B_LFS_Inte Signal')

        # Create the third subplot (spectrogram of lfs odd even amp signal)
        # 继续在 ax3 中绘制图形
        ax3 = fig.add_subplot(gs[2, 0])
        ax3.plot(mir_signal_even.time, mir_signal_even.data[:, 1], color='#D1A6F7', label='Even Amp')  # 浅蓝
        ax3.plot(mir_signal_odd.time, mir_signal_odd.data[:, 1], color='#89CFF0', label='Odd Amp')  # 略深的浅蓝
        ax3.plot(mir_signal_lfs.time, mir_signal_lfs.data[:, 2], color='#1f77b4', label='Lfs Amp')  # 深蓝
        ax3.axhline(y=2, color='green', linestyle='--', label='y=2')
        # 创建 ax4 子图来显示图例
        ax9 = fig.add_subplot(gs[2, 1])
        # 隐藏 ax4 子图的坐标轴
        ax9.axis('off')
        # 在 ax4 中绘制图例，位置设置在子图的中心
        ax9.legend(handles=ax3.lines, loc='center')

        ax3.set_ylabel('Amplitude (Gs)')
        ax3.set_title('Spectrogram of Inte Signal')

        # Create the fourth subplot (time series of Qa)
        # Create the fourth subplot (time series of Qa)
        ax4 = fig.add_subplot(gs[3, 0])
        ax4.plot(qa_signal.time,
                 qa_signal.data, color='blue')
        ax4.set_ylim(0, 8)
        ax4.set_ylabel('Qa')
        ax4.set_title('Time Series of Qa Signal')
        gray_start = 0.9  # 起始灰度（最浅）
        gray_end = 0.2  # 终止灰度（最深）
        n = 7  # 线的数量（可以任意调整）
        step = (gray_start - gray_end) / (n - 1)  # 自动计算步长
        for i in range(1, n + 1):
            gray_level = str(gray_start - (i - 1) * step)
            ax4.axhline(y=i, color=gray_level, linestyle='--')  # 不设置 label

        # Create the third subplot (time series of mode)
        # 在子图上绘制
        ax5 = fig.add_subplot(gs[4, 0])
        # 保存fre曲线的图例线对象
        line_objects = []  # 用于存储fre曲线
        for index, (m, n) in enumerate(self.mn_list):
            time_mn = mn_amp_signal.__getitem__(index).time
            data_mn_amp = mn_amp_signal.__getitem__(index).data
            line, = ax5.plot(time_mn, data_mn_amp, color=colot_list[index], label=f'{m}{n} Amp')
            line_objects.append(line)  # 保存每个fre线对象用于图例

        # 添加水平线，并给它们添加标签
        line_2 = ax5.axhline(y=2, color='lightgray', linestyle='--', label='y=2')
        line_20 = ax5.axhline(y=20, color='red', linestyle='--', label='y=20')
        line_10 = ax5.axhline(y=10, color='green', linestyle='--', label='y=10')

        # 合并所有图例句柄
        handles_left = [line_2, line_20, line_10]  # 水平线图例
        handles_right = line_objects  # fre曲线图例
        # 创建 ax7 子图来显示图例
        ax11 = fig.add_subplot(gs[4, 1])
        # 隐藏 ax7 子图的坐标轴
        ax11.axis('off')
        # 合并所有图例句柄
        handles_left = [line_2, line_20, line_10]  # 水平线图例
        handles_right = line_objects  # fre曲线图例
        # 在 ax7 中绘制图例，位置设置在子图的中心
        ax11.legend(handles=handles_left + handles_right, loc='center', ncol=1)

        # 现在 ax5 不需要再绘制图例，只需绘制数据曲线和水平线
        ax5.set_ylabel('Amplitude (Gs)', labelpad=15)
        ax5.set_title('Time Series of Amp Signal')


        ax6 = fig.add_subplot(gs[5, 0])

        # 保存fre曲线的图例线对象
        line_objects = []  # 用于存储fre曲线
        for index, (m, n) in enumerate(self.mn_list):
            time_mn = mn_fre_signal.__getitem__(index).time
            data_mn_fre = mn_fre_signal.__getitem__(index).data
            line, = ax6.plot(time_mn, data_mn_fre, color=colot_list[index], label=f'{m}{n} Fre')
            line_objects.append(line)  # 保存每个fre线对象用于图例

        # 添加水平线，并给它们添加标签
        line_4k = ax6.axhline(y=4000, color='lightgray', linestyle='--', label='y=4k')
        line_10k = ax6.axhline(y=10000, color='red', linestyle='--', label='y=10k')
        line_15k = ax6.axhline(y=15000, color='green', linestyle='--', label='y=15k')
        line_20k = ax6.axhline(y=20000, color='gray', linestyle='--', label='y=20k')

        # 合并所有图例句柄
        handles_left6 = [line_4k, line_10k, line_15k, line_20k]  # 水平线图例
        handles_right6 = line_objects  # fre曲线图例
        # 创建 ax7 子图来显示图例
        ax11 = fig.add_subplot(gs[5, 1])
        # 隐藏 ax7 子图的坐标轴
        ax11.axis('off')
        # 在 ax7 中绘制图例，位置设置在子图的中心
        ax11.legend(handles=handles_left6 + handles_right6, loc='center', ncol=1)
        # 设置y轴标签并放在右侧
        ax6.set_xlabel('Time (s)')
        ax6.set_ylabel('Frequency (Hz)', labelpad=15)
        ax6.set_title('Time Series of fre Signal')
        # 调整子图布局，确保刻度标签可见
        plt.subplots_adjust(right=0.8)  # 调整右侧边距，确保图例不覆盖标签

        # Create new subplots for the colorbars
        ax7 = fig.add_subplot(gs[0, 1])  # Colorbar for the first spectrogram
        ax8 = fig.add_subplot(gs[1, 1])  # Colorbar for the second spectrogram

        # Create the colorbars and assign them to the new axes
        cbar1 = fig.colorbar(ax1.images[0], cax=ax7, label='Intensity (dB)')
        cbar2 = fig.colorbar(ax2.images[0], cax=ax8, label='Intensity (dB)')
        # Adjust layout to align x-axes vertically and avoid overlap
        plt.subplots_adjust(wspace=3, hspace=0.001)  # 调整子图间的水平和垂直间距
        fig.align_ylabels()
        fig.tight_layout()
        # 设置图形大小的比例（高：宽=30:40）

        mode_tags = "_".join(f"{x}{y}" for x, y in self.mn_list)
        file_name = f"spectrogram_{mode_tags}_{self.shot_no}.png"  # 拼接到文件名
        # Show the plot
        plt.savefig(os.path.join(self.mode_plot_save_file, file_name))


    def transform(self, *signal: Signal) -> Signal:
        # "\\MA","\\B_LFS_Inte","\\new_B_even_n_most_th","\\new_B_odd_n_most_th","\\new_B_th_nm_most_judge","\\qa_1k"
        # signal[0]: "\\MA_POLB_P06"
        # signal[1]: "\\B_LFS_Inte"
        # signal[2]: "\\new_B_even_n_most_th"
        # signal[3]: "\\new_B_odd_n_most_th"
        # signal[4]: "\\new_B_th_nm_most_judge"
        # signal[5]: "\\qa_1k"
        # signal[6:]: f"\\{m}{n}_amp"
        mir_signal_raw = deepcopy(signal.__getitem__(0))
        mir_signal_inte = deepcopy(signal.__getitem__(1))
        mir_signal_even = deepcopy(signal.__getitem__(2))
        mir_signal_odd = deepcopy(signal.__getitem__(3))
        mir_signal_lfs = deepcopy(signal.__getitem__(4))
        qa_signal = deepcopy(signal.__getitem__(5))

        mn_amp_signal = [deepcopy(signal.__getitem__(i)) for i in range(6, 6+len(self.mn_list))]
        mn_fre_signal = [deepcopy(signal.__getitem__(i)) for i in range(6 + len(self.mn_list),len(signal))]
        self.shot_no = signal.__getitem__(0).parent.shot_no

        self.plot_shot_amp_fre(mir_signal_raw, mir_signal_inte, mir_signal_even, mir_signal_odd, mir_signal_lfs, qa_signal,mn_amp_signal, mn_fre_signal)
        return qa_signal












