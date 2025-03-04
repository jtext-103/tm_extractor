from copy import deepcopy
import numpy as np
from scipy import signal as sig
from jddb.processor import *

class M_mode_th(BaseProcessor):
    """
           -------
           用于计算模数
           data1，data2为两道Mirnov探针信号（尽可能近）
           chip_time为做互功率谱时的窗口时间长度，默认为5ms
           down_number为做互功率谱时的降点数（down_number = 1，即取了全时间窗的点，即FFT窗仅为一个），默认为8
           low_fre为所需最低频率，默认为100Hz
           high_fre为所需最高频率，默认为20kHz
           step_fre为选取最大频率时的步长，默认为3kHz（这个其实没用到）
           max_number为选取最大频率的个数，默认为3个
           # var_th为频率间的方差阈值，默认为1e-13这一项没了！！！
           real_angle为两道Mirnov探针间的极向空间角度差，默认为15°
           coherence_th为互相关系数阈值，默认为0.95
           -------
    """

    def __init__(self, real_angle, var_th: float, coherence_th: float, mode_th:float,chip_time=5e-3, down_number=1, low_fre=1e2,
                 high_fre=2e4, step_fre=3e3,
                 max_number=int(4)):
        # var_th=1e-13, coherence_th=0.95, real_angle=7.5
        # var_th=1e-20, coherence_th=0.9, real_angle=15
        # var_th=1e-13, coherence_th=0.9, real_angle=22.5
        ## var_th=1e-14, coherence_th=0.95, real_angle=22.5
        super().__init__(real_angle=real_angle, var_th=var_th, coherence_th=coherence_th, mode_th=mode_th,chip_time=chip_time,
                         down_number=down_number, low_fre=low_fre, high_fre=high_fre, step_fre=step_fre, max_number=max_number)
        self.chip_time = chip_time
        self.down_number = down_number
        self.low_fre = low_fre
        self.high_fre = high_fre
        self.step_fre = step_fre
        self.max_number = max_number
        self.real_angle = real_angle
        self.var_th = var_th
        self.coherence_th = coherence_th
        self.mode_th =mode_th
        # self.var_th = 0
        # self.coherence_th = -100
        # if real_angle == 7.5:
        #     self.var_th = 1e-13
        #     self.coherence_th = 0.95
        #     # self.var_th = 0
        #     # self.coherence_th = -100
        # elif real_angle == 15:
        #     self.var_th = 1e-20
        #     self.coherence_th = 0.9
        # elif real_angle == 22.5:
        #     self.var_th = 1e-13
        #     self.coherence_th = 0.9
        # else:
        #     self.var_th = var_th
        #     self.coherence_th = coherence_th
    def ampd(self, data):
        """
        实现AMPD算法
        :param data: 1-D numpy.ndarray
        :return: 波峰所在索引值的列表
        """
        p_data = np.zeros_like(data, dtype=np.int32)
        count = data.shape[0]
        arr_rowsum = []
        for k in range(1, count // 2 + 1):  # k is step

            row_sum = 0
            for i in range(k, count - k):  # i is index
                if data[i] > data[i - k] and data[i] > data[i + k]:
                    row_sum -= 1
            arr_rowsum.append(row_sum)
        min_index = np.argmin(arr_rowsum)
        max_window_length = min_index
        for k in range(1, max_window_length + 1):

            for i in range(k, count - k):
                if data[i] > data[i - k] and data[i] > data[i + k]:
                    p_data[i] += 1
        return np.where(p_data == max_window_length)[0]

    def transform(self, signal1: Signal, signal2: Signal):
        signal1 = deepcopy(signal1)
        signal2 = deepcopy(signal2)
        # signal2.data = -signal2.data
        # number_of_a_window = int(self.chip_time * signal1.attributes['OriginalSampleRate'])  # window length
        # number_of_windows = len(signal1.time) // number_of_a_window  # number of the chip in the window
        number_of_a_window = len(signal1.data[0])
        number_of_windows = len(signal1.data)
        number_of_f_low = int(self.low_fre * self.chip_time)  # lowest frequency
        number_of_f_high = int(self.high_fre * self.chip_time)  # Highest frequency
        number_of_f_step = int(self.step_fre * self.chip_time)  # select max_frequency length
        new_signal1 = Signal(data=np.empty(shape=[0, 6], dtype=float), attributes=dict())
        new_signal2 = Signal(data=np.empty(shape=[0, 6], dtype=float), attributes=dict())
        new_signal3 = Signal(data=np.empty(shape=[0, 6], dtype=float), attributes=dict())
        new_signal4 = Signal(data=np.empty(shape=[0, 6], dtype=float), attributes=dict())
        # signal1.prop_dict['SampleRate']为原采样率
        new_signal1.attributes = signal1.attributes
        new_signal2.attributes = signal1.attributes
        new_signal3.attributes = signal1.attributes
        new_signal4.attributes = signal1.attributes
        new_signal1.parent = signal1.parent
        new_signal2.parent = signal1.parent
        new_signal3.parent = signal1.parent
        new_signal4.parent = signal1.parent

        # slide——————换成slice_processor
        ###slice_time=5ms, slice与slice的采样率1000k
        for i in range(number_of_windows):  #####改成一个slice读取处理
            # if i ==207:
            #     a=1
            if_mode_var = 1
            # new_signal1.d
            chip_data1 = signal1.data[i]
            chip_data2 = signal2.data[i]
            """做互功率谱,看相关性,并取互功率谱幅值、相位"""
            # calculate cross spectral density
            # self.down_number = self.down_number/nperseg_padded
            (f, csd) = sig.csd(chip_data1, chip_data2, fs=signal1.attributes['OriginalSampleRate'], window='hann',
                               nperseg=int(number_of_a_window / self.down_number),
                               scaling='density')
            (f_coherence, coherence) = sig.coherence(chip_data1, chip_data2,
                                                     fs=signal1.attributes['OriginalSampleRate'],
                                                     window='hann',
                                                     nperseg=int(number_of_a_window/ self.down_number))
            #1000Hz以下相当于滤波
            csd[:5]=np.zeros(5)
            abs_csd = np.abs(csd)
            abs_csd[:5]=np.zeros(5)
            # log_abs_csd = 20 * np.log(abs_csd)
            log_abs_csd = 1e6 * abs_csd
            phase_csd = np.angle(csd) * 180 / np.pi
            """在信号相关系数阈值之上的angle存入数组，否则该数组为空"""
            if_mode_coherence = np.where(coherence > self.coherence_th, 1, 0)
            angle_csd=phase_csd
            # angle_csd = np.where(coherence > self.coherence_th, phase_csd, np.nan)
            """csd中选取一段频率，低频到高频，且做方差,这里就是选了10K以内的"""
            # abs_csd_chosen = np.abs(csd[number_of_f_low // down_number: number_of_f_high // down_number])
            abs_csd_chosen = np.abs(csd[int(number_of_f_low / self.down_number): int(number_of_f_high / self.down_number)])
            var_csd = np.var(np.abs(abs_csd_chosen))
            # 求互功率谱均值，若最大频率大于最小频率，且小于ch_th就证明存在明显的模式, 目前使用方差与互相关系数做约束
            # 判断是否有明显模式
            if var_csd < self.var_th:
                if_mode_var= 0
                # new_signal1.data = np.concatenate((new_signal1.data, np.zeros((1, 5), dtype=float)), axis=0)
                # new_signal2.data = np.concatenate((new_signal2.data, np.zeros((1, 5), dtype=float)), axis=0)
                # new_signal3.data = np.concatenate((new_signal3.data, np.zeros((1, 5), dtype=float)), axis=0)
                # new_signal4.data = np.concatenate((new_signal4.data, np.zeros((1, 5), dtype=float)), axis=0)
            if_csd_mode = if_mode_var * if_mode_coherence
            down_number_of_step = int(number_of_f_step / self.down_number)
            overall_index_of_max_csd_in_every_piece = []
            # max_csd_in_every_piece = []
            """csd分片段处理,滑动窗口每段找出一个最大值,并存其全局最大值索引"""
            # for j in range(int(len(abs_csd_chosen) / down_number_of_step)):  # 把csd分几个片段处理
            #     abs_csd_piece = np.abs(abs_csd_chosen[j * down_number_of_step:(j + 1) * down_number_of_step])
            #     max_csd_in_every_piece.append(max(abs_csd_piece))  # 滑动窗口每段找出一个最大值
            abs_csd_chosen_max_index = self.ampd(abs_csd_chosen)#找出所有最大值索引

            max_csd_in_every_piece = [abs_csd_chosen[i] for i in abs_csd_chosen_max_index]#找出索引对应的元素
            # if len(max_csd_in_every_piece)<=self.max_number:
            #     for i in range(len(self.max_number-max_csd_in_every_piece)):
            #         max_csd_in_every_piece = np.append(max_csd_in_every_piece)
            sorted_indices = np.argsort(max_csd_in_every_piece)[::-1]#元素排序返回局部索引
            max_csd_in_every_piece = [max_csd_in_every_piece[i] for i in sorted_indices]#排序后的元素
            index_of_top_three_max = [np.where(abs_csd_chosen == csd)[0][0] for csd in
                                                    max_csd_in_every_piece]#排序后的abs_csd_chosen全局索引

            overall_index_of_top_three_max = [(index+int(number_of_f_low / self.down_number)) for index in index_of_top_three_max]
            """angle_csd[overall_index_of_top_three_max[0]] / self.real_angle<0.5,则去掉这个最大值"""
            # overall_index_of_top_three_max = overall_index_of_top_three_max[overall_index_of_top_three_max != 0]
            # 假设 angle_csd 是一个已定义的数组，并且 overall_index_of_top_three_max 是其索引数组

            #如果频率对应求得的模数小于0.6，则去掉这个最大值
            overall_index_of_top_three_max = np.array([
                index for index in overall_index_of_top_three_max
                if angle_csd[int(index)] / self.real_angle >= self.mode_th and if_csd_mode[int(index)] != 0
            ])
            # overrall_index_true为了让最后的数组长度为max_number，没有模式的情况补np.nan
            overall_index_true= overall_index_of_top_three_max
            if len( overall_index_of_top_three_max) < self.max_number:
                for i in range(self.max_number - len(overall_index_of_top_three_max)):
                    # max_csd_in_every_piece = np.append(max_csd_in_every_piece,0)
                    overall_index_of_top_three_max=np.append(overall_index_of_top_three_max,0)
                    #如果（most，sec，third，forth）认为third没有最大值，对应位置的索引置为0，并给出的信号这一行全0
                    overall_index_true = np.append(overall_index_true,np.nan)
            #index全都转为整型，多余self.max_number的去掉
            overall_index_of_top_three_max = (np.array(overall_index_of_top_three_max)[0:self.max_number]).astype(int)
            """将top three的频率、幅值、相位对应取出"""
            """ n*3 m1(m, 幅值, 频率, 相位),m2(),m3()"""

            if np.isnan(overall_index_true[0]):
                new_signal1_add =np.zeros((1, 6), dtype=np.int)
            else:
                if (angle_csd[overall_index_of_top_three_max[0]] / self.real_angle)<0.6:
                    a=1
                new_signal1_add = np.array([angle_csd[overall_index_of_top_three_max[0]] / self.real_angle,
                                            1000 * log_abs_csd[overall_index_of_top_three_max[0]] / f[
                                                overall_index_of_top_three_max[0]],
                                            log_abs_csd[overall_index_of_top_three_max[0]],
                                            f[overall_index_of_top_three_max[0]],
                                            angle_csd[overall_index_of_top_three_max[0]], if_csd_mode[overall_index_of_top_three_max[0]]
                                            ]).reshape(1, 6)
            if np.isnan(overall_index_true[1]):
                new_signal2_add = np.zeros((1, 6), dtype=np.int)
            else:
                new_signal2_add = np.array([angle_csd[overall_index_of_top_three_max[1]] / self.real_angle,
                                            1000 * log_abs_csd[overall_index_of_top_three_max[1]] / f[
                                                overall_index_of_top_three_max[1]],
                                            log_abs_csd[overall_index_of_top_three_max[1]],
                                            f[overall_index_of_top_three_max[1]],
                                            angle_csd[overall_index_of_top_three_max[1]], if_csd_mode[overall_index_of_top_three_max[1]]
                                            ]).reshape(1, 6)
            if np.isnan(overall_index_true[2]):
                new_signal3_add = np.zeros((1, 6), dtype=np.int)
            else:
                new_signal3_add = np.array([angle_csd[overall_index_of_top_three_max[2]] / self.real_angle,
                                            1000 * log_abs_csd[overall_index_of_top_three_max[2]] / f[
                                                overall_index_of_top_three_max[2]],
                                            log_abs_csd[overall_index_of_top_three_max[2]],
                                            f[overall_index_of_top_three_max[2]],
                                            angle_csd[overall_index_of_top_three_max[2]], if_csd_mode[overall_index_of_top_three_max[2]]
                                           ]).reshape(1, 6)
            if np.isnan(overall_index_true[3]):
                new_signal4_add = np.zeros((1, 6), dtype=np.int)
            else:
                new_signal4_add = np.array([angle_csd[overall_index_of_top_three_max[3]] / self.real_angle,
                                            1000 * log_abs_csd[overall_index_of_top_three_max[3]] / f[
                                                overall_index_of_top_three_max[3]],
                                            log_abs_csd[overall_index_of_top_three_max[3]],
                                            f[overall_index_of_top_three_max[3]],
                                            angle_csd[overall_index_of_top_three_max[3]], if_csd_mode[overall_index_of_top_three_max[3]]
                                           ]).reshape(1, 6)
            if len(new_signal1_add) == 0:
                new_signal1_add = np.zeros(shape=[1, 6], dtype=float)
            new_signal1.data = np.append(new_signal1.data, new_signal1_add, axis=0)

            if len(new_signal2_add) == 0:
                new_signal2_add = np.zeros(shape=[1, 6], dtype=float)
            new_signal2.data = np.append(new_signal2.data, new_signal2_add, axis=0)

            if len(new_signal3_add) == 0:
                new_signal3_add = np.zeros(shape=[1, 6], dtype=float)
            new_signal3.data = np.append(new_signal3.data, new_signal3_add, axis=0)
            if len(new_signal4_add) == 0:
                new_signal4_add = np.zeros(shape=[1, 6], dtype=float)
            new_signal4.data = np.append(new_signal4.data, new_signal4_add, axis=0)
        return new_signal1, new_signal2, new_signal3, new_signal4


# class M_mode_th(BaseProcessor):
#     """
#            -------
#            用于计算模数
#            data1，data2为两道Mirnov探针信号（尽可能近）
#            chip_time为做互功率谱时的窗口时间长度，默认为5ms
#            down_number为做互功率谱时的降点数（down_number = 1，即取了全时间窗的点，即FFT窗仅为一个），默认为8
#            low_fre为所需最低频率，默认为100Hz
#            high_fre为所需最高频率，默认为20kHz
#            step_fre为选取最大频率时的步长，默认为3kHz（这个其实没用到）
#            max_number为选取最大频率的个数，默认为3个
#            # var_th为频率间的方差阈值，默认为1e-13这一项没了！！！
#            real_angle为两道Mirnov探针间的极向空间角度差，默认为15°
#            coherence_th为互相关系数阈值，默认为0.95
#            -------
#     """
#
#     def __init__(self, real_angle, var_th: float, coherence_th: float, mode_th:float,chip_time=5e-3, down_number=1, low_fre=1e2,
#                  high_fre=2e4, step_fre=3e3,
#                  max_number=int(4)):
#         # var_th=1e-13, coherence_th=0.95, real_angle=7.5
#         # var_th=1e-20, coherence_th=0.9, real_angle=15
#         # var_th=1e-13, coherence_th=0.9, real_angle=22.5
#         ## var_th=1e-14, coherence_th=0.95, real_angle=22.5
#         super().__init__(real_angle=real_angle, var_th=var_th, coherence_th=coherence_th, mode_th=mode_th,chip_time=chip_time,
#                          down_number=down_number, low_fre=low_fre, high_fre=high_fre, step_fre=step_fre, max_number=max_number)
#         # var_th=1e-13, coherence_th=0.95, real_angle=7.5
#         # var_th=1e-20, coherence_th=0.9, real_angle=15
#         # var_th=1e-13, coherence_th=0.9, real_angle=22.5
#         ## var_th=1e-14, coherence_th=0.95, real_angle=22.5
#         self.chip_time = chip_time
#         self.down_number = down_number
#         self.low_fre = low_fre
#         self.high_fre = high_fre
#         self.step_fre = step_fre
#         self.max_number = max_number
#         self.real_angle = real_angle
#         self.var_th = var_th
#         self.coherence_th = coherence_th
#         self.mode_th = mode_th
#         # self.var_th = 0
#         # self.coherence_th = -100
#         # if real_angle == 7.5:
#         #     self.var_th = 1e-13
#         #     self.coherence_th = 0.95
#         #     # self.var_th = 0
#         #     # self.coherence_th = -100
#         # elif real_angle == 15:
#         #     self.var_th = 1e-20
#         #     self.coherence_th = 0.9
#         # elif real_angle == 22.5:
#         #     self.var_th = 1e-13
#         #     self.coherence_th = 0.9
#         # else:
#         #     self.var_th = var_th
#         #     self.coherence_th = coherence_th
#
#     def ampd(self, data):
#         """
#         实现AMPD算法
#         :param data: 1-D numpy.ndarray
#         :return: 波峰所在索引值的列表
#         """
#         p_data = np.zeros_like(data, dtype=np.int32)
#         count = data.shape[0]
#         arr_rowsum = []
#         for k in range(1, count // 2 + 1):  # k is step
#
#             row_sum = 0
#             for i in range(k, count - k):  # i is index
#                 if data[i] > data[i - k] and data[i] > data[i + k]:
#                     row_sum -= 1
#             arr_rowsum.append(row_sum)
#         min_index = np.argmin(arr_rowsum)
#         max_window_length = min_index
#         for k in range(1, max_window_length + 1):
#
#             for i in range(k, count - k):
#                 if data[i] > data[i - k] and data[i] > data[i + k]:
#                     p_data[i] += 1
#         return np.where(p_data == max_window_length)[0]
#
#     def transform(self, signal1: Signal, signal2: Signal):
#         signal1 = deepcopy(signal1)
#         signal2 = deepcopy(signal2)
#         # signal2.data = -signal2.data
#         # number_of_a_window = int(self.chip_time * signal1.attributes['OriginalSampleRate'])  # window length
#         # number_of_windows = len(signal1.time) // number_of_a_window  # number of the chip in the window
#         number_of_a_window = len(signal1.data[0])
#         number_of_windows = len(signal1.data)
#         number_of_f_low = int(self.low_fre * self.chip_time)  # lowest frequency
#         number_of_f_high = int(self.high_fre * self.chip_time)  # Highest frequency
#         number_of_f_step = int(self.step_fre * self.chip_time)  # select max_frequency length
#         new_signal1 = Signal(data=np.empty(shape=[0, 6], dtype=float), attributes=dict())
#         new_signal2 = Signal(data=np.empty(shape=[0, 6], dtype=float), attributes=dict())
#         new_signal3 = Signal(data=np.empty(shape=[0, 6], dtype=float), attributes=dict())
#         new_signal4 = Signal(data=np.empty(shape=[0, 6], dtype=float), attributes=dict())
#         # signal1.prop_dict['SampleRate']为原采样率
#         new_signal1.attributes = signal1.attributes
#         new_signal2.attributes = signal1.attributes
#         new_signal3.attributes = signal1.attributes
#         new_signal4.attributes = signal1.attributes
#         new_signal1.parent = signal1.parent
#         new_signal2.parent = signal1.parent
#         new_signal3.parent = signal1.parent
#         new_signal4.parent = signal1.parent
#
#         # slide——————换成slice_processor
#         ###slice_time=5ms, slice与slice的采样率1000k
#         for i in range(number_of_windows):  #####改成一个slice读取处理
#             # if i ==207:
#             #     a=1
#             if_mode_var = 1
#             # new_signal1.d
#             chip_data1 = signal1.data[i]
#             chip_data2 = signal2.data[i]
#             """做互功率谱,看相关性,并取互功率谱幅值、相位"""
#             # calculate cross spectral density
#             # self.down_number = self.down_number/nperseg_padded
#             (f, csd) = sig.csd(chip_data1, chip_data2, fs=signal1.attributes['OriginalSampleRate'], window='hann',
#                                nperseg=int(number_of_a_window / self.down_number),
#                                scaling='density')
#             (f_coherence, coherence) = sig.coherence(chip_data1, chip_data2,
#                                                      fs=signal1.attributes['OriginalSampleRate'],
#                                                      window='hann',
#                                                      nperseg=int(number_of_a_window / self.down_number))
#             # 1000Hz以下相当于滤波
#             csd[:5] = np.zeros(5)
#             abs_csd = np.abs(csd)
#             abs_csd[:5] = np.zeros(5)
#             # log_abs_csd = 20 * np.log(abs_csd)
#             log_abs_csd = 1e6 * abs_csd
#             phase_csd = np.angle(csd) * 180 / np.pi
#             """在信号相关系数阈值之上的angle存入数组，否则该数组为空"""
#             if_mode_coherence = np.where(coherence > self.coherence_th, 1, 0)
#             angle_csd = phase_csd
#             # angle_csd = np.where(coherence > self.coherence_th, phase_csd, np.nan)
#             """csd中选取一段频率，低频到高频，且做方差,这里就是选了10K以内的"""
#             # abs_csd_chosen = np.abs(csd[number_of_f_low // down_number: number_of_f_high // down_number])
#             abs_csd_chosen = np.abs(
#                 csd[int(number_of_f_low / self.down_number): int(number_of_f_high / self.down_number)])
#             var_csd = np.var(np.abs(abs_csd_chosen))
#             # 求互功率谱均值，若最大频率大于最小频率，且小于ch_th就证明存在明显的模式, 目前使用方差与互相关系数做约束
#             # 判断是否有明显模式
#             if var_csd < self.var_th:
#                 if_mode_var = 0
#                 # new_signal1.data = np.concatenate((new_signal1.data, np.zeros((1, 5), dtype=float)), axis=0)
#                 # new_signal2.data = np.concatenate((new_signal2.data, np.zeros((1, 5), dtype=float)), axis=0)
#                 # new_signal3.data = np.concatenate((new_signal3.data, np.zeros((1, 5), dtype=float)), axis=0)
#                 # new_signal4.data = np.concatenate((new_signal4.data, np.zeros((1, 5), dtype=float)), axis=0)
#             if_csd_mode = if_mode_var * if_mode_coherence
#             down_number_of_step = int(number_of_f_step / self.down_number)
#             overall_index_of_max_csd_in_every_piece = []
#             # max_csd_in_every_piece = []
#             """csd分片段处理,滑动窗口每段找出一个最大值,并存其全局最大值索引"""
#             # for j in range(int(len(abs_csd_chosen) / down_number_of_step)):  # 把csd分几个片段处理
#             #     abs_csd_piece = np.abs(abs_csd_chosen[j * down_number_of_step:(j + 1) * down_number_of_step])
#             #     max_csd_in_every_piece.append(max(abs_csd_piece))  # 滑动窗口每段找出一个最大值
#             abs_csd_chosen_max_index = self.ampd(abs_csd_chosen)  # 找出所有最大值索引
#
#             max_csd_in_every_piece = [abs_csd_chosen[i] for i in abs_csd_chosen_max_index]  # 找出索引对应的元素
#             # if len(max_csd_in_every_piece)<=self.max_number:
#             #     for i in range(len(self.max_number-max_csd_in_every_piece)):
#             #         max_csd_in_every_piece = np.append(max_csd_in_every_piece)
#             sorted_indices = np.argsort(max_csd_in_every_piece)[::-1]  # 元素排序返回局部索引
#             max_csd_in_every_piece = [max_csd_in_every_piece[i] for i in sorted_indices]  # 排序后的元素
#             index_of_top_three_max = [np.where(abs_csd_chosen == csd)[0][0] for csd in
#                                       max_csd_in_every_piece]  # 排序后的abs_csd_chosen全局索引
#
#             overall_index_of_top_three_max = [(index + int(number_of_f_low / self.down_number)) for index in
#                                               index_of_top_three_max]
#             """angle_csd[overall_index_of_top_three_max[0]] / self.real_angle<0.5,则去掉这个最大值"""
#             # overall_index_of_top_three_max = overall_index_of_top_three_max[overall_index_of_top_three_max != 0]
#             # 假设 angle_csd 是一个已定义的数组，并且 overall_index_of_top_three_max 是其索引数组
#
#             # 如果频率对应求得的模数小于0.6，则去掉这个最大值
#             overall_index_of_top_three_max = np.array([
#                 index for index in overall_index_of_top_three_max
#                 if angle_csd[int(index)] / self.real_angle >= self.mode_th and if_csd_mode[int(index)] != 0
#             ])
#             # overrall_index_true为了让最后的数组长度为max_number，没有模式的情况补np.nan
#             overall_index_true = overall_index_of_top_three_max
#             if len(overall_index_of_top_three_max) < self.max_number:
#                 for i in range(self.max_number - len(overall_index_of_top_three_max)):
#                     # max_csd_in_every_piece = np.append(max_csd_in_every_piece,0)
#                     overall_index_of_top_three_max = np.append(overall_index_of_top_three_max, 0)
#                     # 如果（most，sec，third，forth）认为third没有最大值，对应位置的索引置为0，并给出的信号这一行全0
#                     overall_index_true = np.append(overall_index_true, np.nan)
#             # index全都转为整型，多余self.max_number的去掉
#             overall_index_of_top_three_max = (np.array(overall_index_of_top_three_max)[0:self.max_number]).astype(int)
#             """将top three的频率、幅值、相位对应取出"""
#             """ n*3 m1(m, 幅值, 频率, 相位),m2(),m3()"""
#
#             if np.isnan(overall_index_true[0]):
#                 new_signal1_add = np.zeros((1, 6), dtype=float)
#             else:
#                 if (angle_csd[overall_index_of_top_three_max[0]] / self.real_angle) < 0.6:
#                     a = 1
#                 new_signal1_add = np.array([angle_csd[overall_index_of_top_three_max[0]] / self.real_angle,
#                                             1000 * log_abs_csd[overall_index_of_top_three_max[0]] / f[
#                                                 overall_index_of_top_three_max[0]],
#                                             log_abs_csd[overall_index_of_top_three_max[0]],
#                                             f[overall_index_of_top_three_max[0]],
#                                             angle_csd[overall_index_of_top_three_max[0]],
#                                             if_csd_mode[overall_index_of_top_three_max[0]]
#                                             ]).reshape(1, 6)
#             if np.isnan(overall_index_true[1]):
#                 new_signal2_add = np.zeros((1, 6), dtype=float)
#             else:
#                 new_signal2_add = np.array([angle_csd[overall_index_of_top_three_max[1]] / self.real_angle,
#                                             1000 * log_abs_csd[overall_index_of_top_three_max[1]] / f[
#                                                 overall_index_of_top_three_max[1]],
#                                             log_abs_csd[overall_index_of_top_three_max[1]],
#                                             f[overall_index_of_top_three_max[1]],
#                                             angle_csd[overall_index_of_top_three_max[1]],
#                                             if_csd_mode[overall_index_of_top_three_max[1]]
#                                             ]).reshape(1, 6)
#             if np.isnan(overall_index_true[2]):
#                 new_signal3_add = np.zeros((1, 6), dtype=float)
#             else:
#                 new_signal3_add = np.array([angle_csd[overall_index_of_top_three_max[2]] / self.real_angle,
#                                             1000 * log_abs_csd[overall_index_of_top_three_max[2]] / f[
#                                                 overall_index_of_top_three_max[2]],
#                                             log_abs_csd[overall_index_of_top_three_max[2]],
#                                             f[overall_index_of_top_three_max[2]],
#                                             angle_csd[overall_index_of_top_three_max[2]],
#                                             if_csd_mode[overall_index_of_top_three_max[2]]
#                                             ]).reshape(1, 6)
#             if np.isnan(overall_index_true[3]):
#                 new_signal4_add = np.zeros((1, 6), dtype=float)
#             else:
#                 new_signal4_add = np.array([angle_csd[overall_index_of_top_three_max[3]] / self.real_angle,
#                                             1000 * log_abs_csd[overall_index_of_top_three_max[3]] / f[
#                                                 overall_index_of_top_three_max[3]],
#                                             log_abs_csd[overall_index_of_top_three_max[3]],
#                                             f[overall_index_of_top_three_max[3]],
#                                             angle_csd[overall_index_of_top_three_max[3]],
#                                             if_csd_mode[overall_index_of_top_three_max[3]]
#                                             ]).reshape(1, 6)
#             if len(new_signal1_add) == 0:
#                 new_signal1_add = np.zeros(shape=[1, 6], dtype=float)
#             new_signal1.data = np.append(new_signal1.data, new_signal1_add, axis=0)
#
#             if len(new_signal2_add) == 0:
#                 new_signal2_add = np.zeros(shape=[1, 6], dtype=float)
#             new_signal2.data = np.append(new_signal2.data, new_signal2_add, axis=0)
#
#             if len(new_signal3_add) == 0:
#                 new_signal3_add = np.zeros(shape=[1, 6], dtype=float)
#             new_signal3.data = np.append(new_signal3.data, new_signal3_add, axis=0)
#             if len(new_signal4_add) == 0:
#                 new_signal4_add = np.zeros(shape=[1, 6], dtype=float)
#             new_signal4.data = np.append(new_signal4.data, new_signal4_add, axis=0)
#         return new_signal1, new_signal2, new_signal3, new_signal4

class M_mode_th2(BaseProcessor):
    """
           -------
           用于计算模数
           data1，data2为两道Mirnov探针信号（尽可能近）
           chip_time为做互功率谱时的窗口时间长度，默认为5ms
           down_number为做互功率谱时的降点数（down_number = 1，即取了全时间窗的点，即FFT窗仅为一个），默认为8
           low_fre为所需最低频率，默认为100Hz
           high_fre为所需最高频率，默认为20kHz
           step_fre为选取最大频率时的步长，默认为3kHz（这个其实没用到）
           max_number为选取最大频率的个数，默认为3个
           # var_th为频率间的方差阈值，默认为1e-13这一项没了！！！
           real_angle为两道Mirnov探针间的极向空间角度差，默认为15°
           coherence_th为互相关系数阈值，默认为0.95
           -------
    """

    def __init__(self, real_angle, var_th: float, coherence_th: float, chip_time=5e-3, down_number=1, low_fre=1e2,
                 high_fre=2e4, step_fre=3e3,
                 max_number=int(4)):
        # var_th=1e-13, coherence_th=0.95, real_angle=7.5
        # var_th=1e-20, coherence_th=0.9, real_angle=15
        # var_th=1e-12, coherence_th=0.9, real_angle=22.5
        super().__init__(var_th=var_th, coherence_th=coherence_th, real_angle=real_angle,
                         chip_time=chip_time, down_number=down_number, low_fre=low_fre, high_fre=high_fre,
                         step_fre=step_fre, max_number=max_number)

        self.chip_time = chip_time
        self.down_number = down_number
        self.low_fre = low_fre
        self.high_fre = high_fre
        self.step_fre = step_fre
        self.max_number = max_number
        self.real_angle = real_angle
        # self.var_th = 0
        # self.coherence_th = -100
        if real_angle == 7.5:
            self.var_th = 1e-13
            self.coherence_th = 0.95
            # self.var_th = 0
            # self.coherence_th = -100
        elif real_angle == 15:
            self.var_th = 1e-20
            self.coherence_th = 0.9
        elif real_angle == 22.5:
            self.var_th = 1e-12
            self.coherence_th = 0.9
        else:
            self.var_th = var_th
            self.coherence_th = coherence_th
    def ampd(self, data):
        """
        实现AMPD算法
        :param data: 1-D numpy.ndarray
        :return: 波峰所在索引值的列表
        """
        p_data = np.zeros_like(data, dtype=np.int32)
        count = data.shape[0]
        arr_rowsum = []
        for k in range(1, count // 2 + 1):  # k is step

            row_sum = 0
            for i in range(k, count - k):  # i is index
                if data[i] > data[i - k] and data[i] > data[i + k]:
                    row_sum -= 1
            arr_rowsum.append(row_sum)
        min_index = np.argmin(arr_rowsum)
        max_window_length = min_index
        for k in range(1, max_window_length + 1):

            for i in range(k, count - k):
                if data[i] > data[i - k] and data[i] > data[i + k]:
                    p_data[i] += 1
        return np.where(p_data == max_window_length)[0]

    def transform(self, signal1: Signal, signal2: Signal):
        signal1 = deepcopy(signal1)
        signal2 = deepcopy(signal2)
        # signal2.data = -signal2.data
        # number_of_a_window = int(self.chip_time * signal1.attributes['OriginalSampleRate'])  # window length
        # number_of_windows = len(signal1.time) // number_of_a_window  # number of the chip in the window
        number_of_a_window = len(signal1.data[0])
        number_of_windows = len(signal1.data)
        number_of_f_low = int(self.low_fre * self.chip_time)  # lowest frequency
        number_of_f_high = int(self.high_fre * self.chip_time)  # Highest frequency
        number_of_f_step = int(self.step_fre * self.chip_time)  # select max_frequency length
        new_signal1 = Signal(data=np.empty(shape=[0, 6], dtype=float), attributes=dict())
        new_signal2 = Signal(data=np.empty(shape=[0, 6], dtype=float), attributes=dict())
        new_signal3 = Signal(data=np.empty(shape=[0, 6], dtype=float), attributes=dict())
        new_signal4 = Signal(data=np.empty(shape=[0, 6], dtype=float), attributes=dict())
        # signal1.prop_dict['SampleRate']为原采样率
        new_signal1.attributes = signal1.attributes
        new_signal2.attributes = signal1.attributes
        new_signal3.attributes = signal1.attributes
        new_signal4.attributes = signal1.attributes
        new_signal1.parent = signal1.parent
        new_signal2.parent = signal1.parent
        new_signal3.parent = signal1.parent
        new_signal4.parent = signal1.parent

        # slide——————换成slice_processor
        ###slice_time=5ms, slice与slice的采样率1000k
        for i in range(number_of_windows):  #####改成一个slice读取处理

            if_mode_var = 1
            # new_signal1.d
            chip_data1 = signal1.data[i]
            chip_data2 = signal2.data[i]
            """做互功率谱,看相关性,并取互功率谱幅值、相位"""
            # calculate cross spectral density

            # self.down_number = self.down_number/nperseg_padded
            (f, csd) = sig.csd(chip_data1, chip_data2, fs=signal1.attributes['OriginalSampleRate'], window='hann',
                               nperseg=int(number_of_a_window / self.down_number),
                               scaling='density')
            (f_coherence, coherence) = sig.coherence(chip_data1, chip_data2,
                                                     fs=signal1.attributes['OriginalSampleRate'],
                                                     window='hann',
                                                     nperseg=int(number_of_a_window/ self.down_number))
            abs_csd = np.abs(csd)
            # log_abs_csd = 20 * np.log(abs_csd)
            log_abs_csd = 1e6 * abs_csd
            phase_csd = np.angle(csd) * 180 / np.pi
            """在信号相关系数阈值之上的angle存入数组，否则该数组为空"""
            if_mode_coherence = np.where(coherence > self.coherence_th, 1, 0)
            angle_csd=phase_csd
            # angle_csd = np.where(coherence > self.coherence_th, phase_csd, np.nan)
            """csd中选取一段频率，低频到高频，且做方差,这里就是选了10K以内的"""
            # abs_csd_chosen = np.abs(csd[number_of_f_low // down_number: number_of_f_high // down_number])
            abs_csd_chosen = np.abs(csd[int(number_of_f_low / self.down_number): int(number_of_f_high / self.down_number)])
            var_csd = np.var(np.abs(abs_csd_chosen))
            # 求互功率谱均值，若最大频率大于最小频率，且小于ch_th就证明存在明显的模式, 目前使用方差与互相关系数做约束
            # 判断是否有明显模式
            if var_csd < self.var_th:
                if_mode_var= 0
                # new_signal1.data = np.concatenate((new_signal1.data, np.zeros((1, 5), dtype=float)), axis=0)
                # new_signal2.data = np.concatenate((new_signal2.data, np.zeros((1, 5), dtype=float)), axis=0)
                # new_signal3.data = np.concatenate((new_signal3.data, np.zeros((1, 5), dtype=float)), axis=0)
                # new_signal4.data = np.concatenate((new_signal4.data, np.zeros((1, 5), dtype=float)), axis=0)
            down_number_of_step = int(number_of_f_step / self.down_number)
            overall_index_of_max_csd_in_every_piece = []
            # max_csd_in_every_piece = []
            """csd分片段处理,滑动窗口每段找出一个最大值,并存其全局最大值索引"""
            # for j in range(int(len(abs_csd_chosen) / down_number_of_step)):  # 把csd分几个片段处理
            #     abs_csd_piece = np.abs(abs_csd_chosen[j * down_number_of_step:(j + 1) * down_number_of_step])
            #     max_csd_in_every_piece.append(max(abs_csd_piece))  # 滑动窗口每段找出一个最大值
            abs_csd_chosen_max_index = self.ampd(abs_csd_chosen)#找出所有最大值索引

            max_csd_in_every_piece = [abs_csd_chosen[i] for i in abs_csd_chosen_max_index]#找出索引对应的元素
            # if len(max_csd_in_every_piece)<=self.max_number:
            #     for i in range(len(self.max_number-max_csd_in_every_piece)):
            #         max_csd_in_every_piece = np.append(max_csd_in_every_piece)
            sorted_indices = np.argsort(max_csd_in_every_piece)[::-1]#元素排序返回局部索引
            max_csd_in_every_piece = [max_csd_in_every_piece[i] for i in sorted_indices]#排序后的元素
            index_of_top_three_max = [np.where(abs_csd_chosen == csd)[0][0] for csd in
                                                    max_csd_in_every_piece]#排序后的abs_csd_chosen全局索引

            overall_index_of_top_three_max = [(index+int(number_of_f_low / self.down_number)) for index in index_of_top_three_max]
            if len(max_csd_in_every_piece) < self.max_number:
                for i in range(self.max_number - len(max_csd_in_every_piece)):
                    # max_csd_in_every_piece = np.append(max_csd_in_every_piece,0)
                    overall_index_of_top_three_max = np.append(overall_index_of_top_three_max,0)
            # max_csd_in_every_piece = max_csd_in_every_piece[0:self.max_number]
            overall_index_of_top_three_max = (np.array(overall_index_of_top_three_max)[0:self.max_number]).astype(int)
            """将top three的频率、幅值、相位对应取出"""
            """ n*3 m1(m, 幅值, 频率, 相位),m2(),m3()"""
            if_csd_mode = if_mode_var*if_mode_coherence
            new_signal1_add = np.array([angle_csd[overall_index_of_top_three_max[0]] / self.real_angle,
                                        1000 * log_abs_csd[overall_index_of_top_three_max[0]] / f[
                                            overall_index_of_top_three_max[0]],
                                        log_abs_csd[overall_index_of_top_three_max[0]],
                                        f[overall_index_of_top_three_max[0]],
                                        angle_csd[overall_index_of_top_three_max[0]], if_csd_mode[overall_index_of_top_three_max[0]]
                                        ]).reshape(1, 6)
            new_signal2_add = np.array([angle_csd[overall_index_of_top_three_max[1]] / self.real_angle,
                                        1000 * log_abs_csd[overall_index_of_top_three_max[1]] / f[
                                            overall_index_of_top_three_max[1]],
                                        log_abs_csd[overall_index_of_top_three_max[1]],
                                        f[overall_index_of_top_three_max[1]],
                                        angle_csd[overall_index_of_top_three_max[1]], if_csd_mode[overall_index_of_top_three_max[1]]
                                        ]).reshape(1, 6)
            new_signal3_add = np.array([angle_csd[overall_index_of_top_three_max[2]] / self.real_angle,
                                        1000 * log_abs_csd[overall_index_of_top_three_max[2]] / f[
                                            overall_index_of_top_three_max[2]],
                                        log_abs_csd[overall_index_of_top_three_max[2]],
                                        f[overall_index_of_top_three_max[2]],
                                        angle_csd[overall_index_of_top_three_max[2]], if_csd_mode[overall_index_of_top_three_max[2]]
                                       ]).reshape(1, 6)
            new_signal4_add = np.array([angle_csd[overall_index_of_top_three_max[3]] / self.real_angle,
                                        1000 * log_abs_csd[overall_index_of_top_three_max[3]] / f[
                                            overall_index_of_top_three_max[3]],
                                        log_abs_csd[overall_index_of_top_three_max[3]],
                                        f[overall_index_of_top_three_max[3]],
                                        angle_csd[overall_index_of_top_three_max[3]], if_csd_mode[overall_index_of_top_three_max[3]]
                                       ]).reshape(1, 6)
            if len(new_signal1_add) == 0:
                new_signal1_add = np.zeros(shape=[1, 6], dtype=float)
            new_signal1.data = np.append(new_signal1.data, new_signal1_add, axis=0)

            if len(new_signal2_add) == 0:
                new_signal2_add = np.zeros(shape=[1, 6], dtype=float)
            new_signal2.data = np.append(new_signal2.data, new_signal2_add, axis=0)

            if len(new_signal3_add) == 0:
                new_signal3_add = np.zeros(shape=[1, 6], dtype=float)
            new_signal3.data = np.append(new_signal3.data, new_signal3_add, axis=0)
            if len(new_signal4_add) == 0:
                new_signal4_add = np.zeros(shape=[1, 6], dtype=float)
            new_signal4.data = np.append(new_signal4.data, new_signal4_add, axis=0)
        return new_signal1, new_signal2, new_signal3, new_signal4