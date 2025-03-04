# -*-  coding:utf-8  -*-
import numpy as np
from scipy.fftpack import fft
import math
from typing import Tuple, Union
from jddb.processor import Signal, BaseProcessor
from typing import Optional, List
from scipy.fftpack import fft
from copy import deepcopy
class FourBFFTProcessor(BaseProcessor):
    """
       做fft，取幅值前三的，计算平方和的开根号，返回该值以及前三幅值对应的频率
    """
    def __init__(self, singal_rate: int, pad=1):#pad 表示f精度，精度为 1/(fs*pad),f 的长度为 pad*fs/2
        super().__init__(singal_rate=singal_rate, pad=pad)
        self.signal_rate = singal_rate
        self.pad = pad
        # self.threshold_ratio =threshold_ratio

    def transform(self, *signal: Signal):
        # #signal[0]:B_LFS_Inte_slice
        # #signal[1]:m_most_max_all
        # #signal[2]:m_sec_max_all
        # #signal[3]:m_third_max_all
        # #signal[4]:m_forth_max_all
        B_LFS_Inte_slice_signal = deepcopy(signal.__getitem__(0))
        m_most_all_signal = deepcopy((signal.__getitem__(1)))
        m_sec_all_signal = deepcopy((signal.__getitem__(2)))
        m_third_all_signal = deepcopy((signal.__getitem__(3)))
        m_forth_all_signal = deepcopy((signal.__getitem__(4)))
        amp_inte_range_signal = deepcopy((signal.__getitem__(4)))
        # m_signal_list = [m_most_max_all_signal,m_most_sec_all_signal,m_most_third_all_signal,m_most_forth_all_signal]
        lfs_amp_cover = self.fft_lfs_amp(B_LFS_Inte_slice_signal.data)
        lfs_amp_cover[:,0:5]=np.zeros((len(lfs_amp_cover),5))
        data,amp_inte_range_array = self.fft_amp_fre(m_most_all_signal.data,m_sec_all_signal.data,m_third_all_signal.data,m_forth_all_signal.data,lfs_amp_cover,len(B_LFS_Inte_slice_signal.data[0]))
        m_most_all_signal.data =data[0]
        m_sec_all_signal.data = data[1]
        m_third_all_signal.data = data[2]
        m_forth_all_signal.data = data[3]
        amp_inte_range_signal.data=amp_inte_range_array

        return m_most_all_signal, m_sec_all_signal, m_third_all_signal, m_forth_all_signal,amp_inte_range_signal
        # most_b_theta, sec_b_theta, third_b_theta, forth_b_theta
        # (mode, b_amp, fre)

    def fft_lfs_amp(self,  lfs_data):  # u_i 是第i个时刻的u向量
        N = len(lfs_data[0])
        n_pad = N * self.pad
        fs = self.signal_rate
        freq_pad = np.fft.fftfreq(n_pad, 1 / fs)  # 计算频率轴
        f = freq_pad[:int(n_pad / 2)]
        lfs_amp_cover = np.empty(shape=(0, len(f)), dtype=float)
        for i in range(len(lfs_data)):
            lfs_fft_y = fft(lfs_data[i], n=n_pad)
            lfs_normed_abs_y = np.abs(lfs_fft_y) / (N / 2)
            lfs_raw = lfs_normed_abs_y[:int(n_pad / 2)]
            lfs_amp_cover = np.vstack([lfs_amp_cover, lfs_raw])
        return lfs_amp_cover

    def fft_amp_fre(self, m_most_data,m_sec_data,m_third_data,m_forth_data, lfs_data,N):  # u_i 是第i个时刻的u向量
        lfs_amp_1_cover = np.empty(shape=(0, 3), dtype=float)
        lfs_amp_2_cover = np.empty(shape=(0, 3), dtype=float)
        lfs_amp_3_cover = np.empty(shape=(0, 3), dtype=float)
        lfs_amp_4_cover = np.empty(shape=(0, 3), dtype=float)
        amp_inte_range_array = np.zeros(shape=(len(m_most_data),4, 4), dtype=int)
        n_pad = N * self.pad
        fs = self.signal_rate
        freq_pad = np.fft.fftfreq(n_pad, 1 / fs)  # 计算频率轴
        f = freq_pad[:int(n_pad / 2)]
        lfs_amp_fre_cover = np.empty(shape=(0, 3), dtype=float)

        for i in range(len(m_most_data)):
            #if m_most_data[i]不包含0元素：
            if m_most_data[i][-1] != 0:
                lfs_data_i=lfs_data[i]
                # print(" i ",i, "---------------------------------------------- percent ",lfs_data_i[0]/np.sum(lfs_data_i)," % ")
                # if i==47:
                #     print(i)
                E_row = []
                # 判断 A 的前五个元素是否不全为零
                if np.any(m_most_data[i, :] != 0) :  # 如果前五个元素中有不为零的,且most的fre不为200
                    # 选择 A 的第四个元素作为 E 的第一个元素
                    E_row.append(m_most_data[i, 3])
                    # 对 B, C, D 同样做处理
                if np.any(m_sec_data[i, :] != 0):
                    E_row.append(m_sec_data[i, 3])
                if np.any(m_third_data[i, :] != 0):
                    E_row.append(m_third_data[i, 3])
                if np.any(m_forth_data[i, :] != 0):
                    E_row.append(m_forth_data[i, 3])
                    # 如果 E 的长度大于 1，说明 A, B, C, D 中至少有两个不为零
                fre_indices = [np.where(f == E_row[i])[0][0] for i in range(len(E_row))]
                amps,index_range_array=self.array_near(lfs_data[i], fre_indices)
                f_range_array= np.array([f[i] for i in index_range_array.flat]).reshape(index_range_array.shape)
                amp_inte_range_array[i,:len(index_range_array)]=f_range_array
                # print("i {i}, amps {amps}".format(i=i,amps=amps))
                while len(amps) < 4:
                    amps.append(0)
                lfs_amp_1_cover = np.vstack((lfs_amp_1_cover, np.array(
                    [m_most_data[i][0], amps[0], m_most_data[i][3]])))####这个修改一下，我其实需要四个都传进来，然后确定频率
                lfs_amp_2_cover = np.vstack((lfs_amp_2_cover, np.array(
                    [m_sec_data[i][0], amps[1], m_sec_data[i][3]])))####这个修改一下，我其实需要四个都传进来，然后确定频率
                lfs_amp_3_cover = np.vstack((lfs_amp_3_cover, np.array(
                    [m_third_data[i][0], amps[2], m_third_data[i][3]])))####这个修改一下，我其实需要四个都传进来，然后确定频率
                lfs_amp_4_cover = np.vstack((lfs_amp_4_cover, np.array(
                    [m_forth_data[i][0], amps[3], m_forth_data[i][3]])))####这个修改一下，我其实需要四个都传进来，然后确定频率

            else:
                lfs_amp_1_cover = np.vstack((lfs_amp_1_cover, np.array([0,0,0])))####这个修改一下，我其实需要四个都传进来，然后确定频率
                lfs_amp_2_cover = np.vstack((lfs_amp_2_cover, np.array([0,0,0])))####这个修改一下，我其实需要四个都传进来，然后确定频率
                lfs_amp_3_cover = np.vstack((lfs_amp_3_cover, np.array([0,0,0])))####这个修改一下，我其实需要四个都传进来，然后确定频率
                lfs_amp_4_cover = np.vstack((lfs_amp_4_cover, np.array([0,0,0])))####这个修改一下，我其实需要四个都传进来，然后确定频率
        return [lfs_amp_1_cover,lfs_amp_2_cover,lfs_amp_3_cover,lfs_amp_4_cover],amp_inte_range_array

        # 取最大值及其相邻值

    def find_local_peak(self, signal_data, peak_idx, window=3):
        """
        在给定的索引附近窗口内，找到局部最大值的真实位置。
        :param signal: 输入的信号数组
        :param peak_idx: 给定的峰值大致索引
        :param window: 搜索范围，前后各多少个点
        :return: 真实的峰值位置
        """
        # 确保搜索范围不会超出信号的边界
        local_peak_idxs = np.array([])
        for index in range(len(peak_idx)):
            start = max(0, peak_idx[index] - window)
            end = min(len(signal_data), peak_idx[index] + window + 1)
            # 在给定范围内找到最大值的索引
            local_peak_idx = np.argmax(signal_data[start:end]) + start
            local_peak_idxs = np.append(local_peak_idxs, local_peak_idx)
        return local_peak_idxs

    def find_continuous_values(self,array, max_index, threshold=0.1):
        """
        找到数组中以 max_index 为中心，连续大于 threshold 的值组成新数组。

        Parameters:
            array (list or np.ndarray): 输入数组
            max_index (int): 中心索引
            threshold (float): 判断的阈值，大于该值才被保留

        Returns:
            list: 新的连续值数组
        """
        # 转换为 NumPy 数组，方便操作
        array = np.array(array)

        # 初始化左、右索引
        left = max_index
        right = max_index

        # 向左扩展
        while left > 0 and array[left - 1] > threshold:
            left -= 1

        # 向右扩展
        while right < len(array) - 1 and array[right + 1] > threshold:
            right += 1
        return left,right
        # # 提取连续大于阈值的子数组
        # return array[left:right + 1].tolist()

    def formatted_print(self, max_indexs_raw, max_indexs, amp_array, lows, upps, segments,segment_maxs,amps_bs):
        # output = []
        index_range_array=np.zeros(shape=(len(lows),4),dtype=int)
        # # 添加第一行到第五行的内容
        # output.append(f"---------------------amp_array[0] percent {amp_array[0] / np.sum( amp_array)}%---------------------")
        # output.append(f"max_indexs_raw: {max_indexs_raw}")
        # output.append(f"max_indexs: {max_indexs}")
        # output.append(f"amp_array at max_indexs: {[amp_array[int(max_index)] for max_index in max_indexs]}")
        #
        # output.append("Details:")
        for index in range(len(lows)):
            low = lows[index]
            max_index = max_indexs[index]
            upp = upps[index]
            segment_max=segment_maxs[index]
            segment = segments[index]
            amps_b_i=amps_bs[index]
            max_indexs_raw_i = max_indexs_raw[index]
            index_range_array[index]=[low,max_index,max_indexs_raw_i,upp]
            #     output.append(
        #         f"low: {low}, max_index: {max_index}, upp: {upp}, len: {len(segment)}, amp: {amps_b_i}, segment[0]: {segment[0]}, max: {segment_max}, segment[-1]: {segment[-1]}")
        #
        # # 添加分割线
        # output.append("*" * 80)

        # # 一次性打印所有内容
        # print("\n".join(output))
        return index_range_array

    # 取最大值及其相邻值
    def array_near(self, amp_array, max_indexs_raw):
        amps_bs=[]

            # 找到真实的峰值位置
        max_indexs = self.find_local_peak(amp_array, max_indexs_raw, window=1)
        if max_indexs_raw[0]!=0 and max_indexs[0]==0:
            max_indexs[0]=max_indexs_raw[0]
        #找到大于 max_indexs 的最小元素，找到小于 max_indexs 的最大元素
        min_indexs=self.min_index(amp_array, max_indexs)

        # 存储切割后的片段，之后用这个来积分求幅值
        segments = []
        # 对每个 max_index 进行操作
        delta_low_upp = []
        upps = []
        lows = []
        segment_maxs=[]
        for max_index in max_indexs:
            low, upp = self.find_closest_indices(max_index, min_indexs.copy())  # 用 copy 防止修改原 min_indices
            left, right=self.find_continuous_values(amp_array[low:upp], int(max_index-low), threshold=0.1)#根据切分的low, upp片段，找到连续的大于0.1GS的值
            #将left，right转换为全局索引
            low=low+left
            if right==left:
                upp=low+right+1
            else:
                upp=low+right
            delta_index=upp-low#如果这个连续值跨度大于15，那么就取中间的15个点，即允许误差在2kHz
            if delta_index>10:
                low = int(max(low,max_index - 5))
                upp = int(min(upp,max_index + 5))
            upps.append(upp)
            lows.append(low)
            segment = amp_array[low:upp]
            segment_maxs.append(amp_array[int(max_index)])
            segments.append(segment)
            delta_low_upp.append(len(segment))

        # print("max_indexs_raw ", max_indexs_raw,"max_indexs ", max_indexs, "max_index_amp ", amp_array[int(max_index)]," lows ", low, ,"\n upps ", upp, f"segment {len(segment)}, {segment[0]}, {segment[-1]}")
        # 输出切割后的片段
        for i, segment in enumerate(segments):
            amps_bs.append(np.sqrt(np.sum(np.square(segment))))
        index_range_array =self.formatted_print(max_indexs_raw, max_indexs, amp_array, lows, upps, segments, segment_maxs, amps_bs)
        return amps_bs,index_range_array

    def min_index(self, amp_array, max_indexs):
        # 步骤 1: 排序 list1，得到 list2
        max_indexs = np.insert(max_indexs, 0, 0)
        max_indexs_sorted = sorted(max_indexs.tolist())
        max_indexs_sorted = [int(x) for x in max_indexs_sorted]
        # 步骤 2: 使用 list2 对 amp 进行划分
        # 我们要将 list2 划分为多个区间
        segments = []
        for i in range(len(max_indexs_sorted) - 1):
            segments.append(amp_array[max_indexs_sorted[i]:int(max_indexs_sorted[i + 1] + 1)])  # 把每段的元素提取出来

        if max_indexs_sorted[-1] != len(amp_array) - 1:
            segments.append(amp_array[max_indexs_sorted[-1]:])  # 最后一段
        # 步骤 3: 找到每段的最小值索引
        min_indices = []
        for segment in segments:
            min_value = np.min(segment)  # 找到最小值
            min_index = np.where(amp_array == min_value)[0][0]  # 找到最小值的索引
            min_indices.append(min_index)
        if min_indices[0]!= 0:
            min_indices.insert(0, 0)
        if min_indices[-1]!=len(amp_array)-1:
            min_indices.append(len(amp_array)-1)
        return min_indices

    # 定义一个函数，返回最接近的两个索引 low 和 upp
    #找到大于 value 的最小元素，找到小于 value 的最大元素
    def find_closest_indices(self, value, min_indices):
        # 初始化 low 和 upp，设置为 -1 作为默认值
        low, upp = -1, -1

        # 遍历 min_indices，找到最接近的左右索引
        for i in range(1, len(min_indices)):
            if min_indices[i] > value:
                upp = min_indices[i]  # 找到大于 value 的最小元素
                low = min_indices[i - 1]  # 找到小于 value 的最大元素
                break

        return low, upp
