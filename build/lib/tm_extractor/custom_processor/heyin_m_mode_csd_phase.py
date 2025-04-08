import os
from collections import Counter
import math
from typing import Tuple
from jddb.processor import Signal, Shot, ShotSet, BaseProcessor
from typing import Optional, List
from scipy.fftpack import fft
from copy import deepcopy
import pandas as pd
import numpy as np
from scipy import signal as sig
import matplotlib.pyplot as plt
from jddb.file_repo import FileRepo
from jddb.processor import *
class HEyin_M_mode_csd_phase(BaseProcessor):
    def __init__(self,theta,plt_path,plot_phase=False  ):
        super().__init__(theta=theta,plt_path=plt_path,plot_phase=False)

        ma_pol_tags = ["\\MA_POLC_P01"]
        for i in range(2, 7):  # 从2递增到48
            ma_pol_tags.append(f"\\MA_POLC_P{str(i).zfill(2)}")
        for i in range(1, 7):  # 从1递增到48
            ma_pol_tags.append(f"\\MA_POLB_P{str(i).zfill(2)}")
        for i in range(1, 7):  # 从1递增到48
            ma_pol_tags.append(f"\\MA_POLD_P{str(i).zfill(2)}")
        for i in range(1, 7):  # 从1递增到48
            ma_pol_tags.append(f"\\MA_POLA_P{str(i).zfill(2)}")
        self.samplerate = 50000
        self.ma_pol_tags = ma_pol_tags
        self.plt_path = plt_path
        self.plot_phase = plot_phase
        self.number_of_a_window = 250  # 示例窗长度
        self.down_number = 1  # 示例下采样率
        self.theta = theta
        self.shot_no=0
        self.samplerate = 50000
        # theta_c = np.array([-75,-60,-45,-30,-15,0])/180
        # theta_c = np.array([-105, -120, -135, -150, -165, -172.5]) / 180
        # theta_b = np.array([-75, -60, -45, -30, -15, 0]) / 180
        # theta_d = np.array([15, 30, 45, 60, 75, 82.5]) / 180
        # theta_a = np.array([105, 120, 135, 150, 165, 180]) / 180
        # theta = np.concatenate([theta_c, theta_b, theta_d, theta_a])
    def transform(self,  *signal: Signal):
        # #signal[0]:B_LFS_most_signal
        # #signal[1]:B_LFS_sec_signal
        # #signal[2]:B_LFS_third_signal
        # #signal[3]:B_LFS_forth_signal
        mir_array_slice_signal_list = [deepcopy(signal.__getitem__(i)) for i in range(4,len(signal))]
        # B_LFS_most_signal_list=[deepcopy(signal.__getitem__(i)) for i in range(1,5)]
        B_LFS_most_signal = deepcopy(signal.__getitem__(0))
        B_LFS_sec_signal = deepcopy(signal.__getitem__(1))
        B_LFS_third_signal = deepcopy(signal.__getitem__(2))
        B_LFS_forth_signal = deepcopy(signal.__getitem__(3))
        phases_modified_signal = deepcopy(signal.__getitem__(0))
        min_zero_max_index_signal = deepcopy(signal.__getitem__(0))
        self.shot_no = B_LFS_most_signal.parent.shot_no
        self.number_of_a_window = len(signal.__getitem__(4).data[0])
        self.samplerate=mir_array_slice_signal_list[0].attributes['OriginalSampleRate']
        mode_number_array=np.zeros((len(B_LFS_most_signal.data),4),dtype=int)
        phases_modified_data = np.zeros((len(B_LFS_most_signal.data),4,len(self.theta)))
        min_zero_max_index_data = np.zeros((len(B_LFS_most_signal.data), 4, 3,len(self.theta)),dtype=int)
        for i in range(len(B_LFS_most_signal.data)):
            if B_LFS_most_signal.data[i][0]!=0:
                mode_number_array_i=np.array([])
                #传入一个时间段的阵列
                #2324
                chip_data_list=[-mir_array_slice_signal_list[signal_idx].data[i] if signal_idx<6 else mir_array_slice_signal_list[signal_idx].data[i] for signal_idx in range(len(mir_array_slice_signal_list))]
                #heyin
                # chip_data_list = [signal_idx.data[i] for signal_idx in mir_array_slice_signal_list]
                #四个频率的最大值
                max_frequency_list = [signal.__getitem__(a).data[i][2] for a in range(0,4)]
                for j in range(len(max_frequency_list)):
                    if max_frequency_list[j]!=0:
                        frequencies, phases = self.calculate_phase_and_frequency(chip_data_list, max_frequency_list[j])
                        #2324
                        # phases_modified = self.apply_function_within_range(phases, range(0, 6))
                        # phases_modified = self.apply_function_within_range(phases_modified, range(12, 18))
                        #heyin
                        phases_modified=phases
                        #明天写这个
                        #self.find_mode_number(phases_modified) 返回mode_number 返回值为整型
                        index_array_i,mode_number =self.find_mode_number(phases_modified)
                        mode_number_array_i = np.append(mode_number_array_i,mode_number)
                        min_zero_max_index_data[i][j]=index_array_i
                        if (not np.all(phases == 0)) and len(self.theta)==len(phases):
                            phases_modified_data[i][j]=phases_modified
                            if self.plot_phase:
                                self.plot_frequencies_and_phases(frequencies, phases, phases_modified, self.theta, self.shot_no, i, j,mode_number,B_LFS_most_signal.time[i])
                    else:
                        mode_number_array_i = np.append(mode_number_array_i,int(0))
                    # 调用绘图函数
                mode_number_array[i]=mode_number_array_i
        B_LFS_most_signal.data=mode_number_array
        phases_modified_signal.data = phases_modified_data
        min_zero_max_index_signal.data=min_zero_max_index_data
        #这里其实想返回四个B_LFS_most_signal、B_LFS_sec_signal。。。 ，再加一列作为m的值
        #并且加一列单个csd的值
        #我觉得后后面可以再写一个precessor，专门做这个事
        return B_LFS_most_signal,phases_modified_signal,min_zero_max_index_signal

    def find_mode_number(self, phases_modified):
        #min_number 满足phases_modified左边>0,右边<0且右>=自己的元素有几个
        #max_number 满足phases_modified左边<0,且左<=自己的，右边>0元素有几个
        #zero_number 满足phases_modified左边<0,右边>0元素有几个
        #这里的phases_modified是一个一维数组
        #mode_number= [min_number,max_number,zero_number]中出现次数最多的那个值
        min_number=0
        max_number=0
        zero_number=0
        min_index=np.array([],dtype=int)
        max_index=np.array([],dtype=int)
        zero_index=np.array([],dtype=int)
        index_array_i=np.zeros((3,len(phases_modified)),dtype=int)
        for i in range(len(phases_modified)):
            if i==0:
                if phases_modified[i]<0:
                    min_number+=1
                    min_index=np.append(min_index,i)
                if phases_modified[i]==0:
                    zero_number+=1
                    zero_index = np.append(zero_index, i)
                elif phases_modified[i]<0 and phases_modified[i+1]>0:
                    zero_number+=1
                    zero_index = np.append(zero_index, i)
            elif i==len(phases_modified)-1:
                if phases_modified[i]>0:
                    max_number+=1
                    max_index = np.append(max_index, i)
                if phases_modified[i]==0:
                    zero_number+=1
                    zero_index = np.append(zero_index, i)
            else:
                if phases_modified[i-1]>0 and phases_modified[i]<0 and phases_modified[i+1]>=phases_modified[i] and phases_modified[i+1]<0:
                    min_number+=1
                    min_index = np.append(min_index, i)
                if phases_modified[i-1]>0 and phases_modified[i]>0 and phases_modified[i-1]<=phases_modified[i] and phases_modified[i+1]<0:
                    max_number+=1
                    max_index = np.append(max_index, i)
                if (phases_modified[i]<0 and phases_modified[i+1]>0) or phases_modified[i]==0:
                    zero_number+=1
                    zero_index = np.append(zero_index, i)
        index_array_i[0,:len(min_index)]=min_index
        index_array_i[1,:len(zero_index)]=zero_index
        index_array_i[2,:len(max_index)]=max_index
        counter = Counter([min_number,max_number,zero_number])
        mode_number_csd_phase, count = counter.most_common(1)[0]
        return index_array_i,mode_number_csd_phase

    def find_max_csd_index(self, f, csd, max_frequency,four_fre_index_range):
        """在给定的积分区间，找到最大值，求出索引"""
        # 找到最大值的索引
        # max_idx = np.argmax(four_fre_index_range[:,2])
        lim_max_idx=0
        max_idx =np.where(four_fre_index_range[:, 2] == max_frequency)[0]
        if len(max_idx)!=0 and max_idx[0]!=0:
            fre_lower=four_fre_index_range[max_idx[0]][0]
            fre_upp=four_fre_index_range[max_idx[0]][-1]
            if fre_lower==fre_upp:
                fre_upp=fre_upp+1
            max_fre=four_fre_index_range[max_idx[0]][2]
            # 限制最大频率
            csd_lower_to_upp = csd[(f >= fre_lower) & (f <= fre_upp)]

            lim_max_idx = int(np.argmax(np.abs(csd_lower_to_upp))+np.searchsorted(f, fre_lower))
        return lim_max_idx

    def find_idx(self,f,max_frequency):
        max_idx = np.where(f == max_frequency)[0]
        if len(max_idx) != 0:
            return max_idx[0]
        else:
            return 0

    def calculate_phase_and_frequency(self,chip_data_list,max_frequency):
        """
        计算相位和频率
        """
        # chip_data_list 是 n 个一维数组组成的列表
        chip_data1 = chip_data_list[6]  # 第个数组
        phases = []
        frequencies = []

        # 遍历列表中的其他数组
        for i in range(len(chip_data_list)):
            chip_data2 = chip_data_list[i]

            # 计算互功率谱密度 (CSD)
            f, csd = sig.csd(chip_data1, chip_data2, fs=self.samplerate, window='hann',
                             nperseg=int(self.number_of_a_window / self.down_number),
                             scaling='density')
            # 找到 CSD 幅值最大处的索引
            # max_idx =self.find_max_csd_index(f, csd, max_frequency,four_fre_index_range)
            max_idx=self.find_idx(f,max_frequency)
            if max_idx==0:
                phase=np.nan
                frequency=0
            else:
                # 获取最大幅值处的相位
                phase = np.angle(csd[max_idx])
                phases.append(phase)

                # 获取最大幅值对应的频率
                frequency = f[max_idx]
            frequencies.append(frequency)
        # 遍历列表中的其他数组

        return frequencies, [phase / np.pi for phase in phases]

    def find_index_range(self,data, threshold=0.523, num_points=250):
        # 找到第一个大于 threshold 的点的索引
        """
            # 示例数据
            array1 = np.random.random(1000)  # 示例数组1
            array2 = np.random.random(1000)  # 示例数组2

            # 获取索引范围
            index_range = find_index_range(array1)

            # 使用索引范围提取 array2 中的子数组
            sub_array2 = extract_subarray(array2, index_range)

            print("索引范围:", list(index_range))
            print("提取的子数组:", sub_array2)
        """
        start_index = np.argmax(data > threshold)
        # 计算结束索引，确保不超出数组边界
        end_index = start_index + num_points
        if end_index > len(data):
            end_index = len(data)
        # 返回索引范围
        index_range = range(start_index, end_index)
        return index_range

    def extract_subarray(self,data, index_range):
        # 提取索引范围内的子数组
        return data[index_range]

    def plot_frequencies_and_phases(self,frequencies, phases, phases_modified, theta, shot_no, i, j, mode_number,time):
        fig, axs = plt.subplots(3, 1, figsize=(10, 12))

        # 绘制第一个子图：frequencies 的散点图
        axs[0].scatter(theta, frequencies, facecolors='none', edgecolors='b', s=100)  # 空心圆圈
        axs[0].set_ylabel('Frequency (Hz)')
        axs[0].set_xlabel('Theta')
        axs[0].set_title('Frequency vs Theta')
        axs[0].grid(True)

        # 在 x = -1, -0.5, 0, 0.5 处画竖线
        for x_line in [-1, -0.5, 0, 0.5]:
            axs[0].axvline(x=x_line, color='gray', linestyle='--', linewidth=1)

        # 绘制第二个子图：phases 的散点图

        axs[1].scatter(theta, phases, facecolors='none', edgecolors='r', s=100)  # 空心圆圈

        # axs[1].scatter(  theta,phases_modified,facecolors='none', edgecolors='b', s=100)  # 空心圆圈
        axs[1].set_ylabel('Phase (radians)')
        axs[1].set_xlabel('Theta')
        axs[1].set_title('Phase vs Theta')
        axs[1].grid(True)
        axs[1].set_ylim(-1, 1)

        # 在 x = -1, -0.5, 0, 0.5 处画竖线

        for x_line in [-1, -0.5, 0, 0.5]:
            axs[1].axvline(x=x_line, color='gray', linestyle='--', linewidth=1)

        # 绘制第二个子图：phases 的散点图
        # axs[2].scatter(  theta,phases,facecolors='none', edgecolors='r', s=100)  # 空心圆圈

        axs[2].scatter(theta, phases_modified, facecolors='none', edgecolors='b', s=100)  # 空心圆圈
        axs[2].set_ylabel('Phase (radians)')
        axs[2].set_xlabel('Theta')
        axs[2].set_title(f'Phase vs Theta mode_number {mode_number}')
        axs[2].grid(True)
        axs[2].set_ylim(-1, 1)

        # 在 x = -1, -0.5, 0, 0.5 处画竖线

        for x_line in [-1, -0.5, 0, 0.5]:
            axs[2].axvline(x=x_line, color='gray', linestyle='--', linewidth=1)
        plt.tight_layout()

        # 保存图像
        folder_path=os.path.join(self.plt_path,f"{shot_no}" ,f"{shot_no}_num-{j}")
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        save_path = os.path.join(folder_path,f"{shot_no}-{i}_time-{time}_num-{j}_phase.png")
        # save_path = f"E:\笔记本G盘\mypool_lry_jtext_download_dataset\\2023_ip_downtime\heyin_mode_th1\plt\\{shot_no}_time-{i}_num-{j}_phase.png"
        plt.savefig(save_path)
        plt.close(fig)  # 关闭图像，避免在交互式环境中显示

    def apply_function_within_range(self, array, index_range):
        # 创建一个新的数组，复制原数组的数据
        modified_array = deepcopy(array)

        # 在指定的索引范围内应用函数
        for i in index_range:
            # 可能会出错的代码
            if array[i] <= 0.5:
                y = array[i] + 0.5
            else:
                y = array[i] - 1.5
            modified_array[i] = y
        return modified_array





# class HEyin_M_mode_csd_phase(BaseProcessor):
#     def __init__(self,theta,plt_path,plot_phase=False  ):
#         super().__init__(theta=theta,plt_path=plt_path,plot_phase=False)
#
#         ma_pol_tags = ["\\MA_POLC_P01"]
#         for i in range(2, 7):  # 从2递增到48
#             ma_pol_tags.append(f"\\MA_POLC_P{str(i).zfill(2)}")
#         for i in range(1, 7):  # 从1递增到48
#             ma_pol_tags.append(f"\\MA_POLB_P{str(i).zfill(2)}")
#         for i in range(1, 7):  # 从1递增到48
#             ma_pol_tags.append(f"\\MA_POLD_P{str(i).zfill(2)}")
#         for i in range(1, 7):  # 从1递增到48
#             ma_pol_tags.append(f"\\MA_POLA_P{str(i).zfill(2)}")
#         self.samplerate = 50000
#         self.ma_pol_tags = ma_pol_tags
#         self.plt_path = plt_path
#         self.plot_phase = plot_phase
#         self.number_of_a_window = 250  # 示例窗长度
#         self.down_number = 1  # 示例下采样率
#         self.theta = theta
#         self.shot_no=0
#         self.samplerate = 50000
#         # theta_c = np.array([-75,-60,-45,-30,-15,0])/180
#         # theta_c = np.array([-105, -120, -135, -150, -165, -172.5]) / 180
#         # theta_b = np.array([-75, -60, -45, -30, -15, 0]) / 180
#         # theta_d = np.array([15, 30, 45, 60, 75, 82.5]) / 180
#         # theta_a = np.array([105, 120, 135, 150, 165, 180]) / 180
#         # theta = np.concatenate([theta_c, theta_b, theta_d, theta_a])
#
#     def transform(self,  *signal: Signal):
#         # #signal[0]:MA_POLB_P06
#         # #signal[1]:B_LFS_Inte
#         # #signal[2]:Ip
#         # #signal[3]:qa#在这一步计算一次qa_raw
#         mir_array_slice_signal_list = [deepcopy(signal.__getitem__(i)) for i in range(5,len(signal))]
#
#         B_LFS_n_amp_inte_range_signal = deepcopy(signal.__getitem__(0))
#         # B_LFS_most_signal_list=[deepcopy(signal.__getitem__(i)) for i in range(1,5)]
#         B_LFS_most_signal = deepcopy(signal.__getitem__(1))
#         B_LFS_sec_signal = deepcopy(signal.__getitem__(2))
#         B_LFS_third_signal = deepcopy(signal.__getitem__(3))
#         B_LFS_forth_signal = deepcopy(signal.__getitem__(4))
#         phase_modified_signal = deepcopy(signal.__getitem__(1))
#         self.shot_no = B_LFS_most_signal.parent.shot_no
#         self.number_of_a_window = len(signal.__getitem__(5).data[0])
#         self.samplerate=mir_array_slice_signal_list[0].attributes['OriginalSampleRate']
#         mode_number_array=np.zeros((len(B_LFS_most_signal.data),4),dtype=int)
#         phases_modified=np.zeros((len(B_LFS_most_signal.data),len(self.theta)),dtype=float)
#         for i in range(len(B_LFS_most_signal.data)):
#             phases_modified_i=np.zeros(len(self.theta),dtype=float)
#             # if i==261:
#             #     debug_a=1
#             if B_LFS_most_signal.data[i][0]!=0:
#                 mode_number_array_i=np.array([])
#                 #传入一个时间段的阵列
#                 #2324
#                 # chip_data_list=[-signal_idx.data[i] if i<6 else signal_idx.data[i] for signal_idx in mir_array_slice_signal_list]
#                 #heyin
#                 chip_data_list = [signal_idx.data[i] for signal_idx in mir_array_slice_signal_list]
#                 #四个频率的最大值
#                 max_frequency_list = [signal.__getitem__(a).data[i][2] for a in range(1,5)]
#                 for j in range(len(max_frequency_list)):
#                     if j==1:
#                         debug_a=1
#                     if max_frequency_list[j]!=0:
#                         frequencies, phases = self.calculate_phase_and_frequency(chip_data_list, max_frequency_list[j],B_LFS_n_amp_inte_range_signal.data[i])
#                         #2324
#                         # phases_modified = self.apply_function_within_range(phases, range(0, 6))
#                         # phases_modified = self.apply_function_within_range(phases_modified, range(12, 18))
#                         #明天写这个
#                         #self.find_mode_number(phases_modified) 返回mode_number 返回值为整型
#                         #2324去掉
#                         phases_modified_i=phases
#                         mode_number =self.find_mode_number(phases_modified_i)
#                         mode_number_array_i = np.append(mode_number_array_i,mode_number)
#                         if not np.all(phases == 0) and len(self.theta)==len(phases) and self.plot_phase:
#                             self.plot_frequencies_and_phases(frequencies, phases, phases_modified_i, self.theta, self.shot_no, i, j,mode_number,B_LFS_most_signal.time[i])
#                     else:
#                         mode_number_array_i = np.append(mode_number_array_i,int(0))
#                     # 调用绘图函数
#                 mode_number_array[i]=mode_number_array_i
#                 phases_modified[i]=phases_modified_i
#         B_LFS_most_signal.data=mode_number_array
#         phase_modified_signal.data=phases_modified
#         #这里其实想返回四个B_LFS_most_signal、B_LFS_sec_signal。。。 ，再加一列作为m的值
#         #并且加一列单个csd的值
#         #我觉得后后面可以再写一个precessor，专门做这个事
#         return B_LFS_most_signal,phase_modified_signal
#
#     def find_mode_number(self, phases_modified):
#         #min_number 满足phases_modified左边>0,右边<0且右>=自己的元素有几个
#         #max_number 满足phases_modified左边<0,且左<=自己的，右边>0元素有几个
#         #zero_number 满足phases_modified左边<0,右边>0元素有几个
#         #这里的phases_modified是一个一维数组
#         #mode_number= [min_number,max_number,zero_number]中出现次数最多的那个值
#         min_number=0
#         max_number=0
#         zero_number=0
#         for i in range(len(phases_modified)):
#             if i==0:
#                 if phases_modified[i]<0:
#                     min_number+=1
#                 if phases_modified[i]==0:
#                     zero_number+=1
#                 elif phases_modified[i]<0 and phases_modified[i+1]>0:
#                     zero_number+=1
#             elif i==len(phases_modified)-1:
#                 if phases_modified[i]>0:
#                     max_number+=1
#                 if phases_modified[i]==0:
#                     zero_number+=1
#             else:
#                 if phases_modified[i-1]>0 and phases_modified[i]<0 and phases_modified[i+1]>=phases_modified[i] and phases_modified[i+1]<0:
#                     min_number+=1
#                 if phases_modified[i-1]>0 and phases_modified[i]>0 and phases_modified[i-1]<=phases_modified[i] and phases_modified[i+1]<0:
#                     max_number+=1
#                 if (phases_modified[i]<0 and phases_modified[i+1]>0) or phases_modified[i]==0:
#                     zero_number+=1
#         counter = Counter([min_number,max_number,zero_number])
#         mode_number_csd_phase, count = counter.most_common(1)[0]
#         return mode_number_csd_phase
#
#     def find_max_csd_index(self, f, csd, max_frequency,four_fre_index_range):
#         """在给定的积分区间，找到最大值，求出索引"""
#         # 找到最大值的索引
#         # max_idx = np.argmax(four_fre_index_range[:,2])
#         lim_max_idx=0
#         max_idx =np.where(four_fre_index_range[:, 2] == max_frequency)[0]
#         if len(max_idx)!=0 and max_idx[0]!=0:
#             fre_lower=four_fre_index_range[max_idx[0]][0]
#             fre_upp=four_fre_index_range[max_idx[0]][-1]
#             if fre_lower==fre_upp:
#                 fre_upp=fre_upp+1
#             max_fre=four_fre_index_range[max_idx[0]][2]
#             # 限制最大频率
#             csd_lower_to_upp = csd[(f >= fre_lower) & (f <= fre_upp)]
#
#             lim_max_idx = int(np.argmax(np.abs(csd_lower_to_upp))+np.searchsorted(f, fre_lower))
#         return lim_max_idx
#
#     def find_idx(self,f,max_frequency):
#         max_idx = np.where(f == max_frequency)[0]
#         if len(max_idx) != 0:
#             return max_idx[0]
#         else:
#             return 0
#
#     def calculate_phase_and_frequency(self,chip_data_list,max_frequency,four_fre_index_range):
#         """
#         计算相位和频率
#         """
#         # chip_data_list 是 n 个一维数组组成的列表
#         chip_data1 = chip_data_list[6]  # 第个数组
#         phases = []
#         frequencies = []
#
#         # 遍历列表中的其他数组
#         for i in range(len(chip_data_list)):
#             chip_data2 = chip_data_list[i]
#
#             # 计算互功率谱密度 (CSD)
#             f, csd = sig.csd(chip_data1, chip_data2, fs=self.samplerate, window='hann',
#                              nperseg=int(self.number_of_a_window / self.down_number),
#                              scaling='density')
#             # 找到 CSD 幅值最大处的索引
#             # max_idx =self.find_max_csd_index(f, csd, max_frequency,four_fre_index_range)
#             max_idx=self.find_idx(f,max_frequency)
#             if max_idx==0:
#                 phase=np.nan
#                 frequency=0
#             else:
#                 # 获取最大幅值处的相位
#                 phase = np.angle(csd[max_idx])
#                 phases.append(phase)
#
#                 # 获取最大幅值对应的频率
#                 frequency = f[max_idx]
#             frequencies.append(frequency)
#         # 遍历列表中的其他数组
#
#         return frequencies, [phase / np.pi for phase in phases]
#
#     def find_index_range(self,data, threshold=0.523, num_points=250):
#         # 找到第一个大于 threshold 的点的索引
#         """
#             # 示例数据
#             array1 = np.random.random(1000)  # 示例数组1
#             array2 = np.random.random(1000)  # 示例数组2
#
#             # 获取索引范围
#             index_range = find_index_range(array1)
#
#             # 使用索引范围提取 array2 中的子数组
#             sub_array2 = extract_subarray(array2, index_range)
#
#             print("索引范围:", list(index_range))
#             print("提取的子数组:", sub_array2)
#         """
#         start_index = np.argmax(data > threshold)
#         # 计算结束索引，确保不超出数组边界
#         end_index = start_index + num_points
#         if end_index > len(data):
#             end_index = len(data)
#         # 返回索引范围
#         index_range = range(start_index, end_index)
#         return index_range
#
#     def extract_subarray(self,data, index_range):
#         # 提取索引范围内的子数组
#         return data[index_range]
#
#     def plot_frequencies_and_phases(self,frequencies, phases, phases_modified, theta, shot_no, i, j, mode_number,time):
#         fig, axs = plt.subplots(3, 1, figsize=(10, 12))
#
#         # 绘制第一个子图：frequencies 的散点图
#         axs[0].scatter(theta, frequencies, facecolors='none', edgecolors='b', s=100)  # 空心圆圈
#         axs[0].set_ylabel('Frequency (Hz)')
#         axs[0].set_xlabel('Theta')
#         axs[0].set_title('Frequency vs Theta')
#         axs[0].grid(True)
#
#         # 在 x = -1, -0.5, 0, 0.5 处画竖线
#         for x_line in [-1, -0.5, 0, 0.5]:
#             axs[0].axvline(x=x_line, color='gray', linestyle='--', linewidth=1)
#
#         # 绘制第二个子图：phases 的散点图
#
#         axs[1].scatter(theta, phases, facecolors='none', edgecolors='r', s=100)  # 空心圆圈
#
#         # axs[1].scatter(  theta,phases_modified,facecolors='none', edgecolors='b', s=100)  # 空心圆圈
#         axs[1].set_ylabel('Phase (radians)')
#         axs[1].set_xlabel('Theta')
#         axs[1].set_title('Phase vs Theta')
#         axs[1].grid(True)
#         axs[1].set_ylim(-1, 1)
#
#         # 在 x = -1, -0.5, 0, 0.5 处画竖线
#
#         for x_line in [-1, -0.5, 0, 0.5]:
#             axs[1].axvline(x=x_line, color='gray', linestyle='--', linewidth=1)
#
#         # 绘制第二个子图：phases 的散点图
#         # axs[2].scatter(  theta,phases,facecolors='none', edgecolors='r', s=100)  # 空心圆圈
#
#         axs[2].scatter(theta, phases_modified, facecolors='none', edgecolors='b', s=100)  # 空心圆圈
#         axs[2].set_ylabel('Phase (radians)')
#         axs[2].set_xlabel('Theta')
#         axs[2].set_title(f'Phase vs Theta mode_number {mode_number}')
#         axs[2].grid(True)
#         axs[2].set_ylim(-1, 1)
#
#         # 在 x = -1, -0.5, 0, 0.5 处画竖线
#
#         for x_line in [-1, -0.5, 0, 0.5]:
#             axs[2].axvline(x=x_line, color='gray', linestyle='--', linewidth=1)
#         plt.tight_layout()
#
#         # 保存图像
#         folder_path=os.path.join(self.plt_path, f"{shot_no}_num-{j}")
#         if not os.path.exists(folder_path):
#             os.makedirs(folder_path)
#         save_path = os.path.join(self.plt_path, f"{shot_no}_num-{j}",f"{shot_no}-{i}_time-{time}_num-{j}_phase.png")
#         # save_path = f"E:\笔记本G盘\mypool_lry_jtext_download_dataset\\2023_ip_downtime\heyin_mode_th1\plt\\{shot_no}_time-{i}_num-{j}_phase.png"
#         plt.savefig(save_path)
#         plt.close(fig)  # 关闭图像，避免在交互式环境中显示
#
#     def apply_function_within_range(self, array, index_range):
#         # 创建一个新的数组，复制原数组的数据
#         modified_array = deepcopy(array)
#
#         # 在指定的索引范围内应用函数
#         for i in index_range:
#             if array[i] <= 0.5:
#                 y = array[i] + 0.5
#             else:
#                 y = array[i] - 1.5
#             modified_array[i] = y
#         return modified_array
