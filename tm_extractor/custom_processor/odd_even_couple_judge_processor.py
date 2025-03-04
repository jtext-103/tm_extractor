# -*-  coding:utf-8  -*-
import numpy as np
from jddb.processor import Signal, BaseProcessor
from copy import deepcopy

class OddEvenCoupleJudgeProcessor(BaseProcessor):
    """
        input:
            OddEvenCoupleJudgeProcessor给出奇偶分离以及倍频/非倍频模式，可以判断出是否存在耦合模式
           couple_signal:time*2 第一个1表示 奇偶分离，第二个1表示倍频
            1.非奇偶+非倍频====>无耦合：0,0  [0,0,0,-1], [freq1,freq2,freq3,-1]
            2.非奇偶+倍频======>不确定，要看csd：0,1 [freq1,freq2,freq3,-1] [freq1,freq2,freq3,-1]
            3.奇偶+非倍频======>耦合：1,0 [freq1,0,0,-1] [0,freq2,0,-1]
            4.奇偶+倍频======>耦合：1,1 [freq1,0,0,-1] [0,0,0,-1]
           couple_fre_signal:time*4 0表示不是耦合的，fre表示耦合，-1表示无
           uncouple_fre_signal:time*4 fre表示无耦合，0表示耦合的，-1表示无
         #在1的情况下直接输入模数
         #在2的情况下，需要输入csd以及花瓣模数， self.phase花瓣模数、耦合的时候看斜率不对的情况=>self.双奇数双偶数耦合(是奇数还是偶数是由phase花瓣数确定的)
         #在3的情况下，需要输入csd以及花瓣模数，self.phase花瓣模数(给出了一个奇数/偶数)、self.phase花瓣模数、耦合的时候看斜率不对的情况
         #在4的情况下，需要输入csd以及花瓣模数，重新确定一下模数
        def phase花瓣模数、耦合的时候看斜率不对的情况
        def csd求整数、求耦合的数
        def 奇偶耦合
        def 双奇数双偶数耦合(是奇数还是偶数是由phase花瓣数确定的)
        Returns:
            couple_mode_signal：time*4*2（两个m）*2（两个n）
            uncouple_mode_signal：time*4*2（1个m,1个n）
    """

    def __init__(self, threshold=2):  # pad 表示f精度，精度为 1/(fs*pad),f 的长度为 pad*fs/2
        super().__init__(threshold=threshold)
        self.threshold=threshold
    def transform(self, *signal: Signal):
        # #signal[0]:mode_fre_signal
        # #signal[1]:undouble_fre_signal
        # #signal[2]:even_m_most_max_all
        # #signal[3]:even_m_sec_max_all
        # #signal[4]:even_m_third_max_all
        # #signal[5]:even_m_forth_max_all
        # #signal[6]:odd_m_most_max_all
        # #signal[7]:odd_m_sec_max_all
        # #signal[8]:odd_m_third_max_all
        # #signal[9]:odd_m_forth_max_all
        mode_fre_signal = deepcopy(signal.__getitem__(0))
        undouble_fre_signal = deepcopy(signal.__getitem__(1))
        even_n_signal = [deepcopy(signal.__getitem__(i)) for i in range(2,6)]
        odd_n_signal = [deepcopy(signal.__getitem__(i)) for i in range(6,10)]
        couple_signal = deepcopy(signal.__getitem__(0))
        couple_fre_signal= deepcopy(signal.__getitem__(0))
        uncouple_fre_signal = deepcopy(signal.__getitem__(0))
        couple_state_signal = deepcopy(signal.__getitem__(0))

        couple_fre_data=np.zeros((len(mode_fre_signal.data), 4), dtype=float)
        uncouple_fre_data=np.zeros((len(mode_fre_signal.data), 4), dtype=float)
        couple_state_data=np.zeros(shape=(len(mode_fre_signal.data), 2), dtype=int)

        for i in range(len(mode_fre_signal.data)):
            if np.sum(mode_fre_signal.data[i])>0:
                even_n_signal_i=np.array([even_n_signal[j].data[i] for j in range(len(even_n_signal))])
                odd_n_signal_i = np.array([odd_n_signal[j].data[i] for j in range(len(odd_n_signal))])
                couple_double_state, couple_double_fre_i = self.couple_double_judge(mode_fre_signal.data[i], undouble_fre_signal.data[i])
                #couple_double_fre_i指的是可以作为独立撕裂模的频率，0表示不能作为独立撕裂模的频率，-1表示<2Gs不考虑这个频率
                couple_odd_even_state, couple_odd_even_fre_i, uncouple_odd_even_fre_i = self.odd_even_judge_couple(couple_double_fre_i, even_n_signal_i, odd_n_signal_i)
                # 如果奇偶分离，且与基频非常近似，则认为只有基频,这个就是在undouble_fre_signal中不为1的，我们就不考虑这个fre了
                # 倍频需要检查一下有没有n=2且 如基频2/1 倍频不是4/2的这种情况(这个先不做了，如果是耦合的情况下很容易测不准，这个不考虑吧)
                #    1.非奇偶+非倍频====>无耦合：0,0  [0,0,0,-1], [freq1,freq2,freq3,-1]
                #    2.非奇偶+倍频======>不确定，要看csd：0,1 [freq1,0,0,-1] [0,0,0,-1]
                #    3.奇偶+非倍频======>耦合：1,0 [freq1,0,0,-1] [0,freq2,0,-1]
                #    4.奇偶+倍频======>耦合：1,1 [freq1,0,0,-1] [0,0,0,-1]
                if couple_odd_even_state==0 and couple_double_state==0 :#非奇偶+非倍频====>无耦合
                    uncouple_fre_data[i]=couple_double_fre_i
                    couple_fre_data[i][couple_double_fre_i==-1]=-1
                    couple_state_data[i]=np.array([0,0])
                elif couple_odd_even_state==0 and couple_double_state==1:#非奇偶+倍频======>不确定
                    couple_fre_data[i]=couple_double_fre_i
                    uncouple_fre_data[i][couple_odd_even_fre_i==-1]=-1
                    couple_state_data[i] = np.array([0, 1])
                elif couple_odd_even_state==1 and couple_double_state==0:#奇偶+非倍频======>耦合
                    couple_fre_data[i]=couple_odd_even_fre_i
                    uncouple_fre_data[i]=uncouple_odd_even_fre_i
                    couple_state_data[i] = np.array([1, 0])
                else:#奇偶+倍频======>耦合
                    couple_fre_data[i]=couple_odd_even_fre_i
                    uncouple_fre_data[i]=uncouple_odd_even_fre_i
                    couple_state_data[i] = np.array([1, 1])
            else:
                # couple_fre_data[i] = mode_fre_signal.data[i]
                # uncouple_fre_data[i] = mode_fre_signal.data[i]
                couple_fre_data[i] = np.array([-1, -1, -1, -1])
                uncouple_fre_data[i] = np.array([-1, -1, -1, -1])
                couple_state_data[i] = np.array([-1, -1])
            uncouple_fre_data[i][undouble_fre_signal.data[i] == -1] = -1
            couple_fre_data[i][undouble_fre_signal.data[i] == -1] = -1
            uncouple_fre_data[i][undouble_fre_signal.data[i] == 0] = -1
            couple_fre_data[i][undouble_fre_signal.data[i] == 0] = -1
        couple_signal.data=couple_state_data
        couple_fre_signal.data=couple_fre_data
        uncouple_fre_signal.data=uncouple_fre_data
        return couple_signal, couple_fre_signal, uncouple_fre_signal
    def odd_even_judge_couple(self, mode_fre_data, even_n_data, odd_n_data):
        couple=0
        couple_odd_even_fre_i=np.zeros((len(mode_fre_data)), dtype=int)
        uncouple_odd_even_fre_i=np.zeros((len(mode_fre_data)), dtype=int)
        for i in range(len(mode_fre_data)):
            if mode_fre_data[i]!=-1 and mode_fre_data[i]!=0:
                even_n_row_indices = np.where(even_n_data == mode_fre_data[i])[0]
                odd_n_row_indices = np.where(even_n_data == mode_fre_data[i])[0]
                # 可能出错的代码,假如不是耦合的时候，可能会出现奇偶分离odd even的幅值都不能满足》0.5的情况
                if len(odd_n_row_indices)>0 and len(even_n_row_indices)>0 :
                    if even_n_data[even_n_row_indices][0][1] >= self.threshold and odd_n_data[odd_n_row_indices][0][1] >= self.threshold:
                        couple = couple + 1
                        couple_odd_even_fre_i[i] = mode_fre_data[i]
                    else:
                        uncouple_odd_even_fre_i[i] = mode_fre_data[i]
        if couple>=1:
            couple_odd_even_state=1
        else:
            couple_odd_even_state=0
        couple_odd_even_fre_i[mode_fre_data==-1]=-1
        uncouple_odd_even_fre_i[mode_fre_data == -1] = -1
        return couple_odd_even_state, couple_odd_even_fre_i, uncouple_odd_even_fre_i
    def couple_double_judge(self, mode_fre_i, undouble_fre_i):
        couple_double_fre_i=deepcopy(mode_fre_i)
        couple_double_state = 0
        couple_double_fre_i[undouble_fre_i == 0] = 0
        couple_double_fre_i[undouble_fre_i == -1] = -1
        if np.sum(undouble_fre_i==1)==1:
            couple_double_state=1
        return couple_double_state, couple_double_fre_i