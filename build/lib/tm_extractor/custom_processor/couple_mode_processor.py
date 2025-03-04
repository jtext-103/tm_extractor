# -*-  coding:utf-8  -*-
import numpy as np
from jddb.processor import Signal, BaseProcessor
from copy import deepcopy



class CoupleModeProcessor(BaseProcessor):
    """
         给出奇偶分离以及倍频/非倍频模式，可以判断出是否存在耦合模式
         1.非奇偶+非倍频====>无耦合：0,0  [0,0,0,-1], [freq1,freq2,freq3,-1]
         2.非奇偶+倍频======>不确定，要看csd：0,1 [freq1,freq2,freq3,-1] [freq1,freq2,freq3,-1]
              csd的花瓣太难看出来了，所以这个不确定,我们的标签名字就叫做奇偶分离的耦合标签
              # 但是n_cs按照[:0.3],[0.3,0.7],[0.7:]，这个不准，不用
         3.奇偶+非倍频======>耦合：1,0 [freq1,0,0,-1] [0,freq2,0,-1]
             if n_number=n_csd（四舍五入）时：
              1.例如m_number==2为偶数，那么n_csd推导奇数：
              2.[(2+1)/2,(2+3)/2,(2+5)/2,...]先挖去n_number=1的1/1，再挖去qa条件下的所有m/n如qa<5,5/1挖去；则为参数==>[2.5,3.5,...]
              3.if  只有一个[2.5],则无需选最接近的
                else：n_csd更接近哪个数组的元素，就认为m_number是几，
             if n_number=!n_csd（四舍五入）时：
              1.例如n_number=2,n_csd<1.5或>2.5,另一个n可能的值为[1,3,4,5...]=>取[(1+2)/2,(3+2)/2,(4+2)/2,....]中n_csd最接近的值
              2.例如n_number=1,n_csd>1.5,另一个n可能的值为[2,3,4,5...]=>取[(2+1)/2,(3+1)/2,(4+1)/2,....]中n_csd最接近的值
              3.再分析m_number
                  3.1.例如m_number==2为偶数，那么n_csd推导奇数：
                  3.2 [(2+1)/2,(2+3)/2,(2+5)/2,...]先挖去n_number，再挖去qa条件下的所有m/n取最大的n下不满足的，如m=8 即8/2挖去；则为参数==>[1.5,2.5,3.5,...]
                  3.3 if  只有一个[2.5],则无需选最接近的
                      else：n_csd更接近哪个数组的元素，就认为m_number是几，
              4.确定了如 m 2 、3 ;n 2、1 （2/1 3/2） 则需要分析这一炮完整的过程中有没有出现过确定的 这种有21 或者32的时间《这个新给个processor都可》如果没有就按照m 2<3；n 1<2排序分配

         4.奇偶+倍频======>耦合：1,1 [freq1,0,0,-1] [0,0,0,-1]
         3,4起始都是一个判断方法
         2.难度大一些，但情况少，还需要根据qa判断是不是准的
         Returns:
             couple_signal:time*2 第一个1表示 奇偶分离，第二个1表示倍频
             couple_fre_signal:time*4 0表示不是耦合的，fre表示耦合，-1表示无
             uncouple_fre_signal:time*4 fre表示无耦合，0表示耦合的，-1表示无
      """

    def __init__(self):  # pad 表示f精度，精度为 1/(fs*pad),f 的长度为 pad*fs/2
        super().__init__()
        self.shot_no=0

    def transform(self, *signal: Signal):
        # #signal[0]:fre_mode_signal #-1:无撕裂模，0:0.5~2，1:>2
        # #signal[2]:couple_state_signal
        # #signal[3]:couple_fre_signal
        # #signal[4]:uncouple_fre_signal
        # #signal[5]:phases_modified
        # #signal[6]:m_n_most_max_th
        # #signal[7]:m_n_sec_max_th
        # #signal[8]:m_n_third_max_th
        # #signal[9]:m_n_forth_max_th
        # #signal[10]:mode_n_number_signal
        # #signal[11]:even_most_max_signal
        # #signal[12]:even_sec_max_signal
        # #signal[13]:even_third_max_signal
        # #signal[14]:even_forth_max_signal
        # #signal[15]:odd_most_max_signal
        # #signal[16]:odd_sec_max_signal
        # #signal[17]:odd_third_max_signal
        # #signal[18]:odd_forth_max_signal
        # 5-8"\\new_B_LFS_n_m_most_judge", "\\new_B_LFS_n_m_sec_judge", "\\new_B_LFS_n_m_third_judge", "\\new_B_LFS_n_m_forth_judge"
        mode_fre_signal = deepcopy(signal.__getitem__(0))
        self.shot_no = mode_fre_signal.parent.shot_no
        couple_state_signal = deepcopy(signal.__getitem__(1))
        # mode_undouble_fre_signal=deepcopy(signal.__getitem__(2))
        couple_fre_signal = deepcopy(signal.__getitem__(2))
        uncouple_fre_signal = deepcopy(signal.__getitem__(3))
        phases_modified_signal = deepcopy(signal.__getitem__(4))
        m_n_most_max_judge_signal = deepcopy(signal.__getitem__(5))
        m_n_most_sec_judge_signal = deepcopy(signal.__getitem__(6))
        m_n_most_third_judge_signal = deepcopy(signal.__getitem__(7))
        m_n_most_forth_judge_signal = deepcopy(signal.__getitem__(8))
        even_max_signal = deepcopy(signal.__getitem__(9))
        even_sec_signal = deepcopy(signal.__getitem__(10))
        even_third_signal = deepcopy(signal.__getitem__(11))
        even_forth_signal = deepcopy(signal.__getitem__(12))
        odd_max_signal = deepcopy(signal.__getitem__(13))
        odd_sec_signal = deepcopy(signal.__getitem__(14))
        odd_third_signal = deepcopy(signal.__getitem__(15))
        odd_forth_signal = deepcopy(signal.__getitem__(16))
        qa_signal = deepcopy(signal.__getitem__(17))
        # 创建要添加的四列 (n 行 4 列)，比如全为 0
        new_columns = np.zeros((m_n_most_max_judge_signal.data.shape[0], 4), dtype=int)
        # 拼接数组
        m_n_most_max_judge_signal.data = np.hstack((m_n_most_max_judge_signal.data, new_columns))
        m_n_most_sec_judge_signal.data = np.hstack((m_n_most_sec_judge_signal.data, new_columns))
        m_n_most_third_judge_signal.data = np.hstack((m_n_most_third_judge_signal.data, new_columns))
        m_n_most_forth_judge_signal.data = np.hstack((m_n_most_forth_judge_signal.data, new_columns))
        # couple_mode_data=np.zeros((len(couple_state_signal.data), 4,2,2), dtype=float)#[[m,n],[m,n]]
        # uncouple_mode_data=np.zeros((len(couple_state_signal.data), 4,2), dtype=float)#[m,n]
        for i in range(  len(couple_state_signal.data)):
            m_n_most_data_i = np.array([m_n_most_max_judge_signal.data[i], m_n_most_sec_judge_signal.data[i],
                                        m_n_most_third_judge_signal.data[i], m_n_most_forth_judge_signal.data[i]])
            if np.sum(mode_fre_signal.data[i]) > 0:
                even_data_i = np.array([even_max_signal.data[i], even_sec_signal.data[i], even_third_signal.data[i],
                                        even_forth_signal.data[i]])
                odd_data_i = np.array([odd_max_signal.data[i], odd_sec_signal.data[i], odd_third_signal.data[i],
                                       odd_forth_signal.data[i]])
                if couple_state_signal.data[i, 0] == 0 and couple_state_signal.data[i, 1] == 0:  # 非奇偶+非倍频====>无耦合
                    m_n_most_data_i = self.uncouple_mode_data(m_n_most_data_i, mode_fre_signal.data[i],
                                                              uncouple_fre_signal.data[i])  # 修改一下，不是所有的都写，为0的写0000
                    # 只修改mode_fre_signal！=0的
                elif couple_state_signal.data[i, 0] == 0 and couple_state_signal.data[i, 1] == 1:  # 非奇偶+倍频======>不确定
                    # m_n_most_i=self.unsure_couple_data(m_n_most_data_i, couple_fre_signal.data[i],phases_modified_signal.data[i],mode_undouble_fre_signal.data[i])
                    m_n_most_data_i = self.uncouple_mode_data(m_n_most_data_i, mode_fre_signal.data[i],
                                                              couple_fre_signal.data[i])
                    m_n_most_data_i = self.uncouple_mode_data(m_n_most_data_i, mode_fre_signal.data[i],
                                                              uncouple_fre_signal.data[i])  # 修改一下，不是所有的都写，为0的写0000
                    couple_fre_data = np.array([0, 0, 0, 0])
                    couple_fre_data[uncouple_fre_signal.data[i] == -1] = -1
                    uncouple_fre_signal.data[i] = couple_fre_signal.data[i]
                    couple_fre_signal.data[i] = couple_fre_data

                elif couple_state_signal.data[i, 0] == 1:  # 奇偶+非倍频======>耦合,#奇偶+倍频======>耦合
                    # 处理couple_fre_signal.data[i]
                    # if (couple_fre_signal.data[i] == 5600).any():
                    #     a = 1
                    m_n_most_data_i = self.couple_sure_mode_data(m_n_most_data_i, couple_fre_signal.data[i],
                                                                 even_data_i, odd_data_i, qa_signal.data[i])
                    # 处理uncouple_fre_signal.data[i]
                    m_n_most_data_i = self.uncouple_mode_data(m_n_most_data_i, mode_fre_signal.data[i],
                                                              uncouple_fre_signal.data[i])
            else:
                m_n_most_data_i[0][-4:] = np.full((4), mode_fre_signal.data[i, 0])
                m_n_most_data_i[1][-4:] = np.full((4), mode_fre_signal.data[i, 1])
                m_n_most_data_i[2][-4:] = np.full((4), mode_fre_signal.data[i, 2])
                m_n_most_data_i[3][-4:] = np.full((4), mode_fre_signal.data[i, 3])
            m_n_most_max_judge_signal.data[i] = m_n_most_data_i[0]
            m_n_most_sec_judge_signal.data[i] = m_n_most_data_i[1]
            m_n_most_third_judge_signal.data[i] = m_n_most_data_i[2]
            m_n_most_forth_judge_signal.data[i] = m_n_most_data_i[3]
        return m_n_most_max_judge_signal, m_n_most_sec_judge_signal, m_n_most_third_judge_signal, m_n_most_forth_judge_signal, couple_fre_signal, uncouple_fre_signal

    def round_number(self, number):
        if number < 1.5:
            return 1
        else:
            integer_part = int(number)  # 获取整数部分
            decimal_part = number - integer_part  # 获取小数部分
            if decimal_part >= 0.55:
                return integer_part + 1  # 小数部分 >= 0.55，向上取整
            else:
                return integer_part  # 小数部分 < 0.55，舍去小数部分

    def generate_n_numbers(self, qa_i, m):
        n_max = m / qa_i
        if int(n_max) + 1 - n_max <= 0.2:
            limit = int(n_max) + 1
        # elif int(n_max) + 1 - n_max >= 0.9:
        #     limit = int(n_max)
        else:
            if int(n_max) > 0:
                limit = int(n_max)
            else:
                limit = 1
        return np.array([i for i in range(limit, 10, 1) if i / m != 1])

    def generate_even_numbers(self, qa, n, even_number=True):
        # limit = round(qa * n)
        m_max = qa * n
        if int(m_max) + 1 - m_max <= 0.25:
            limit = int(m_max) + 1
        else:
            limit = int(m_max)
        if even_number:
            return np.array([i for i in range(2, limit + 1, 2) if i / n != 1])
        else:
            return np.array([i for i in range(1, limit + 1, 2) if i / n != 1])

    def find_closest_index(self, arr, target):
        """
        找到数组中与目标值最接近的元素的索引

        :param arr: 输入数组
        :param target: 目标值
        :return: 最接近目标值的元素索引
        """
        try:
            # arr = np.array([])  # 示例数组
            if len(arr) == 0:
                list1 = []
                raise ValueError(f"The array is empty. shot_no: {self.shot_no}")
                return list1
            else:
                closest_index = min(range(len(arr)), key=lambda i: abs(arr[i] - target))
                return [closest_index]
                # 如果数组非空，继续处理逻辑
        except ValueError as e:
            print(f"Error: {e} shot_no: {self.shot_no}")
            return []
        # if not arr:
        #
        #     list1 = []
        #     return list1
        # # 使用 min 函数和 key 参数找到最接近的索引
        # else:
        #     closest_index = min(range(len(arr)), key=lambda i: abs(arr[i] - target))
        #     return [closest_index]
    def mode_m_n_identical(self, m, n,n_csd):
        if m == n :
            return m,self.round_number(n_csd)
        else:
            return m,n
    def couple_sure_mode_data(self, m_n_most_data_i, couple_fre_i_data, even_data_i, odd_data_i,
                              qa_i):  # 这种情况下couple_fre_i_data应该与mode_undouble_fre_i直接对应了
        # 3.奇偶+非倍频======>耦合：1,0 [freq1,0,0,-1] [0,freq2,0,-1]
        #      if n_number=n_csd（四舍五入）时：m_n_most_data_i[index,-1]
        #       1.例如m_number==2为偶数，那么n_csd推导奇数：
        #       2.[(2+1)/2,(2+3)/2,(2+5)/2,...]先挖去n_number=1的1/1，再挖去qa条件下的所有m/n如qa<5,5/1挖去；则为参数==>[2.5,3.5,...]
        #       3.if  只有一个[2.5],则无需选最接近的
        #         else：n_csd更接近哪个数组的元素，就认为m_number是几，
        #      if n_number=!n_csd（四舍五入）时：
        #       1.例如n_number=2,n_csd<1.5或>2.5,另一个n可能的值为[1,3,4,5...]=>取[(1+2)/2,(3+2)/2,(4+2)/2,....]中n_csd最接近的值
        #       2.例如n_number=1,n_csd>1.5,另一个n可能的值为[2,3,4,5...]=>取[(2+1)/2,(3+1)/2,(4+1)/2,....]中n_csd最接近的值
        #       3.再分析m_number
        #           3.1.例如m_number==2为偶数，那么n_csd推导奇数：
        #           3.2 [(2+1)/2,(2+3)/2,(2+5)/2,...]先挖去n_number，再挖去qa条件下的所有m/n取最大的n下不满足的，如m=8 即8/2挖去；则为参数==>[1.5,2.5,3.5,...]
        #           3.3 if  只有一个[2.5],则无需选最接近的
        #               else：n_csd更接近哪个数组的元素，就认为m_number是几，
        #       4.确定了如 m 2 、3 ;n 2、1 （2/1 3/2） 则需要分析这一炮完整的过程中有没有出现过确定的 这种有21 或者32的时间《这个新给个processor都可》如果没有就按照m 2<3；n 1<2排序分配
        # 找到对应的频率的even，odd的幅值
        for index in range(len(couple_fre_i_data)):
            if couple_fre_i_data[index] != -1 and couple_fre_i_data[index] != 0:
                even_index_fre = \
                self.find_closest_index([even_data_i[even_index][2] for even_index in range(len(even_data_i))],
                                        couple_fre_i_data[index])[0]
                odd_index_fre = \
                self.find_closest_index([odd_data_i[odd_index][2] for odd_index in range(len(odd_data_i))],
                                        couple_fre_i_data[index])[0]
                even_amp = even_data_i[even_index_fre][1]
                odd_amp = odd_data_i[odd_index_fre][1]
                if self.round_number(m_n_most_data_i[index, 0]) == m_n_most_data_i[index, 8]:
                    if m_n_most_data_i[index, 7] % 2 == 0:  # m_number==2为偶数，那么n_csd推导奇数
                        m_choose_list = self.generate_even_numbers(qa_i, m_n_most_data_i[index, 8], even_number=False)
                        m1_m2_list = [
                            (m_choose_i * odd_amp + m_n_most_data_i[index, 7] * even_amp) / (odd_amp + even_amp) for
                            m_choose_i in m_choose_list]
                        m_closest_index = self.find_closest_index(m1_m2_list, m_n_most_data_i[index, 6])
                        if len(m_closest_index) != 0:  # 偶数mn，奇数mn
                            m_n_most_data_i[index, -4:] = np.array(
                                [m_n_most_data_i[index, 7], m_n_most_data_i[index, 8],
                                 m_choose_list[m_closest_index[0]], m_n_most_data_i[index, 8]])
                        else:
                            m_n_most_data_i[index, -4:] = np.array(
                                [m_n_most_data_i[index, 7], m_n_most_data_i[index, 8],
                                 int(1 + m_choose_list[m_closest_index[0]]), m_n_most_data_i[index, 8]])
                    else:
                        m_choose_list = self.generate_even_numbers(qa_i, m_n_most_data_i[index, 8], even_number=True)
                        m1_m2_list = [
                            (m_choose_i * even_amp + m_n_most_data_i[index, 7] * odd_amp) / (odd_amp + even_amp) for
                            m_choose_i in m_choose_list]
                        m_closest_index = self.find_closest_index(m1_m2_list, m_n_most_data_i[index, 6])
                        if len(m_closest_index) != 0:  # 偶数mn，奇数mn
                            m_n_most_data_i[index, -4:] = np.array(
                                [m_choose_list[m_closest_index[0]], m_n_most_data_i[index, 8],
                                 m_n_most_data_i[index, 7], m_n_most_data_i[index, 8]])
                        else:
                            m_n_most_data_i[index, -4:] = np.array(
                                [int(-1 + m_n_most_data_i[index, 7]), m_n_most_data_i[index, 8],
                                 m_n_most_data_i[index, 7], m_n_most_data_i[index, 8]])
                else:  # n_number=!n_csd（四舍五入）时
                    # 求n是多少
                    n_choose_list = self.generate_n_numbers(qa_i, max(m_n_most_data_i[index, 6] + 1,
                                                                      m_n_most_data_i[index, 7]))
                    n_closest_index = self.find_closest_index(n_choose_list / 2, m_n_most_data_i[index, 0])
                    # if m_n_most_data_i[index, 8] % 2 == 0:
                    #     n_choose_list = self.generate_even_numbers(qa_i, 6, even_number=True)
                    #     # n1_n2_list = [(n_choose_i * odd_amp + m_n_most_data_i[index, 7] * even_amp) / (odd_amp + even_amp) for n_choose_i in n_choose_list]
                    #     closest_index = self.find_closest_index(n_choose_list / 2, m_n_most_data_i[index, 0])
                    # else:
                    #     n_choose_list = self.generate_even_numbers(qa_i, 6, even_number=False)
                    #     closest_index = self.find_closest_index(n_choose_list / 2, m_n_most_data_i[index, 0])
                    # 找到最接近的n
                    if len(n_closest_index) != 0:  # 偶数mn，奇数mn
                        n_others = n_choose_list[n_closest_index[0]]
                    else:
                        n_others = m_n_most_data_i[index, 8]
                    #       3.再分析m_number
                    #           3.1.例如m_number==2为偶数，那么n_csd推导奇数：
                    #           3.2 [(2+1)/2,(2+3)/2,(2+5)/2,...]先挖去n_number，再挖去qa条件下的所有m/n取最大的n下不满足的，如m=8 即8/2挖去；则为参数==>[1.5,2.5,3.5,...]
                    #           3.3 if  只有一个[2.5],则无需选最接近的
                    #               else：n_csd更接近哪个数组的元素，就认为m_number是几，
                    if m_n_most_data_i[index, 7] % 2 == 0:
                        m_choose_list = self.generate_even_numbers(qa_i, max(n_others, m_n_most_data_i[index, 8]),
                                                                   even_number=False)
                        m1_m2_list = [
                            (m_choose_i * odd_amp + m_n_most_data_i[index, 7] * even_amp) / (odd_amp + even_amp) for
                            m_choose_i in m_choose_list]
                        m_closest_index = self.find_closest_index(m1_m2_list, m_n_most_data_i[index, 0])
                        if len(m_closest_index) != 0:  # 偶数mn，奇数mn
                            m_others = m_choose_list[m_closest_index[0]]
                        else:
                            m_others = int(1 + m_n_most_data_i[index, 7])

                        if n_others == m_n_most_data_i[index, 8]:
                            m_n_most_data_i[index, -4:] = np.array(
                                [m_n_most_data_i[index, 7], m_n_most_data_i[index, 8], m_others,
                                 m_n_most_data_i[index, 8]])
                        else:
                            list_m = [m_n_most_data_i[index, 7], m_others]
                            list_n = [m_n_most_data_i[index, 8], n_others]
                            if max(list_m) != m_n_most_data_i[index, 7]:  # max(list_m)放在奇数里面
                                m_n_most_data_i[index, -4:] = np.array(
                                    [min(list_m), min(list_n), max(list_m), max(list_n)])
                            else:
                                m_n_most_data_i[index, -4:] = np.array(
                                    [max(list_m), max(list_n), min(list_m), min(list_n)])
                    else:  # m_number==2为奇数，那么n_csd推导偶数
                        m_choose_list = self.generate_even_numbers(qa_i, max(n_others, m_n_most_data_i[index, 8]),
                                                                   even_number=True)
                        m1_m2_list = [
                            (m_choose_i * even_amp + m_n_most_data_i[index, 7] * odd_amp) / (odd_amp + even_amp) for
                            m_choose_i in m_choose_list]
                        m_closest_index = self.find_closest_index(m1_m2_list, m_n_most_data_i[index, 0])
                        if len(m_closest_index) != 0:  # 偶数mn，奇数mn
                            m_others = m_choose_list[m_closest_index[0]]
                        else:
                            m_others = int(1 + m_n_most_data_i[index, 7])

                        if n_others == m_n_most_data_i[index, 8]:
                            m_n_most_data_i[index, -4:] = np.array(
                                [m_others, m_n_most_data_i[index, 8], m_n_most_data_i[index, 7],
                                 m_n_most_data_i[index, 8]])
                        else:
                            list_m = [m_n_most_data_i[index, 7], m_others]
                            list_n = [m_n_most_data_i[index, 8], n_others]
                            if max(list_m) == m_n_most_data_i[index, 7]:  # max(list_m)放在奇数里面
                                m_n_most_data_i[index, -4:] = np.array(
                                    [min(list_m), min(list_n), max(list_m), max(list_n)])
                            else:
                                m_n_most_data_i[index, -4:] = np.array(
                                    [max(list_m), max(list_n), min(list_m), min(list_n)])
            if m_n_most_data_i[index, -4]!=0 and m_n_most_data_i[index, -3]!=0:
                m_n_most_data_i[index, -4], m_n_most_data_i[index, -3]=self.mode_m_n_identical(m_n_most_data_i[index, -4], m_n_most_data_i[index, -3], m_n_most_data_i[index, 0])
            if m_n_most_data_i[index, -2]!=0 and m_n_most_data_i[index, -1]!=0:
                m_n_most_data_i[index, -2], m_n_most_data_i[index, -1]=self.mode_m_n_identical(m_n_most_data_i[index, -2], m_n_most_data_i[index, -1], m_n_most_data_i[index, 0])

        return m_n_most_data_i
        # def unsure_couple_data(self, m_n_most_data_i, couple_fre_i_data, phases_modified_i, mode_undouble_fre_i):#这种情况下couple_fre_i_data应该与mode_undouble_fre_i直接对应了
        #     uncouple_fre_i=np.zeros(4, dtype=float)
        #     couple_fre_i=deepcopy(couple_fre_i_data)
        #     m_n_most_i=deepcopy(m_n_most_data_i)
        #     for index in range(len(couple_fre_i_data)):
        #         if mode_undouble_fre_i[index]==-1:
        #             m_n_most_i[index,-4:]=np.array([-1,-1,-1,-1])
        #             uncouple_fre_i[index] = -1
        #         # couple_fre_i_data[index]!=0 ,m_n_most_i[index, -4:]本来就为0
        #         elif mode_undouble_fre_i[index]==-1 and mode_undouble_fre_i[index]!=0:
        #             couple=self.csd_phases_couple_judge(m_n_most_data_i[index, 6], index,phases_modified_i,m_n_most_data_i[index, 0])
        #             if couple==1:#耦合
        #                 #写一个函数 判断n、判断m
        #                 #在这个processor之前应该phase也求一个n
        #                 m_n_most_i[index, -4:]=
        #                 # uncouple_fre_i[index] = 0
        #             else:#不耦合
        #                 m_n_most_i[index, -4:] = np.array([m_n_most_data_i[index, 6], self.mode_int(m_n_most_data_i[index, 0]), -1, -1])
        #                 couple_fre_i[index] = 0
        #                 uncouple_fre_i[index] = couple_fre_i_data[index]
        #     return m_n_most_i, couple_fre_i, uncouple_fre_i

    def mode_int(self, mode_number):
        # mode_number求一个四舍五入
        return round(mode_number)

    def uncouple_mode_data(self, m_n_most_data_i, mode_fre_data_i,
                           mode_undouble_fre_i):  # 给定的频率都返回唯一的模式，这个不会影响耦合的频率，可以放在最后用
        m_n_most_i = deepcopy(m_n_most_data_i)
        for index in range(len(mode_fre_data_i)):
            if mode_undouble_fre_i[index] == -1:
                m_n_most_i[index, -4:] = np.array([-1, -1, -1, -1])
            elif mode_undouble_fre_i[index] != -1 and mode_undouble_fre_i[index] != 0:
                if m_n_most_data_i[index, 7] % 2 == 0:
                    m_n_most_i[index, -4:] = np.array([m_n_most_data_i[index, 7],
                                                       self.mode_int(m_n_most_data_i[index, 8]) if m_n_most_data_i[
                                                                                                       index, 7] != 0 else 1,
                                                       -1, -1])
                else:
                    m_n_most_i[index, -4:] = np.array([-1, -1, m_n_most_data_i[index, 7],
                                                       self.mode_int(m_n_most_data_i[index, 8]) if m_n_most_data_i[
                                                                                                       index, 7] != 0 else 1])
            if m_n_most_i[index, -4]!=0 and m_n_most_i[index, -3]!=0:
                m_n_most_i[index, -4], m_n_most_i[index, -3]=self.mode_m_n_identical(m_n_most_i[index, -4], m_n_most_i[index, -3], m_n_most_i[index, 0])
            if m_n_most_data_i[index, -2]!=0 and m_n_most_i[index, -1]!=0:
                m_n_most_i[index, -2], m_n_most_i[index, -1]=self.mode_m_n_identical(m_n_most_i[index, -2], m_n_most_i[index, -1], m_n_most_i[index, 0])
        return m_n_most_i





