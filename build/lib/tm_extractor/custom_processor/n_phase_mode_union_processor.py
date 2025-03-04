import os
from collections import Counter
import math
from typing import Tuple
from copy import deepcopy
import numpy as np
from jddb.processor import *

class NPhaseModeUnionProcessor(BaseProcessor):
    """
        A processor for handling and transforming signals related to phase mode union.
        This class performs operations such as inserting columns, combining signal data,
        and handling asymmetric magnetic probe distributions in the context of tearing mode analysis.

        Args:
            insert_col_index (int, optional): The column index at which to insert new data. Defaults to 7.
            fre_index (int, optional): The index of the frequency data within the signals. Defaults to 3.
        """
    def __init__(self,insert_col_index=7,fre_index=3):
        super().__init__(insert_col_index=insert_col_index,fre_index=fre_index)
        self.col_index = int(insert_col_index)
        if fre_index>insert_col_index:
            self.fre_index=fre_index+1
        else:
            self.fre_index=fre_index
    def transform(self, *signal: Signal) -> Tuple[Signal, ...]:
        #new_B_LFS_n_m_most_th 0
        #new_B_LFS_n_m_sec_th 1
        #new_B_LFS_n_m_third_th 2
        #new_B_LFS_n_m_forth_th 3
        #n_mode_number_signal 4
        """
        Transforms the input signals by inserting new columns and processing the data according to the mode coupling rules.

        The method takes in multiple signals (n_most_signal, n_sec_signal, n_third_signal, n_forth_signal, n_mode_number_signal),
        inserts new columns, and updates the data based on the frequency and mode number values.

        Args:
            *signal: A tuple of Signal objects, each representing a specific tearing mode signal.

        Returns:
            tuple: A tuple of transformed Signal objects (n_most_signal, n_sec_signal, n_third_signal, n_forth_signal).
        """
        # Extract and deepcopy the signals
        n_most_signal = deepcopy(signal.__getitem__(0))
        n_sec_signal = deepcopy(signal.__getitem__(1))
        n_third_signal = deepcopy(signal.__getitem__(2))
        n_forth_signal = deepcopy(signal.__getitem__(3))
        # m_four_signal_list = [deepcopy(signal.__getitem__(i)) for i in range(4, 8)]
        n_mode_number_signal = deepcopy(signal.__getitem__(4))
        new_columns = np.zeros((n_most_signal.data.shape[0], 1), dtype=int)

        n_most_signal.data = self.insert_column(n_most_signal.data, new_columns, self.col_index)
        # n_most_signal.data = np.hstack((n_most_signal.data, new_columns))
        n_sec_signal.data = self.insert_column(n_sec_signal.data, new_columns, self.col_index)
        n_third_signal.data = self.insert_column(n_third_signal.data, new_columns, self.col_index)
        n_forth_signal.data = self.insert_column(n_forth_signal.data, new_columns, self.col_index)
        for i in range(len(n_most_signal.data)):
            n_four_signal_list = [n_most_signal.data[i],n_sec_signal.data[i],n_third_signal.data[i],n_forth_signal.data[i]]
            for j in range(len(n_four_signal_list)):
                if n_four_signal_list[j][self.fre_index] == -1 or n_four_signal_list[j][self.fre_index] == 0:
                    n_four_signal_list[j][-1] = n_four_signal_list[j][self.fre_index]
                else:
                    n_four_signal_list[j][-1] = n_mode_number_signal.data[i][j]
            n_most_signal.data[i] = n_four_signal_list[0]
            n_sec_signal.data[i] = n_four_signal_list[1]
            n_third_signal.data[i] = n_four_signal_list[2]
            n_forth_signal.data[i] = n_four_signal_list[3]
        return n_most_signal,n_sec_signal,n_third_signal,n_forth_signal

    def insert_column(self, arr1, arr2, k):

        """
        Inserts a column from arr2 into arr1 at position k and returns the new array.

        Args:
            arr1 (np.ndarray): The original array with shape (a, b).
            arr2 (np.ndarray): The array to insert, should have shape (a, 1).
            k (int): The column index at which to insert arr2 (0-based index).

        Returns:
            np.ndarray: The new array with the inserted column.

        Raises:
            ValueError: If the number of rows in arr1 and arr2 do not match or if arr2 is not a column vector.
        """
        # Check if arr1 and arr2 have compatible shapes
        if arr1.shape[0] != arr2.shape[0]:
            raise ValueError("The number of rows in arr1 and arr2 must be the same.")
        if arr2.shape[1] != 1:
            raise ValueError("arr2 must be a column vector (shape: a, 1).")

        # Insert the column at the specified index
        if 1<k<len(arr1[0])-1:
            left = arr1[:, :k + 1]
            right = arr1[:, k + 1:]
            new_arr = np.hstack((left, arr2, right))
        elif k==1:
            new_arr = np.hstack((arr2,arr1))
        else:
            new_arr = np.hstack((arr1, arr2))
        return new_arr

