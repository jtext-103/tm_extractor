# -*-  coding:utf-8  -*-
from typing import Tuple
from jddb.processor import Signal, BaseProcessor
from copy import deepcopy

class SplitAmpFreProcessor(BaseProcessor):
    """
        calculate the mean and standard deviation of the given signal
    """

    def __init__(self):
        super().__init__()

    def transform(self, signal: Signal) -> Tuple[Signal, ...]:
        new_signal_amp = deepcopy(signal)
        new_signal_fre = deepcopy(signal)
        new_signal_amp.data = new_signal_amp.data[:,0]
        new_signal_fre.data = new_signal_fre.data[:,1]
        return new_signal_amp, new_signal_fre
