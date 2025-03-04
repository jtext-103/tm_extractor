#  -*-  coding: utf-8  -*-
import math
from typing import Tuple
from jddb.processor import Signal, Shot, ShotSet, BaseProcessor
import numpy as np
from copy import deepcopy
class SliceProcessor(BaseProcessor):
    """
            input the point number of the window  and overlap rate of the given window ,
        then the sample rate is recalculated,  return a signal_input of time window sequence
    """
    def __init__(self, window_length: int, overlap: float):
        super().__init__(window_length=window_length, overlap=overlap)
        assert (0 <= overlap <= 1), "Overlap is not between 0 and 1."
        self.params.update({"WindowLength": window_length,
                            "Overlap": overlap})

    def transform(self, signal: Signal) -> Signal:
        window_length = self.params["WindowLength"]
        overlap = self.params["Overlap"]
        new_signal = deepcopy(signal)
        raw_sample_rate = new_signal.attributes["SampleRate"]
        step = int(window_length * round((1 - overlap), 3))

        down_time = new_signal.time[-1]

        # down_time = round(down_time, 3)

        idx = len(signal.data)
        window = list()
        while (idx - window_length) >= 0:
            window.append(new_signal.data[idx - window_length:idx])
            idx -= step
        window.reverse()
        new_signal.attributes['SampleRate'] = raw_sample_rate / step
        new_signal.data = np.array(window)
        new_signal.attributes['StartTime']  = down_time - (len(window)-1) / new_signal.attributes['SampleRate']
        # new_signal.attributes['StartTime'] = round(new_start_time, 3)
        new_signal.attributes['OriginalSampleRate'] = raw_sample_rate
        return new_signal