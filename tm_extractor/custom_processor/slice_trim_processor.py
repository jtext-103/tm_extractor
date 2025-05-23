#  -*-  coding: utf-8  -*-
import math
from typing import Tuple
from jddb.processor import Signal, Shot, ShotSet, BaseProcessor

class SliceTrimProcessor(BaseProcessor):
    """
            input the point number of the window  and overlap rate of the given window ,
        then the sample rate is recalculated,  return a signal_input of time window sequence
    """
    def __init__(self):
        super().__init__()

    def transform(self, *signal: Signal) -> Tuple[Signal, ...]:
        """Trim all signals to the same length with the shortest one.

        Args:
            *signal: The signals to be trimmed.

        Returns: Tuple[Signal, ...]: The signals trimmed.
        """
        lengths = [len(each_signal.data) for each_signal in signal]
        min_length = min(lengths)
        for each_signal in signal:
            if len(each_signal.data)>min_length:
                each_signal.data = each_signal.data[-min_length:]
                each_signal.attributes["StartTime"]=each_signal.parent.labels["DownTime"]-(len(each_signal.data)-1)/each_signal.attributes["SampleRate"]
        return signal

