# processor_registry.py

from jddb.processor import Signal, Shot, ShotSet, BaseProcessor
from jddb.processor.basic_processors import ClipProcessor, ResamplingProcessor, TrimProcessor
from tm_extractor.custom_processor import LowPassProcessor, InteProcessor, AmpJudgeProcessor, EvenOddProcessor, SliceProcessor, \
    SliceTrimProcessor, M_mode_th, N_mode_csd_phase, ModeFreNoisyCheck, ModeFreCheck, MNModeUnionProcessor, \
    M_mode_csd_phase, NPhaseModeUnionProcessor, FourBFFTProcessor, AmpIsTearingProcessor, ReOrderProcessor, \
    DivertorSafetyFactorProcessor, DoubleJudgeProcessor, OddEvenCoupleJudgeProcessor, \
    CoupleModeProcessor, SmallModeProcessor, IsTMProcessor, SpecificModeExtractProcessor, SplitAmpFreProcessor, \
    PlotModeAmpProcessor

processor_registry = {
    'ClipProcessor': ClipProcessor,
    'ResamplingProcessor': ResamplingProcessor,
    'LowPassProcessor': LowPassProcessor,
    'TrimProcessor': TrimProcessor,
    'InteProcessor': InteProcessor,
    'AmpJudgeProcessor': AmpJudgeProcessor,
    'EvenOddProcessor': EvenOddProcessor,
    'SliceProcessor': SliceProcessor,
    'SliceTrimProcessor': SliceTrimProcessor,
    'M_mode_th': M_mode_th,
    'ModeFreNoisyCheck': ModeFreNoisyCheck,
    'ModeFreCheck': ModeFreCheck,
    'N_mode_csd_phase': N_mode_csd_phase,
    'M_mode_csd_phase': M_mode_csd_phase,
    'MNModeUnionProcessor': MNModeUnionProcessor,
    'NPhaseModeUnionProcessor': NPhaseModeUnionProcessor,
    'FourBFFTProcessor': FourBFFTProcessor,
    'AmpIsTearingProcessor': AmpIsTearingProcessor,
    'ReOrderProcessor': ReOrderProcessor,
    # 'PlotShotProcessor': PlotShotProcessor,
    'DivertorSafetyFactorProcessor': DivertorSafetyFactorProcessor,
    'DoubleJudgeProcessor': DoubleJudgeProcessor,
    'OddEvenCoupleJudgeProcessor': OddEvenCoupleJudgeProcessor,
    'CoupleModeProcessor': CoupleModeProcessor,
    'SmallModeProcessor': SmallModeProcessor,
    'IsTMProcessor': IsTMProcessor,
    'SpecificModeExtractProcessor': SpecificModeExtractProcessor,
    'SplitAmpFreProcessor': SplitAmpFreProcessor,
    'PlotModeAmpProcessor': PlotModeAmpProcessor
}
