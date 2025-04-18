import json
from typing import List
from jddb.extractor import BaseExtractor
from tm_extractor.custom_processor import (LowPassProcessor, InteProcessor, AmpJudgeProcessor, EvenOddProcessor, SliceProcessor, \
    SliceTrimProcessor, M_mode_th, N_mode_csd_phase, ModeFreNoisyCheck, ModeFreCheck, MNModeUnionProcessor, \
    M_mode_csd_phase, NPhaseModeUnionProcessor, FourBFFTProcessor, AmpIsTearingProcessor, ReOrderProcessor, \
    DivertorSafetyFactorProcessor, DoubleJudgeProcessor, OddEvenCoupleJudgeProcessor, \
    CoupleModeProcessor, SmallModeProcessor, IsTMProcessor, SpecificModeExtractProcessor, SplitAmpFreProcessor, \
    PlotModeAmpProcessor)
from tm_extractor.custom_processor import IsTearingJudgeProcessor
from jddb.processor.basic_processors import ClipProcessor, ResamplingProcessor, TrimProcessor
from jddb.processor import Step, Pipeline



class TMExtractor(BaseExtractor):
    # Constants for config keys
    JTEXT = "jtext"
    MINOR_RADIUS="minor_radius_a"
    MAJOR_RADIUS="major_radius_R"
    PLOT_RAW_TAGS = "plot_raw_tags"
    MA_POL_TAGS = "ma_pol_tags"
    MA_TOR_TAGS = "ma_tor_tags"
    CUTOFF_FREQ = "cutoff_freq"
    LOW_FEILD_TAGS = "low_feild_tags"
    HIGH_FEILD_TAGS = "high_feild_tags"
    IP = "\\ip"
    BT = "\\bt"
    M_CSD_REAL_ANGLE = "m_csd_real_angle"
    N_CSD_REAL_ANGLE = "n_csd_real_angle"
    M_CSD_TAGS = "m_csd_tags"
    N_CSD_TAGS = "n_csd_tags"
    PLT_SAVE_BFFT_PATH = "plt_save_bfft_path"
    M_CSD_PLOT_PATH = "m_csd_plt_path"
    N_CSD_PLOT_PATH = "n_csd_plt_path"
    BIGGER_THRESHOLD = "bigger_threshold"
    THETA_MA_POL_TAGS = "theta_ma_pol_tags"
    THETA_MA_TOR_TAGS = "theta_ma_tor_tags"
    MN_LIST = "mn"
    PLT_MODE_AMP_PATH = "plt_mode_amp_path"
    NS="NS"
    def __init__(self, config_file_path: str,plt_mode_amp_path:str):
        """
        Initializes the TMExtractor with the provided configuration file path.

        Args:
            config_file_path (str): Path to the configuration file.
        """
        # Call the parent class constructor to load the config file
        super().__init__(config_file_path)

        # Store the config file path and load the configuration
        self.config_file_path = config_file_path
        config = self.load_config(self.config_file_path)

        # Initialize various tags and parameters based on the configuration
        self.minor_radius_a=config[self.JTEXT][self.MINOR_RADIUS]
        self.major_radius_R= config[self.JTEXT][self.MAJOR_RADIUS]
        self.ma_pol_tags = config[self.JTEXT][self.MA_POL_TAGS]  # Magnetic axis poloidal tags
        self.ma_tor_tags = config[self.JTEXT][self.MA_TOR_TAGS]  # Magnetic axis toroidal tags
        self.cutoff_freq = int(config[self.JTEXT][self.CUTOFF_FREQ])  # Cutoff frequency for signal processing
        self.low_feild_input_tags = config[self.JTEXT][self.LOW_FEILD_TAGS]  # Tags for low-field input signals
        self.high_feild_input_tags = config[self.JTEXT][self.HIGH_FEILD_TAGS]  # Tags for high-field input signals

        # Define the input and output clip tags for the pipeline
        self.input_clip_tags = self.ma_pol_tags + [self.IP, self.BT]
        self.output_clip_tags = config[self.JTEXT][self.MA_POL_TAGS] + [self.IP, self.BT]

        # Resample tags for low and high field input tags
        self.input_tags_inte_resampled_tags = self.low_feild_input_tags + self.high_feild_input_tags
        # self.input_tags_inte_resampled_tags = [tag + "_raw" for tag in self.input_tags_inte_resampled_tags]
        self.ns=config[self.JTEXT][self.NS]
        # Time slice for processing
        self.time_slice = 0.005

        # Sliced versions of magnetic axis poloidal and toroidal tags
        self.ma_pol_tags_sliced = [tag + "_sliced" for tag in self.ma_pol_tags]
        self.ma_tor_tags_sliced = [tag + "_sliced" for tag in self.ma_tor_tags]

        # CSD (current sheet detection) angles and tags
        self.m_csd_real_angle = config[self.JTEXT][self.M_CSD_REAL_ANGLE]  # -15 degrees
        self.n_csd_real_angle = config[self.JTEXT][self.N_CSD_REAL_ANGLE]  # 22.5 degrees
        self.m_csd_tags = [tag + "_sliced" for tag in config[self.JTEXT][self.M_CSD_TAGS]]
        self.n_csd_tags = [tag + "_sliced" for tag in config[self.JTEXT][self.N_CSD_TAGS]]

        # Paths for saving and plotting data
        self.plt_save_bfft_path = config[self.JTEXT][self.PLT_SAVE_BFFT_PATH]
        self.m_csd_plt_path = config[self.JTEXT][self.M_CSD_PLOT_PATH]
        self.n_csd_plt_path = config[self.JTEXT][self.N_CSD_PLOT_PATH]

        # Threshold for larger signals
        self.bigger_threshold = config[self.JTEXT][self.BIGGER_THRESHOLD]  # Threshold value (e.g., 2)

        # Tags for theta-related magnetic axis measurements
        self.theta_ma_pol_tags = config[self.JTEXT][self.THETA_MA_POL_TAGS]
        self.theta_ma_tor_tags = config[self.JTEXT][self.THETA_MA_TOR_TAGS]

        # Magnetic numbers (m, n) for analysis
        self.mn_list = config[self.JTEXT][self.MN_LIST]

        # Plotting signal tags for mode amplitude and frequency
        self.plot_signal_tags = (
                ["\\{0}{1}_amp".format(m, n) for m, n in self.mn_list]
                + ["\\{0}{1}_fre".format(m, n) for m, n in self.mn_list]
        )

        # Path for saving the mode amplitude plot
        self.plt_mode_amp_path = plt_mode_amp_path

        # Raw signal tags to plot
        self.plot_raw_tags = config[self.JTEXT][self.PLOT_RAW_TAGS]

    def load_config(self):
        with open(self.config_file_path, 'r', encoding='utf-8') as f:  # 使用UTF-8编码
            return json.load(f)
    def extract_steps(self) -> List[Step]:
        """
        Extract steps for the TMExtractor process.

        This method returns a tuple containing multiple instances of the 
        Step class, each representing a different step in the process.

        Returns:
            tuple: A list where each element is an instance of the Step class.
        """
        steps1 = [
            # 1 clip all
            Step(ClipProcessor(start_time=0, end_time_label="DownTime"), input_tags=self.input_clip_tags,
                 output_tags=self.output_clip_tags),
            # 2 lowpass ma_pol and ma_tor
            Step(LowPassProcessor(cutoff_freq=self.cutoff_freq), input_tags=self.ma_tor_tags + self.ma_pol_tags,
                 output_tags=self.ma_tor_tags + self.ma_pol_tags),
            # 3 resample inte
            Step(ResamplingProcessor(250000), input_tags=self.input_tags_inte_resampled_tags,
                 output_tags=["\\low_feild_tags_250k", "\\high_feild_tags_250k"]),
            # 4 clip inte
            Step(ClipProcessor(start_time=0.145, end_time_label="DownTime"),
                 input_tags=["\\low_feild_tags_250k", "\\high_feild_tags_250k"],
                 output_tags=["\\low_feild_tags_250k", "\\high_feild_tags_250k"]),
            # 5 trim inte
            Step(TrimProcessor(), input_tags=[["\\low_feild_tags_250k", "\\high_feild_tags_250k"]],
                 output_tags=[["\\low_feild_tags_250k", "\\high_feild_tags_250k"]]),
            # 6 resample 50k
            Step(ResamplingProcessor(50000), input_tags=self.ma_tor_tags + self.ma_pol_tags,
                 output_tags=self.ma_tor_tags + self.ma_pol_tags),
            # 7 clip 50k
            Step(ClipProcessor(start_time=0.145, end_time_label="DownTime"), input_tags=self.ma_tor_tags + self.ma_pol_tags,
                 output_tags=self.ma_tor_tags + self.ma_pol_tags),
            # 8 trim 50k
            Step(TrimProcessor(), input_tags=[self.ma_tor_tags + self.ma_pol_tags], output_tags=[self.ma_tor_tags + self.ma_pol_tags]),
            # 9 resample 1k
            Step(ResamplingProcessor(1000), input_tags=["\\ip", "\\bt"], output_tags=["\\ip_1k", "\\bt_1k"]),
            # 10 clip 1k
            Step(ClipProcessor(start_time=0.145, end_time_label="DownTime"), input_tags=["\\ip_1k", "\\bt_1k"],
                 output_tags=["\\ip_1k", "\\bt_1k"]),
            # 11 trim 1k
            Step(TrimProcessor(), input_tags=[["\\ip_1k", "\\bt_1k"]], output_tags=[["\\ip_1k", "\\bt_1k"]]),
            # 12 inte
            Step(InteProcessor(NS=self.ns), input_tags=[["\\low_feild_tags_250k", "\\high_feild_tags_250k"]],
                 output_tags=[["\\B_HFS_Inte", "\\B_LFS_Inte"]]),
            # #13 odd even
            Step(EvenOddProcessor(), input_tags=[["\\B_HFS_Inte", "\\B_LFS_Inte"]],
                 output_tags=[["\\B_odd_Inte", "\\B_even_Inte"]]),
            # #14 slice odd even inte
            Step(SliceProcessor(window_length=int(250000 * self.time_slice), overlap=0.8),
                 input_tags=["\\B_odd_Inte", "\\B_even_Inte", "\\B_HFS_Inte", "\\B_LFS_Inte"],
                 output_tags=["\\B_odd_Inte_slice", "\\B_even_Inte_slice", "\\B_HFS_Inte_slice", "\\B_LFS_Inte_slice"]),
            # #15 slice trim odd even
            Step(SliceProcessor(window_length=int(50000 * self.time_slice), overlap=0.8),
                 input_tags=self.ma_tor_tags + self.ma_pol_tags,
                 output_tags=self.ma_pol_tags_sliced + self.ma_tor_tags_sliced),
            # #16 slice trim m_csd
            Step(SliceTrimProcessor(), input_tags=[
                self.ma_pol_tags_sliced + self.ma_tor_tags_sliced + ["\\B_odd_Inte_slice", "\\B_even_Inte_slice",
                                                           "\\B_HFS_Inte_slice", "\\B_LFS_Inte_slice"] + ["\\ip_1k",
                                                                                                          "\\bt_1k"]],
                 output_tags=[self.ma_pol_tags_sliced + self.ma_tor_tags_sliced + ["\\B_odd_Inte_slice", "\\B_even_Inte_slice",
                                                                         "\\B_HFS_Inte_slice", "\\B_LFS_Inte_slice"] + [
                                  "\\ip_1k", "\\bt_1k"]]),
            # 17 m_csd
            Step(M_mode_th(var_th=1e-20, coherence_th=0.95, real_angle=self.m_csd_real_angle, mode_th=0.5),
                 input_tags=[self.m_csd_tags],
                 output_tags=[["\\m_most_max_th", "\\m_sec_max_th", "\\m_third_max_th", "\\m_forth_max_th"]]),
            # 18 m_csd reorder
            Step(M_mode_th(var_th=1e-14, coherence_th=0.95, real_angle=self.n_csd_real_angle, mode_th=0.5),
                 input_tags=[self.n_csd_tags],
                 output_tags=[["\\n_most_max_th", "\\n_sec_max_th", "\\n_third_max_th", "\\n_forth_max_th"]]),
            # 19 m_csd reorder
            Step(ModeFreNoisyCheck(),
                 input_tags=[
                     ["\\n_most_max_th", "\\n_sec_max_th", "\\n_third_max_th", "\\n_forth_max_th", "\\m_most_max_th",
                      "\\m_sec_max_th", "\\m_third_max_th", "\\m_forth_max_th"]],
                 output_tags=[["\\n_most_max_th", "\\n_sec_max_th", "\\n_third_max_th", "\\n_forth_max_th"]]),
            # 20 Judge whether the radial extraction frequency is correct based on the poloidal extracted frequency
            Step(ModeFreCheck(n_real_angle=self.n_csd_real_angle,m_real_angle=self.m_csd_real_angle, down_number=1, coherence_th=0.95), input_tags=[
                ["\\n_most_max_th", "\\n_sec_max_th", "\\n_third_max_th", "\\n_forth_max_th", "\\m_most_max_th",
                 "\\m_sec_max_th", "\\m_third_max_th", "\\m_forth_max_th"] + self.n_csd_tags+self.m_csd_tags],
                 output_tags=[["\\n_most_max_th", "\\n_sec_max_th", "\\n_third_max_th", "\\n_forth_max_th"]]),
            # 21 reorder Reorder the magnetic probes based on FFT
            Step(FourBFFTProcessor(singal_rate=250000), input_tags=[
                ["\\B_LFS_Inte_slice", "\\n_most_max_th", "\\n_sec_max_th", "\\n_third_max_th", "\\n_forth_max_th"]],
                 output_tags=[["\\B_LFS_n_most_th", "\\B_LFS_n_sec_th", "\\B_LFS_n_third_th", "\\B_LFS_n_forth_th",
                               "\\B_LFS_n_amp_inte_range"]]),
            # 22 Determine whether the magnetic probe indicates tearing mode
            Step(AmpIsTearingProcessor(amp_threshold_ratio=0.5, f_upper_threshold=1500),
                 input_tags=["\\B_LFS_n_most_th", "\\B_LFS_n_sec_th", "\\B_LFS_n_third_th", "\\B_LFS_n_forth_th"],
                 output_tags=[["\\new_B_LFS_n_most_th", "\\new_B_LFS_n_most_th_is_tearing"],
                              ["\\new_B_LFS_n_sec_th", "\\new_B_LFS_n_sec_th_is_tearing"],
                              ["\\new_B_LFS_n_third_th", "\\new_B_LFS_n_third_th_is_tearing"],
                              ["\\new_B_LFS_n_forth_th", "\\new_B_LFS_n_forth_th_is_tearing"]]),
            # 23 Reorder
            Step(ReOrderProcessor(), input_tags=[
                ["\\new_B_LFS_n_most_th", "\\new_B_LFS_n_sec_th", "\\new_B_LFS_n_third_th", "\\new_B_LFS_n_forth_th",
                 "\\new_B_LFS_n_most_th_is_tearing", "\\new_B_LFS_n_sec_th_is_tearing",
                 "\\new_B_LFS_n_third_th_is_tearing", "\\new_B_LFS_n_forth_th_is_tearing"]],
                 output_tags=[["\\new_B_LFS_n_most_th", "\\new_B_LFS_n_sec_th", "\\new_B_LFS_n_third_th",
                               "\\new_B_LFS_n_forth_th",
                               "\\new_B_LFS_n_most_th_is_tearing", "\\new_B_LFS_n_sec_th_is_tearing",
                               "\\new_B_LFS_n_third_th_is_tearing", "\\new_B_LFS_n_forth_th_is_tearing"]]),
            # 24 Reorder the odd mode based on FFT
            Step(FourBFFTProcessor(singal_rate=250000), input_tags=[
                ["\\B_even_Inte_slice", "\\n_most_max_th", "\\n_sec_max_th", "\\n_third_max_th", "\\n_forth_max_th"]],
                 output_tags=[["\\B_even_n_most_th", "\\B_even_n_sec_th", "\\B_even_n_third_th", "\\B_even_n_forth_th",
                               "\\B_even_n_amp_inte_range"]]),
            # 25 Determine whether there is tearing for the odd mode
            Step(AmpIsTearingProcessor(amp_threshold_ratio=0.5, f_upper_threshold=1500),
                 input_tags=["\\B_even_n_most_th", "\\B_even_n_sec_th", "\\B_even_n_third_th", "\\B_even_n_forth_th"],
                 output_tags=[["\\new_B_even_n_most_th", "\\new_B_even_n_most_th_is_tearing"],
                              ["\\new_B_even_n_sec_th", "\\new_B_even_n_sec_th_is_tearing"],
                              ["\\new_B_even_n_third_th", "\\new_B_even_n_third_th_is_tearing"],
                              ["\\new_B_even_n_forth_th", "\\new_B_even_n_forth_th_is_tearing"]]),
            # 26 Rearrange the odd mode.
            Step(ReOrderProcessor(), input_tags=[
                ["\\new_B_even_n_most_th", "\\new_B_even_n_sec_th", "\\new_B_even_n_third_th",
                 "\\new_B_even_n_forth_th",
                 "\\new_B_even_n_most_th_is_tearing", "\\new_B_even_n_sec_th_is_tearing",
                 "\\new_B_even_n_third_th_is_tearing", "\\new_B_even_n_forth_th_is_tearing"]],
                 output_tags=[["\\new_B_even_n_most_th", "\\new_B_even_n_sec_th", "\\new_B_even_n_third_th",
                               "\\new_B_even_n_forth_th",
                               "\\new_B_even_n_most_th_is_tearing", "\\new_B_even_n_sec_th_is_tearing",
                               "\\new_B_even_n_third_th_is_tearing", "\\new_B_even_n_forth_th_is_tearing"]]),
            # 27 Rearrange the even mode based on FFT
            Step(FourBFFTProcessor(singal_rate=250000), input_tags=[
                ["\\B_odd_Inte_slice", "\\n_most_max_th", "\\n_sec_max_th", "\\n_third_max_th", "\\n_forth_max_th"]],
                 output_tags=[["\\B_odd_n_most_th", "\\B_odd_n_sec_th", "\\B_odd_n_third_th", "\\B_odd_n_forth_th",
                               "\\B_odd_n_amp_inte_range"]]),
            # 28 偶数模判断是否撕裂
            Step(AmpIsTearingProcessor(amp_threshold_ratio=0.5, f_upper_threshold=1500),
                 input_tags=["\\B_odd_n_most_th", "\\B_odd_n_sec_th", "\\B_odd_n_third_th", "\\B_odd_n_forth_th"],
                 output_tags=[["\\new_B_odd_n_most_th", "\\new_B_odd_n_most_th_is_tearing"],
                              ["\\new_B_odd_n_sec_th", "\\new_B_odd_n_sec_th_is_tearing"],
                              ["\\new_B_odd_n_third_th", "\\new_B_odd_n_third_th_is_tearing"],
                              ["\\new_B_odd_n_forth_th", "\\new_B_odd_n_forth_th_is_tearing"]]),
            # 29 Determine whether the even mode is tearing
            Step(ReOrderProcessor(), input_tags=[
                ["\\new_B_odd_n_most_th", "\\new_B_odd_n_sec_th", "\\new_B_odd_n_third_th", "\\new_B_odd_n_forth_th",
                 "\\new_B_odd_n_most_th_is_tearing", "\\new_B_odd_n_sec_th_is_tearing",
                 "\\new_B_odd_n_third_th_is_tearing", "\\new_B_odd_n_forth_th_is_tearing"]],
                 output_tags=[["\\new_B_odd_n_most_th", "\\new_B_odd_n_sec_th", "\\new_B_odd_n_third_th",
                               "\\new_B_odd_n_forth_th",
                               "\\new_B_odd_n_most_th_is_tearing", "\\new_B_odd_n_sec_th_is_tearing",
                               "\\new_B_odd_n_third_th_is_tearing", "\\new_B_odd_n_forth_th_is_tearing"]]),
            # 30 plt bfft
            Step(DivertorSafetyFactorProcessor(minor_radius_a=self.minor_radius_a, major_radius_R=self.major_radius_R),
                 input_tags=[["\\ip_1k", "\\bt_1k"]], output_tags=["\\qa_1k"]),
            # 31 plt bfft
            # Step(PlotShotProcessor(plt_save=self.plt_save_bfft_path, minor_radius_a=0.22, major_radius_a=1.05),
            #      input_tags=[
            #          self.plot_raw_tags + ["\\B_LFS_Inte", "\\ip_1k", "\\bt_1k", "\\new_B_LFS_n_most_th", "\\qa_1k"]],
            #      output_tags=["\\qa_1k"]),
            # 31 m_csd
            Step(processor=M_mode_csd_phase(theta=self.theta_ma_pol_tags, plt_path=self.m_csd_plt_path, plot_phase=False),
                 input_tags=[["\\new_B_LFS_n_most_th", "\\new_B_LFS_n_sec_th", "\\new_B_LFS_n_third_th",
                              "\\new_B_LFS_n_forth_th"] + self.ma_pol_tags_sliced],
                 output_tags=[["\\new_B_LFS_n_most_mode_number", "\\phases_modified", "\\min_zero_max_index"]]),
            # 32 m_csd union
            Step(processor=MNModeUnionProcessor(pol_samperate=50000, real_angle=self.m_csd_real_angle), input_tags=[
                ["\\new_B_LFS_n_most_th", "\\new_B_LFS_n_sec_th", "\\new_B_LFS_n_third_th", "\\new_B_LFS_n_forth_th",
                 "\\m_most_max_th", "\\m_sec_max_th", "\\m_third_max_th", "\\m_forth_max_th",
                 "\\new_B_LFS_n_most_mode_number"] + self.m_csd_tags],
                 output_tags=[["\\new_B_LFS_n_m_most_th", "\\new_B_LFS_n_m_sec_th", "\\new_B_LFS_n_m_third_th",
                               "\\new_B_LFS_n_m_forth_th"]]),
            # 33 m_csd reorder
            Step(AmpJudgeProcessor(threshold=2),input_tags=[["\\new_B_LFS_n_m_most_th", "\\new_B_LFS_n_m_sec_th", "\\new_B_LFS_n_m_third_th", "\\new_B_LFS_n_m_forth_th",
                                                         "\\new_B_even_n_most_th","\\new_B_even_n_sec_th", "\\new_B_even_n_third_th","\\new_B_even_n_forth_th",
                                                         "\\new_B_odd_n_most_th", "\\new_B_odd_n_sec_th","\\new_B_odd_n_third_th", "\\new_B_odd_n_forth_th"
                                                         ]],
            output_tags=[["\\mode_fre","\\new_B_LFS_n_m_most_judge", "\\new_B_LFS_n_m_sec_judge", "\\new_B_LFS_n_m_third_judge", "\\new_B_LFS_n_m_forth_judge"]],
        ),
            # 34 double
            Step(DoubleJudgeProcessor(), input_tags=["\\mode_fre"], output_tags=["\\mode_undouble_fre"]),
            # 35 odd even couple
            Step(OddEvenCoupleJudgeProcessor(threshold=2), input_tags=[
                ["\\mode_fre", "\\mode_undouble_fre", "\\new_B_even_n_most_th", "\\new_B_even_n_sec_th",
                 "\\new_B_even_n_third_th", "\\new_B_even_n_forth_th",
                 "\\new_B_odd_n_most_th", "\\new_B_odd_n_sec_th", "\\new_B_odd_n_third_th", "\\new_B_odd_n_forth_th"]],
                 output_tags=[["\\couple_state", "\\couple_fre", "\\uncouple_fre"]]),
            # 36 n_csd
            Step(N_mode_csd_phase(theta=self.theta_ma_tor_tags, plt_path=self.n_csd_plt_path, plot_phase=False),
                 input_tags=[["\\new_B_LFS_n_m_most_judge", "\\new_B_LFS_n_m_sec_judge", "\\new_B_LFS_n_m_third_judge",
                              "\\new_B_LFS_n_m_forth_judge"] + self.ma_tor_tags_sliced],
                 output_tags=[
                     ["\\new_B_LFS_n_tor_most_mode_number", "\\n_tor_phases_modified", "\\n_tor_min_zero_max_index"]]),
            # 37 n_csd union
            Step(NPhaseModeUnionProcessor(insert_col_index=-1, fre_index=2),
                 input_tags=[["\\new_B_LFS_n_m_most_judge", "\\new_B_LFS_n_m_sec_judge", "\\new_B_LFS_n_m_third_judge",
                              "\\new_B_LFS_n_m_forth_judge", "\\new_B_LFS_n_tor_most_mode_number"]],
                 output_tags=[
                     ["\\new_B_LFS_n2_m_most_judge", "\\new_B_LFS_n2_m_sec_judge", "\\new_B_LFS_n2_m_third_judge",
                      "\\new_B_LFS_n2_m_forth_judge"]]),
            # 38 couple mode
            Step(CoupleModeProcessor(),
                 input_tags=[["\\mode_fre","\\couple_state", "\\couple_fre", "\\uncouple_fre","\\phases_modified",
             "\\new_B_LFS_n2_m_most_judge","\\new_B_LFS_n2_m_sec_judge","\\new_B_LFS_n2_m_third_judge","\\new_B_LFS_n2_m_forth_judge",
             "\\new_B_even_n_most_th", "\\new_B_even_n_sec_th", "\\new_B_even_n_third_th", "\\new_B_even_n_forth_th",
             "\\new_B_odd_n_most_th", "\\new_B_odd_n_sec_th", "\\new_B_odd_n_third_th", "\\new_B_odd_n_forth_th","\\qa_1k"]],
                output_tags=[["\\new_B_LFS_nm_most_judge","\\new_B_LFS_nm_sec_judge","\\new_B_LFS_nm_third_judge","\\new_B_LFS_nm_forth_judge",
             "\\couple_nm_fre", "\\uncouple_nm_fre"]]),
            # 39 Determine the mode number based on Qa.
            Step(SmallModeProcessor(threshold_down=0.5,threshold_upp=self.bigger_threshold),input_tags=[["\\new_B_LFS_nm_most_judge", "\\new_B_LFS_nm_sec_judge",
            "\\new_B_LFS_nm_third_judge","\\new_B_LFS_nm_forth_judge","\\qa_1k"]],
             output_tags=[["\\new_B_th_nm_most_judge", "\\new_B_th_nm_sec_judge", "\\new_B_th_nm_third_judge","\\new_B_th_nm_forth_judge"]]),
            # 40 Determine whether the mode is tearing based on the amplitude of the mode
            Step(IsTMProcessor(), input_tags=[
                ["\\new_B_LFS_nm_most_judge", "\\new_B_LFS_nm_sec_judge", "\\new_B_LFS_nm_third_judge",
                 "\\new_B_LFS_nm_forth_judge"]],
                 output_tags=["\\new_B_th_is_tearing"]),
        ]

        steps2 = []
        for m, n in self.mn_list:
            steps2.extend([
                # 41 Extract the specific mode
                Step(SpecificModeExtractProcessor(m=m, n=n, bigger_threshold=self.bigger_threshold),
                     input_tags=[["\\new_B_th_nm_most_judge", "\\new_B_th_nm_sec_judge", "\\new_B_th_nm_third_judge",
                                  "\\new_B_th_nm_forth_judge", "\\new_B_even_n_most_th", "\\new_B_even_n_sec_th",
                                  "\\new_B_even_n_third_th", "\\new_B_even_n_forth_th",
                                  "\\new_B_odd_n_most_th", "\\new_B_odd_n_sec_th", "\\new_B_odd_n_third_th",
                                  "\\new_B_odd_n_forth_th"]],
                     output_tags=[["\\{0}{1}_amp_fre".format(m, n), "\\{0}{1}_is_couple".format(m, n),
                                   "\\{0}{1}_is_exist".format(m, n),
                                   "\\{0}{1}_bigger_than_{2}_Gs".format(m, n, self.bigger_threshold)]]),
                # 42 Split the amplitude and frequency of the mode
                Step(SplitAmpFreProcessor(),
                     input_tags=["\\{0}{1}_amp_fre".format(m, n)],
                     output_tags=[["\\{0}{1}_amp".format(m, n), "\\{0}{1}_fre".format(m, n)]])
            ])

        steps3 = [
            Step(PlotModeAmpProcessor(mode_plot_save_file=self.plt_mode_amp_path, mn_list=self.mn_list), input_tags=[
                self.plot_raw_tags + ["\\B_LFS_Inte", "\\new_B_even_n_most_th", "\\new_B_odd_n_most_th",
                                 "\\new_B_th_nm_most_judge",
                                 "\\qa_1k"] + self.plot_signal_tags],
                 output_tags=["\\qa_1k"])]
        steps4 = [
            Step(IsTearingJudgeProcessor(),
                 input_tags=[["\\new_B_th_nm_most_judge", "\\new_B_th_nm_sec_judge", "\\new_B_th_nm_third_judge",
                              "\\new_B_th_nm_forth_judge"]],
                 output_tags=["\\is_tearing"])]

        steps = steps1 + steps2 + steps3 + steps4

        return steps
    def make_pipeline(self):
        return Pipeline(self.extract_steps())