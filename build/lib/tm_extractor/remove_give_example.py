from jddb.file_repo import FileRepo
from jddb.processor import ShotSet

from utils import load_config
from jddb.processor.basic_processors import *
# from processors import *
# import ReviseStartTimeProcessor




if __name__ == '__main__':
    example_set = ShotSet(FileRepo('E:\笔记本G盘\mypool_lry_jtext_download_dataset\\2324_CNNiTrans_raw_processing\\test_for_pipline_example\example\$shot_2$00'))

    MINOR_RADIUS_OF_J_TEXT = 0.255
    MAJOR_RADIUS_OF_J_TEXT = 1.05
    config = load_config('tearing_mode_extractor/default_config_json.json')

    #
    ma_pol_tags = config["jtext"]["ma_pol_tags"]
    ma_tor_tags = config["jtext"]["ma_tor_tags"]
    cutoff_freq = int(config["jtext"]["cutoff_freq"])
    low_feild_input_tags = config["jtext"]["low_feild_tags"]#MA_POLB_P06
    high_feild_input_tags = config["jtext"]["high_feild_tags"]#MA_POLA_P06
    # resapmled_250k_output_tags = config["jtext"]["out_put_tags"]#"\\MA_POLB_P06_250k", "\\MA_POLA_P06_250k"
    # low_feild_250k_output_tags = "\\low_feild_tags_250k"
    # high_feild_250k_output_tags = "\\high_feild_tags_250k"
    #
    input_clip_tags=ma_pol_tags+["\\ip","\\bt"]
    output_clip_tags=config["jtext"]["ma_pol_tags"]+["\\ip","\\bt"]
    #
    # input_lowpass_tags=ma_tor_tags+ma_pol_tags
    # output_lowpass_tags=ma_tor_tags+ma_pol_tags
    #
    input_tags_inte_resampled_tags=low_feild_input_tags+high_feild_input_tags
    # output_tags_inte_resampled_tags=low_feild_250k_output_tags+high_feild_250k_output_tags
    #
    # input_trim_250k_tags=input_tags_inte_resampled_tags
    # output_trim_250k_tags=output_tags_inte_resampled_tags
    #
    # input_resample_50k_tags=ma_tor_tags+ma_pol_tags
    # output_resample_50k_tags=ma_tor_tags+ma_pol_tags
    #
    # input_trim_50k_tags=ma_tor_tags+ma_pol_tags
    # output_trim_50k_tags=ma_tor_tags+ma_pol_tags
    #
    # output_inte_tags=["\\B_HFS_Inte", "\\B_LFS_Inte"]
    #
    # time_slice=config["jtext"]["time_slice"]#0.005
    time_slice= 0.005
    ma_pol_tags_sliced = [tag + "_sliced" for tag in ma_pol_tags]
    ma_tor_tags_sliced = [tag + "_sliced" for tag in ma_tor_tags]
    m_csd_real_angle = config["jtext"]["m_csd_real_angle"]#-15
    n_csd_real_angle = config["jtext"]["n_csd_real_angle"]#22.5
    m_csd_tags=[tag + "_sliced" for tag in config["jtext"]["m_csd_tags"]]#["\\MA_POLB_P06", "\\MA_POLD_P01"]
    n_csd_tags=[tag + "_sliced" for tag in config["jtext"]["n_csd_tags"]]#["\\MA_TOR1_P03", "\\MA_TOR1_P04"]
    #
    plt_save_bfft_path=config["jtext"]["plt_save_bfft_path"]#"/mypool/lry/iTransformer/2324_CNNiTrans_raw_processing/plt_bfft"
    #
    # theta_c = np.array([-172.5, -165,-150,-135,-120,-105]) / 180
    # theta_b = np.array([-75, -60, -45, -30, -15, 0]) / 180
    # theta_d = np.array([15, 30, 45, 60, 75, 82.5]) / 180
    # theta_a = np.array([105, 120, 135, 150, 165, 180]) / 180
    # theta2024 = np.concatenate([theta_c, theta_b, theta_d, theta_a])
    theta_ma_pol_tags = config["jtext"]["theta_ma_pol_tags"]
    m_csd_plt_path=config["jtext"]["m_csd_plt_path"]#"/mypool/lry/iTransformer/2324_CNNiTrans_raw_processing/plt_m_csd"
    #
    bigger_threshold=config["jtext"]["bigger_threshold"]#2
    theta_ma_tor_tags = config["jtext"]["theta_ma_tor_tags"]
    n_csd_plt_path=config["jtext"]["n_csd_plt_path"]#"/mypool/lry/iTransformer/2324_CNNiTrans_raw_processing/plt_n_csd"
    mn_list=config["jtext"]["mn"]
    plot_signal_tags = ["\\{0}{1}_amp".format(m, n) for m, n in mn_list]+["\\{0}{1}_fre".format(m, n) for m, n in mn_list]
    plt_mode_amp_path=config["jtext"]["plt_mode_amp_path"]#"/mypool/lry/iTransformer/2324_CNNiTrans_raw_processing/plt_mode_amp"
    plot_raw_tags=config["jtext"]["plot_raw_tags"]
    shot_list=sorted(example_set.shot_list)
    for shot_no in shot_list:
        shot=example_set.get_shot(shot_no)
        shot.remove_signal(tags=["\\ip", "\\bt"]+ma_pol_tags+ma_tor_tags,keep=True)
        if shot_no==1093416:
            shot.labels["DownTime"]=0.3225
        shot.save(FileRepo(
            'E:\笔记本G盘\mypool_lry_jtext_download_dataset\\2324_CNNiTrans_raw_processing\\test_for_pipline_example\example\$shot_2$00'))
