import torch
import soundfile as sf
import pickle
import numpy as np
from scipy.io import wavfile

from BIGDATA_SNR_PL_LSTM_DATASET.gen_feat import BatchDataLoader, SpeechMixDataset, FeatureDataCreator
from BIGDATA_SNR_PL_LSTM_DATASET.stft_istft import STFT

from BIGDATA_SNR_PL_LSTM_DATASET.config import *
from BIGDATA_SNR_PL_LSTM_DATASET.util import _calc_alpha, gen_list, _calc_stft, create_folder, _calc_irm, read_audio, \
    write_audio, _cal_log

device = torch.device("cuda:" + str(CUDA_ID[0]) if torch.cuda.is_available() else "cpu")

STFT = STFT(WIN_LEN, WIN_OFFSET).cuda(CUDA_ID[0])

def cut_data(data_type):
    # workspace = config_s.workspace                        # 工作空间
    data_dir = DATA_PATH                          # premix_data数据目录
    fs = SAMPLE_RATE
    tra_speech_filename = os.path.join(workspace,'speech.txt')             # 训练数据（1800条）
    val_speech_filename = os.path.join(workspace, 'val.txt')               # 验证数据（200条）
    core_test192_filename = os.path.join(workspace, 'core_test192.txt')    # 测试数据（192条）
    all_filename = os.path.join(workspace, 'all.txt')                      # 所有数据（2192条）
    noise_dir = os.path.join(data_dir, 'noise')           # 噪声（noise）目录

    noise_name = [na for na in os.listdir(noise_dir) if na.lower().endswith(".wav")]
    print(f'noise_name:{noise_name}')
    if data_type == 'train':
        i = 0
        # 创建训练数据文件夹，存放训练数据
        with open(tra_speech_filename, 'r') as tra_file_to_read:
            while True:                                # 不能用while True,因为读到最后一行时，为空，此时出错，会影响后续运行
            # if i == 899:
              #     print(f'i={i}')
            # while tra_file_to_read.readline():           # 只要读取的行不为空，则继续
            # for i in range(1, 1800):
                # 读取训练数据

                i += 1
                if i == 1799:
                    print(f'i={i}')
                lines = tra_file_to_read.readline()  # 整行读取数据，也即获取语音数据对应的文件名
                # print(lines)
                speech_na = lines.split()[0]  # 去除“\n”
                speech_na_path = os.path.join(data_dir, "speech", "%s.wav" % speech_na)     # 语音所在路径
                speech = read_audio(speech_na_path)                                           # 读取语音数据
                tr_speech_path = os.path.join(data_dir, "train_data", "%s.wav" % speech_na)         # 定义存放训练数据的文件夹
                create_folder(os.path.dirname(tr_speech_path))                                # 创建存放训练数据的文件夹
                write_audio(tr_speech_path, speech, fs)                                       # 写入语音数据
                if i == 1800:     # 限制读取完最后一行结束，不加的话会出错
                    break
        tra_file_to_read.close()                                                              # 关闭文件
        print(f'i={i}')
    elif data_type == 'valid':
        # 创建验证数据文件夹，存放验证数据
        i = 0
        with open(val_speech_filename, 'r') as val_file_to_read:
            while True:
            # while val_file_to_read.readline():
            # for i in range(1, 200):
                # 读取验证数据
                i += 1
                lines = val_file_to_read.readline()  # 整行读取数据，也即获取语音数据对应的文件名
                # print(lines)
                speech_na = lines.split()[0]  # 去除“\n”
                speech_na_path = os.path.join(data_dir, "speech", "%s.wav" % speech_na)
                speech = read_audio(speech_na_path)
                tr_speech_path = os.path.join(data_dir, "valid_data", "%s.wav" % speech_na)
                create_folder(os.path.dirname(tr_speech_path))
                write_audio(tr_speech_path, speech, fs)
                if i == 200:                    # 限制读取完最后一行结束，不加的话会出错
                    break
        val_file_to_read.close()
        print(f'i={i}')
    elif data_type == 'test':
        # 创建测试数据文件夹，存放测试数据
        i = 0
        with open(core_test192_filename, 'r') as tes_file_to_read:
            while True:
            # while tes_file_to_read.readline():
            # for i in range(1, 192):
                i += 1
                line = tes_file_to_read.readline()
                tmp = os.path.splitext(line)[0]                     # 示例：tmp：F:\BaiduNetdiskDownload\TIMIT-wav\TEST\DR1\MDAB0\SI1039
                speech_na = tmp.split('\\')[4] + '_' + tmp.split('\\')[5] + '_' + tmp.split("\\")[6]                           # 示例：tmp_：SI1039
                with open(all_filename, 'r') as all_file_to_read:
                    # for i in range(1, 2192):
                    j = 0
                    while True:
                    # while all_file_to_read.readline():
                        j += 1
                        all_line = all_file_to_read.readline()
                        all_tmp = all_line.split()[0]  # 测试数据的文件名称      # 示例：DR1_FMEM0_SX387
                        # all_tmp_ = all_tmp.split("_")[2]                        # 示例：SX387
                        if all_tmp == speech_na:
                            # print(all_tmp)
                            test_na = os.path.join(data_dir, "speech", "%s.wav" % all_tmp)
                            speech = read_audio(test_na)
                            tes_speech_path = os.path.join(data_dir, "test_data", "%s.wav" % all_tmp)
                            create_folder(os.path.dirname(tes_speech_path))
                            write_audio(tes_speech_path, speech, fs)
                        if j == 2192:      # 限制读取完最后一行结束，不加的话会出错
                            break
                if i == 192:               # 限制读取完最后一行结束，不加的话会出错
                    break
                all_file_to_read.close()

        tes_file_to_read.close()
        print(f'i={i}')
    else:
        print("Data_type Error!")
    pass

def cutNoise():
    # /data01/limiao/little_lstm_data
    data_dir = DATA_PATH               # 数据目录
    # /data01/limiao/lstm_data/noise
    noise_dir = os.path.join(data_dir, 'noise')      # 噪声所在目录
    noise_files = os.listdir(noise_dir)              # 噪声目录下的文件名称列表
    for file in noise_files:                         # 遍历噪声列表文件，对噪声进行分割
        sample_rate, y_data = wavfile.read(os.path.join(noise_dir, file))

        third = y_data.shape[0] / 3
        data_train = y_data[:int(third)]                 # 前1/3部分作为训练噪声
        data_valid = y_data[int(third):int(2*third)]     # 1/3到2/3部分作为验证噪声
        data_test = y_data[int(2*third):]                # 后1/3部分作为测试噪声

        tra_noise_dir = os.path.join(data_dir, 'noise_train')
        tes_noise_dir = os.path.join(data_dir, 'noise_test')
        val_noise_dir = os.path.join(data_dir, 'noise_valid')
        create_folder(tra_noise_dir)                          # 创建保存训练噪声的目录
        create_folder(tes_noise_dir)                          # 创建保存测试噪声的目录
        create_folder(val_noise_dir)                          # 创建保存验证噪声的目录

        tra_noise = os.path.join(tra_noise_dir, '%s_%s' % ('train', file))
        tes_noise = os.path.join(tes_noise_dir, '%s_%s' % ('test', file))
        val_noise = os.path.join(val_noise_dir, '%s_%s' % ('valid', file))
        if file != 'factory2.wav' and file != 'op.wav':
            sf.write(tra_noise, data_train, sample_rate)  # 写入训练噪声

        sf.write(tes_noise, data_test, sample_rate)           # 写入测试噪声
        sf.write(val_noise,data_valid,sample_rate)            # 写入验证噪声

    print('Cut noise success!')
    pass

def getdata(data_type, snr):
    clean_list = gen_list(CLEAN_TR_PATH, '.wav')
    noise_list = gen_list(NOISE_TR_PATH, '.wav')

    for i in range(len(noise_list)):
        noise_path = NOISE_TR_PATH + '/' + noise_list[i]
        noise_audio, _ = sf.read(noise_path)

        for j in range(len(clean_list)):
            clean_path = CLEAN_TR_PATH + '/' + clean_list[j]
            noise_audio, _ = sf.read(noise_path)
            clean_audio, _ = sf.read(clean_path)

            tmp_spe = clean_list[j].split(".")[0]
            tmp_noi = noise_list[i].split(".")[0]

            n_sample = len(clean_audio)
            nframe = (n_sample - WIN_LEN) // WIN_OFFSET + 1  # WIN_LEN = 320    WIN_OFFSET = 160
            n_sample = (nframe + 1) * WIN_OFFSET
            # 获取相同长度的语音和噪声
            clean_audio_re = clean_audio[0: n_sample]
            noise_audio_re = noise_audio[0: n_sample]
            # data = np.stack((clip, clean, noise), 1)  # 横向拼接

            # 加噪
            mix_train_wav_path = os.path.join(DATA_PATH, 'mix_train_wav', 'train_{}dB.{}.{}.wav'.format(snr, tmp_spe, tmp_noi))
            mix_train_wav = clean_audio_re + noise_audio_re * _calc_alpha(SNR=snr, speech=clean_audio_re, noise=noise_audio_re)
            noise_wav = noise_audio_re * _calc_alpha(SNR=snr, speech=clean_audio_re, noise=noise_audio_re)
            create_folder(os.path.dirname(mix_train_wav_path))
            sf.write(file=mix_train_wav_path, data=mix_train_wav, samplerate=SAMPLE_RATE)

            # 计算STFT以及IRM mask
            mix_train_wav_spec_irm_path = os.path.join(DATA_PATH, 'mix_train_spec_mask', 'train_{}dB_mask.{}.{}.p'.format(snr, tmp_spe, tmp_noi))
            create_folder(os.path.dirname(mix_train_wav_spec_irm_path))
            mix_train_wav_spec = _calc_stft(mix_train_wav)
            noise_wav_spec = _calc_stft(noise_wav)
            mix_train_wav_irm = _calc_irm(speech=mix_train_wav_spec, noise=noise_wav_spec)
            data1 = [mix_train_wav_spec, mix_train_wav_irm]
            pickle.dump(data1, open(mix_train_wav_spec_irm_path, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

            mix_train_5dB_wav_path = os.path.join(DATA_PATH, 'mix_train_5dB_wav',
                                                  'train_5dB.{}.{}.wav'.format(tmp_spe, tmp_noi))
            mix_train_dBs_mask_path = os.path.join(DATA_PATH, 'mix_train_dBs_mask',
                                                   'train_mask.{}.{}.p'.format(tmp_spe, tmp_noi))

            if not os.path.exists(mix_train_dBs_mask_path):
                mix_train_5dB = clean_audio_re + noise_audio_re * _calc_alpha(SNR=5, speech=clean_audio_re, noise=noise_audio_re)
                noise_train_5dB = noise_audio_re * _calc_alpha(SNR=5, speech=clean_audio_re, noise=noise_audio_re)
                create_folder(os.path.dirname(mix_train_5dB_wav_path))
                sf.write(file=mix_train_5dB_wav_path, data=mix_train_5dB, samplerate=SAMPLE_RATE)

                # 计算STFT特征以及IRM mask
                mix_train_5dB_spec = _calc_stft(mix_train_5dB)
                noise_train_5dB_spec = _calc_stft(noise_train_5dB)
                mix_train_5dB_irm = _calc_irm(speech=mix_train_5dB_spec, noise=noise_train_5dB_spec)


                mix_train_10dB_wav_path = os.path.join(DATA_PATH, 'mix_train_10dB_wav',
                                                       'train_10dB.{}.{}.wav'.format(tmp_spe, tmp_noi))
                mix_train_10dB = clean_audio_re + noise_audio_re * _calc_alpha(SNR=10, speech=clean_audio_re, noise=noise_audio_re)
                noise_train_10dB = noise_audio_re * _calc_alpha(SNR=10, speech=clean_audio_re, noise=noise_audio_re)
                create_folder(os.path.dirname(mix_train_10dB_wav_path))
                sf.write(file=mix_train_10dB_wav_path, data=mix_train_10dB, samplerate=SAMPLE_RATE)

                # 计算STFT特征以及IRM mask
                mix_train_10dB_spec = _calc_stft(mix_train_10dB)
                noise_train_10dB_spec = _calc_stft(noise_train_10dB)
                mix_train_10dB_irm = _calc_irm(speech=mix_train_10dB_spec, noise=noise_train_10dB_spec)

                mix_train_15dB_wav_path = os.path.join(DATA_PATH, 'mix_train_15dB_wav',
                                                       'train_15dB.{}.{}.wav'.format(tmp_spe, tmp_noi))
                mix_train_15dB = clean_audio_re + noise_audio_re * _calc_alpha(SNR=15, speech=clean_audio_re, noise=noise_audio_re)
                noise_train_15dB = noise_audio_re * _calc_alpha(SNR=15, speech=clean_audio_re, noise=noise_audio_re)
                create_folder(os.path.dirname(mix_train_15dB_wav_path))
                sf.write(file=mix_train_15dB_wav_path, data=mix_train_15dB, samplerate=SAMPLE_RATE)

                # 计算STFT特征以及IRM mask
                mix_train_15dB_spec = _calc_stft(mix_train_15dB)
                noise_train_15dB_spec = _calc_stft(noise_train_15dB)
                mix_train_15dB_irm = _calc_irm(speech=mix_train_15dB_spec, noise=noise_train_15dB_spec)

                mix_train_20dB_wav_path = os.path.join(DATA_PATH, 'mix_train_20dB_wav',
                                                       'train_20dB.{}.{}.wav'.format(tmp_spe, tmp_noi))
                mix_train_20dB = clean_audio_re + noise_audio_re * _calc_alpha(SNR=20, speech=clean_audio_re, noise=noise_audio_re)
                noise_train_20dB = noise_audio_re * _calc_alpha(SNR=20, speech=clean_audio_re, noise=noise_audio_re)
                create_folder(os.path.dirname(mix_train_20dB_wav_path))
                sf.write(file=mix_train_20dB_wav_path, data=mix_train_20dB, samplerate=SAMPLE_RATE)

                # 计算STFT特征以及IRM mask
                mix_train_20dB_spec = _calc_stft(mix_train_20dB)
                noise_train_20dB_spec = _calc_stft(noise_train_20dB)
                mix_train_20dB_irm = _calc_irm(speech=mix_train_20dB_spec, noise=noise_train_20dB_spec)


                # 保存STFT以及IRM
                data2 = [mix_train_5dB_spec, mix_train_10dB_spec, mix_train_15dB_spec, mix_train_20dB_spec, mix_train_5dB_irm, mix_train_10dB_irm, mix_train_15dB_irm, mix_train_20dB_irm]
                mix_train_dBs_mask_path = os.path.join(DATA_PATH, 'mix_train_dBs_mask', 'train_mask.{}.{}.p'.format(tmp_spe, tmp_noi))
                create_folder(os.path.dirname(mix_train_dBs_mask_path))
                pickle.dump(data2, open(mix_train_dBs_mask_path, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    pass

def gen_mixdata(data_type, snr):
    CLEAN_PATH = DATA_PATH + '/{}_data/'.format(data_type)
    NOISE_PATH = DATA_PATH + '/noise_{}'.format(data_type)

    clean_list = gen_list(CLEAN_PATH, '.wav')
    noise_list = gen_list(NOISE_PATH, '.wav')

    for i in range(len(noise_list)):
        noise_path = NOISE_PATH + '/' + noise_list[i]
        noise_audio, _ = sf.read(noise_path)

        for j in range(len(clean_list)):
            clean_path = CLEAN_PATH + '/' + clean_list[j]
            noise_audio, _ = sf.read(noise_path)
            clean_audio, _ = sf.read(clean_path)

            tmp_spe = clean_list[j].split(".")[0]
            tmp_noi = noise_list[i].split(".")[0]

            n_sample = len(clean_audio)
            nframe = (n_sample - WIN_LEN) // WIN_OFFSET + 1  # WIN_LEN = 320    WIN_OFFSET = 160
            n_sample = (nframe + 1) * WIN_OFFSET
            # 获取相同长度的语音和噪声
            clean_audio_re = clean_audio[0: n_sample]
            noise_audio_re = noise_audio[0: n_sample]
            # data = np.stack((clip, clean, noise), 1)  # 横向拼接

            # 加噪
            mix_wav_path = os.path.join(DATA_PATH, 'mix_{}_wav'.format(data_type), '{}_{}dB.{}.{}.wav'.format(data_type, snr, tmp_spe, tmp_noi))
            noise_wav_path = os.path.join(DATA_PATH, 'noise_{}_wav'.format(data_type), '{}_{}dB.{}.{}.wav'.format(data_type, snr, tmp_spe, tmp_noi))
            mix_wav = clean_audio_re + noise_audio_re * _calc_alpha(SNR=snr, speech=clean_audio_re, noise=noise_audio_re)
            noise_wav = noise_audio_re * _calc_alpha(SNR=snr, speech=clean_audio_re, noise=noise_audio_re)
            create_folder(os.path.dirname(mix_wav_path))
            create_folder(os.path.dirname(noise_wav_path))
            sf.write(file=mix_wav_path, data=mix_wav, samplerate=SAMPLE_RATE)
            sf.write(file=noise_wav_path, data=noise_wav, samplerate=SAMPLE_RATE)

    pass

# def get_mean_std(mean_std_path):
#     print('——————————求均值以及标准差——————————')
#     tr_total = []
#     featureDataCreator = FeatureDataCreator()
#
#     # 训练集数据
#     tr_list = gen_list(TR_PATH, '.wav')    # '/data02/limiao/data/BIGDATA_SNR_PL_LSTM/mix_train_spec_mask'
#     for i in range(len(tr_list)):
#         data_path = TR_PATH + '/' + tr_list[i]
#         tr_mix_dataset = SpeechMixDataset(TR_PATH, 'tr')
#         tr_batch_dataloader = BatchDataLoader(tr_mix_dataset, MEAN_BATCH_SIZE, is_shuffle=True, workers_num=1)
#         for i, batch_info in enumerate(tr_batch_dataloader.get_dataloader()):
#             feature_ = featureDataCreator(batch_info)
#             tr_mix_wav = feature_.mix_feat_b[0].to(device)
#             tr_mix_wav_spec_real, tr_mix_wav_spec_img = STFT.transformri(tr_mix_wav)
#             tr_mix_wav_spec_real = tr_mix_wav_spec_real.permute(0, 2, 1)
#             tr_mix_wav_spec_img = tr_mix_wav_spec_img.permute(0, 2, 1)
#             amplitude_x = torch.sqrt(tr_mix_wav_spec_real ** 2 + tr_mix_wav_spec_img ** 2)  # 计算幅度
#             tr_mix_wav_log = _cal_log(amplitude_x)  # 计算LPS
#
#             tr_mix_wav_log = torch.squeeze(tr_mix_wav_log)
#
#             tr_mix_wav_log = tr_mix_wav_log.cuda().data.cpu().detach().numpy()
#             tr_total.append(tr_mix_wav_log)
#
#     tr_total_con = np.concatenate(tr_total, axis=1)   # 横向拼接
#     tr_total_con_tensor = torch.Tensor(tr_total_con).float().to(device)
#     tr_mean = torch.mean(tr_total_con_tensor)
#     tr_std = torch.std(tr_total_con_tensor)
#     data = [tr_mean, tr_std]
#
#
#     pickle.dump(data, open(mean_std_path, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
#
#
#
#     pass


def get_mean_std(mean_std_path):
    print('——————————求均值以及标准差——————————')
    tr_total = []
    tr_total_tmp = []
    featureDataCreator = FeatureDataCreator()

    # 训练集数据
    tr_mix_dataset = SpeechMixDataset(TR_PATH, 'tr')
    tr_batch_dataloader = BatchDataLoader(tr_mix_dataset, MEAN_BATCH_SIZE, is_shuffle=True, workers_num=1)
    for i, batch_info in enumerate(tr_batch_dataloader.get_dataloader()):
        feature_ = featureDataCreator(batch_info)
        tr_mix_wav = feature_.mix_feat_b[0].to(device)
        tr_mix_wav_spec_real, tr_mix_wav_spec_img = STFT.transformri(tr_mix_wav)
        tr_mix_wav_spec_real = tr_mix_wav_spec_real.permute(0, 2, 1)
        tr_mix_wav_spec_img = tr_mix_wav_spec_img.permute(0, 2, 1)
        amplitude_x = torch.sqrt(tr_mix_wav_spec_real ** 2 + tr_mix_wav_spec_img ** 2)  # 计算幅度
        tr_mix_wav_log = _cal_log(amplitude_x)  # 计算LPS

        tr_mix_wav_log = torch.squeeze(tr_mix_wav_log)

        tr_mix_wav_log = tr_mix_wav_log.cuda().data.cpu().detach().numpy()
        tr_total.append(tr_mix_wav_log)


    tr_total_con = np.concatenate(tr_total, axis=0)  # 横向拼接
    tr_total_con_tensor = torch.Tensor(tr_total_con).float().to(device)
    tr_mean = torch.mean(tr_total_con_tensor)
    tr_std = torch.std(tr_total_con_tensor)
    data = [tr_mean, tr_std]


    pickle.dump(data, open(mean_std_path, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)



    pass


def main():

    # cut_data('train')
    # cut_data('test')
    # cut_data('valid')
    #
    # cutNoise()
    #
    # gen_mixdata(data_type='train', snr=-5)
    # print("Get data successfully!")
    #
    # gen_mixdata(data_type='train', snr=0)
    # print("Get data successfully!")
    #
    # gen_mixdata(data_type='train', snr=5)
    # print("Get data successfully!")
    #
    # gen_mixdata(data_type='test', snr=-5)
    # print("Get data successfully!")
    #
    # gen_mixdata(data_type='test', snr=0)
    # print("Get data successfully!")
    #
    # gen_mixdata(data_type='test', snr=5)
    # print("Get data successfully!")
    #
    # gen_mixdata(data_type='valid', snr=-5)
    # print("Get data successfully!")
    #
    # gen_mixdata(data_type='valid', snr=0)
    # print("Get data successfully!")
    #
    # gen_mixdata(data_type='valid', snr=5)
    # print("Get data successfully!")
    mean_std_path = MEAN_STD_PATH
    get_mean_std(mean_std_path)

    pass

if __name__ == '__main__':

    main()