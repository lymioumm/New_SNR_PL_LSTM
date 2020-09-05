# from gen_train_data import TrainDataCreator
import pickle
import time

import librosa

from BIGDATA_SNR_PL_LSTM_DATASET.compute_loss import *
import torch.nn as nn
from BIGDATA_SNR_PL_LSTM_DATASET.gen_feat import BatchDataLoader, SpeechMixDataset, FeatureDataCreator
from BIGDATA_SNR_PL_LSTM_DATASET.model_handle import *
from BIGDATA_SNR_PL_LSTM_DATASET.net_lstm import NetLstm
from BIGDATA_SNR_PL_LSTM_DATASET.config import *
from BIGDATA_SNR_PL_LSTM_DATASET.util import gen_list, create_folder, _cal_log
from BIGDATA_SNR_PL_LSTM_DATASET.log import Logger

import os
from torch.autograd.variable import *
import soundfile as sf
import torch
from pypesq import pesq
from pystoi import stoi
import subprocess
# from get_mean_variance import cal_mean_variance, get_mean_variance
from BIGDATA_SNR_PL_LSTM_DATASET.stft_istft import STFT
device = torch.device("cuda:" + str(CUDA_ID[0]) if torch.cuda.is_available() else "cpu")
# STFT = STFT(WIN_LEN, WIN_OFFSET).cuda(CUDA_ID[0])
STFT = STFT(WIN_LEN, WIN_OFFSET).to(device)

def Generate_txt(path):
    a = 0
    # /data/limiao/SNR_PL_data/new/features/spectrograms/train/MixSNR
    # /data/limiao/SNR_PL_data/new/batch_data/MixSNR/train
    # dir = '/data/limiao/SNR_PL_data/new/batch_data/MixSNR/train'  # 语音数据文件的地址
    label = a
    # os.listdir的结果就是一个list集，可以使用list的sort方法来排序。如果文件名中有数字，就用数字的排序
    files = os.listdir(path)  # 列出dirname下的目录和文件
    # files.sort()  # 排序
    score = open('./score.txt', 'w')
    # text = open('./test.txt', 'w')
    for file in files:
        fileType = os.path.split(file)  # os.path.split()：按照路径将文件名和路径分割开
        if fileType[1] == '.txt':
            continue
        name = str(dir) + '/' + file + ' ' + str(int(label)) + '\n'
        label += 1
        score.write(name)
        score.flush()
    score.close()


    pass

def test_model(epoch, snr, score_txt):
    log = Logger(LOG_FILE_NAME_TEST, level='info').logger

    featureDataCreator = FeatureDataCreator()
    net_work = NetLstm()
    device = torch.device("cuda:" + str(CUDA_ID[0]) if torch.cuda.is_available() else "cpu")
    net_work = nn.DataParallel(net_work, device_ids=CUDA_ID)
    net_work.to(device)
    loss_helper = LossHelper()
    # optim = torch.optim.Adam(net_work.parameters(), lr=LR)
    log.info('START TESTING...\n')
    TRAIN = False
    net_work.eval()
    if not os.path.exists(TT_OUT_PATH):
        os.makedirs(TT_OUT_PATH)
    if not os.path.exists(TT_OUT_PATH_FORDOWNLOAD):
        os.makedirs(TT_OUT_PATH_FORDOWNLOAD)
    # model_name = MODEL_NAME
    # epoch = 182
    model_name = 'epoch_{}_nnet_model.pickle'.format(epoch)
    # epoch_1_nnet_model.pickle
    # bestmode_full_path = os.path.join(BEST_MODEL_PATH, model_name)
    # model_general_path = '/home/ZhangXueLiang/LiMiao/pycharmProjects/speech_declip_crnn/model/epoch_{}_nnet_model.pickle'.format(epoch)
    model_general_path = os.path.join(MODEL_PATH, model_name)
    # optim_dict, loss, net_state_dict= resume_model_test(net_work, bestmode_full_path)   # 网络模型也会在此加载
    optim_dict, loss, net_state_dict = resume_model_test(net_work, model_general_path)  # 网络模型也会在此加载
    log.info(f'epoch_{epoch}_snr is :{snr}dB  loss:{loss}\n')
    # print(f'snr is :{snr}dB  loss:{loss}')
    # optim.load_state_dict(optim_dict)
    # STFT = STFT(WIN_LEN, WIN_OFFSET).cuda(CUDA_ID[0])
    # tt_mix_dataset = SpeechMixDataset(TT_MASK_PATH, 'tt')
    # tt_batch_dataloader = BatchDataLoader(tt_mix_dataset, TT_BATCH_SIZE, is_shuffle=False, workers_num=4)
    # tt_lst = gen_list(TT_PATH, '.wav')
    # tt_len = len(tt_lst)
    #
    # tt_lst = gen_list(TT_PATH, '.wav')
    # tt_mix_len = len(tt_lst)  # 200
    # /data01/limiao/BIGDATA_SNR_PL_LSTM/mix_test_wav_0dB
    TT_PATH_SNR = DATA_PATH + '/mix_test_wav_{}dB/'.format(snr)
    tt_mix_dataset = SpeechMixDataset(TT_PATH_SNR, 'tt')
    tt_batch_dataloader = BatchDataLoader(tt_mix_dataset, TT_BATCH_SIZE, is_shuffle=False, workers_num=1)
    # TT_batch_num = tt_mix_len // TT_BATCH_SIZE

    net_work.eval()

    pesq_score_est_list = []
    pesq_score_mix_list = []
    pesq_gap_list = []

    stoi_score_est_list = []
    stoi_score_mix_list = []
    stoi_gap_list = []

    score_txt_dir = os.path.join('./score_txt')
    create_folder(score_txt_dir)
    score_est_txt = open('{}/{}dB_{}_est_score.txt'.format(score_txt, snr, model_name), 'a')
    score_est_txt.write('\n\n\n------------------New Testing------------------\n')
    score_est_txt.write('                                                                       ' + 'score_pesq' + '                    ' + 'score_stoi' + '\n')

    score_est_txt.flush()
    score_est_txt.close()

    score_mix_txt = open('{}/{}dB_{}_mix_score.txt'.format(score_txt, snr, model_name), 'a')
    score_mix_txt.write('\n\n\n------------------New Testing------------------\n')
    score_mix_txt.write('                                                                      ' + 'score_pesq' + '                    ' + 'score_stoi' + '\n')

    score_mix_txt.flush()
    score_mix_txt.close()

    score_gap_txt = open('{}/{}dB_{}_gap_score.txt'.format(score_txt, snr, model_name), 'a')
    score_gap_txt.write('\n\n\n------------------New Testing------------------\n')
    score_gap_txt.write('                                                                      ' + 'score_pesq' + '                     ' + 'score_stoi' + '\n')

    score_gap_txt.flush()
    score_gap_txt.close()
    for i, batch_info in enumerate(tt_batch_dataloader.get_dataloader()):
        feature_ = featureDataCreator(batch_info)
        input_data_c1 = feature_.mix_feat_b[0].to(device)
        # input_data_c1 = train_info_.mix_feat_b
        real_part, imag_part = STFT.transformri(input_data=input_data_c1)  # 提取带噪测试数据的STFT特征
        # 计算amplitude
        real_part = real_part.permute(0, 2, 1)
        imag_part = imag_part.permute(0, 2, 1)
        amplitude_x = torch.sqrt(real_part ** 2 + imag_part ** 2)  # 计算幅度
        # mix_phase = torch.autograd.Variable(torch.atan(imag_part.data/real_part.data))
        phase_x = torch.autograd.Variable(torch.atan2(imag_part.data, real_part.data))  # 计算相位
        # # 验证傅里叶变换
        # y_label = feature_.label_mask_b[0]
        # real_part_y, imag_part_y = STFT.transformri(y_label)
        #
        # real_part_y = real_part_y.permute(0, 2, 1)
        # imag_part_y = imag_part_y.permute(0, 2, 1)
        # amplitude_y = torch.sqrt(real_part_y ** 2 + imag_part_y ** 2)
        #
        # y_phase = torch.autograd.Variable(torch.atan2(imag_part_y.data, real_part_y.data))
        #
        #
        # real_tmp = amplitude_y * torch.cos(y_phase)      # 验证恢复后的实部
        # imag_tmp = amplitude_y * torch.sin(y_phase)      # 验证恢复后的虚部
        #
        # clean_speech_x = STFT.inverse(torch.stack([real_tmp, imag_tmp], 3))
        # clean_speech_xx = torch.squeeze(clean_speech_x)
        # clean_speech_xxx = clean_speech_xx.cuda().data.cpu().detach().numpy()
        #
        # clean_speech_ = feature_.label_mask_b[0]
        # clean_speech_tmp = torch.squeeze(clean_speech_)
        # clean_speech = clean_speech_tmp.cuda().data.cpu().detach().numpy()
        #
        # score_est = pesq(clean_speech, clean_speech_xxx, SAMPLE_RATE)
        # print(f'scroe_est: {score_est}')

        # 计算LPS
        lps_x = _cal_log(amplitude_x)  # 计算LPS

        # input_feature = torch.stack([STFT_C1,phase_C1],-1)

        est_ = net_work(input_data_c1=lps_x)
        # # 计算测试数据的loss
        # y_label = feature_.label_mask_b[0]
        # real_part_y, imag_part_y = STFT.transformri(y_label)
        #
        # real_part_y = real_part_y.permute(0, 2, 1)
        # imag_part_y = imag_part_y.permute(0, 2, 1)
        # amplitude_y = torch.sqrt(real_part_y ** 2 + imag_part_y ** 2)
        # # 计算LPS
        # lps_y = _cal_log(amplitude_y)
        #
        # est_tmp = torch.exp(est_) - 1.0
        #
        #
        #
        # # cost1 = torch.pow(est_tmp - lps_y, 2)
        # cost1 = torch.pow(est_ - lps_y, 2)
        # # sum_mask0 = torch.sum(loss_info.mask_for_loss[:, :, :FFT_SIZE], dim=1)
        # sum_mask0 = torch.sum(feature_.mask_for_loss_b[:, :, :FFT_SIZE])
        # mask = feature_.mask_for_loss_b[:, :, : FFT_SIZE]
        # cost0_ = cost1 * mask
        # # cost0 = torch.sum(cost0, dim=1) / sum_mask0
        # sum_cost0 = torch.sum(cost0_)
        # cost0 = sum_cost0 / sum_mask0
        # print(f'loss:{cost0}')
        est_x = torch.exp(est_) - 1.0  # amplitude

        y_label = feature_.label_mask_b[0]
        real_part_y, imag_part_y = STFT.transformri(y_label)

        real_part_y = real_part_y.permute(0, 2, 1)
        imag_part_y = imag_part_y.permute(0, 2, 1)
        amplitude_y = torch.sqrt(real_part_y ** 2 + imag_part_y ** 2)
        # 计算LPS
        lps_y = _cal_log(amplitude_y)

        # no_in = feature_.mix_feat_b[0].to(device)
        # no_real_part, no_imag_part = STFT.transformri(input_data=no_in)  # 提取带噪测试数据的STFT特征
        # 计算amplitude
        # no_real_part = no_real_part.permute(0, 2, 1)
        # no_imag_part = no_imag_part.permute(0, 2, 1)
        # no_amplitude_x = torch.sqrt(no_real_part ** 2 + no_imag_part ** 2)  # 计算幅度
        # mix_phase = torch.autograd.Variable(torch.atan(imag_part.data/real_part.data))
        # no_phase_x = torch.autograd.Variable(torch.atan2(no_imag_part.data, no_real_part.data))  # 计算相位
        #
        # # no_real_c =
        #
        # no_cost1 = torch.pow(amplitude_y - no_amplitude_x, 2)
        # # sum_mask0 = torch.sum(loss_info.mask_for_loss[:, :, :FFT_SIZE], dim=1)
        # sum_mask0 = torch.sum(feature_.mask_for_loss_b[:, :, :FFT_SIZE])
        # mask = feature_.mask_for_loss_b[:, :, : FFT_SIZE]
        # no_cost0_ = no_cost1 * mask
        # # cost0 = torch.sum(cost0, dim=1) / sum_mask0
        # no_sum_cost0 = torch.sum(no_cost0_)
        # no_cost0 = no_sum_cost0 / sum_mask0
        # print(f'loss:{no_cost0}')

        # cost1 = torch.pow(est_tmp - lps_y, 2)
        cost1 = torch.pow(est_x - lps_y, 2)
        # sum_mask0 = torch.sum(loss_info.mask_for_loss[:, :, :FFT_SIZE], dim=1)
        sum_mask0 = torch.sum(feature_.mask_for_loss_b[:, :, :FFT_SIZE])
        mask = feature_.mask_for_loss_b[:, :, : FFT_SIZE]
        cost0_ = cost1 * mask
        # cost0 = torch.sum(cost0, dim=1) / sum_mask0
        sum_cost0 = torch.sum(cost0_)
        cost0 = sum_cost0 / sum_mask0
        # print(f'loss:{cost0}')

        real_part_est = est_x * torch.cos(phase_x)
        imag_part_est = est_x * torch.sin(phase_x)

        # est_speech_ = STFT.inverse(torch.stack([est_y, mix_phase], 3))
        est_speech_ = STFT.inverse(torch.stack([real_part_est, imag_part_est], 3))
        est_speech = torch.squeeze(est_speech_)
        est_speech_numpy = est_speech.cuda().data.cpu().detach().numpy()

        mix_wav_name = batch_info.filename_batch[0][0]
        clean_wav_name = batch_info.filename_batch[1][0]
        Enh_wav_name = 'epoch_{}_Enh_{}'.format(epoch, mix_wav_name)
        log.info(f'epoch_{epoch}_wav_name:{mix_wav_name}: loss:{cost0}\n')
        # print(f'wav_name:{mix_wav_name}: loss:{cost0}')



        # 创建存放增强后语音数据的文件
        test_Enh_path = os.path.join(TT_OUT_PATH, 'model_epoch_{}'.format(epoch), Enh_wav_name)
        if not os.path.exists(os.path.dirname(test_Enh_path)):
            os.makedirs(os.path.dirname(test_Enh_path))
        sf.write(file=test_Enh_path, data=est_speech_numpy, samplerate=16000)

        # clean_wav_path = os.path.join(CLEAN_TT_PATH, clean_wav_name)
        # mix_wav_path = os.path.join(TT_PATH, mix_wav_name)
        # clean_speech, _ = sf.read(clean_wav_path)
        # mix_speech, _ = sf.read(mix_wav_path)
        # train_info, est_spec = to_est_spec(batch_info)

        # for i in range(len(tt_lst)):
        #     test_wav_path = os.path.join(TT_PATH, tt_lst[i])
        #     # data = pickle.load(open(test_wav_path, 'rb'))
        #     # [mix_wav_spec, clean_wav_irm, clean_spec] = data
        #     # mix_wav_spec_real = np.real(mix_wav_spec)
        #     # mix_wav_spec_img = np.imag(mix_wav_spec)
        #     #
        #     # mix_wav_spec_real = torch.Tensor(mix_wav_spec_real).float().permute(1, 0)
        #     # mix_wav_spec_img = torch.Tensor(mix_wav_spec_img).float().permute(1, 0)
        #     #
        #     # mix_amplitude = torch.sqrt(mix_wav_spec_real ** 2 + mix_wav_spec_img ** 2)
        #     # mix_amplitude_log = torch.log(mix_amplitude + 1e-8)
        #     # feature_ = []
        #     #
        #     # mix_amplitude_log_three = mix_amplitude_log.view(1, -1, 161)
        #     est_irm = net_work(input_data_c1=mix_amplitude_log_three, feature_=feature_)
        #
        #     numpy_test_output_target2_con_out = est_irm.cpu().detach().numpy()  # 将tensor类型转换为numpy类型
        #
        #     numpy_test_output_two = numpy_test_output_target2_con_out.reshape(-1, 161)
        #     # 得到目标掩码后，乘上带噪语音得到测试数据降噪后的结果
        #     test_Enh = mix_wav_spec * numpy_test_output_two.T
        #
        #     # 合成降噪后的语音,并写入文件目录
        #     est_pcm = librosa.istft(test_Enh, win_length=320, hop_length=160)
        #     na_Enh0 = tt_lst[i].split('.')[0]
        #     na_Enh1 = tt_lst[i].split('.')[1]
        #     na_Enh2 = tt_lst[i].split('.')[2]
        #     na_Enh = na_Enh0 + '.' + na_Enh1 + '.' + na_Enh2
        #
        #     na_mix_tmp = na_Enh0.split('_')[0] + '_' + na_Enh0.split('_')[1] + '.' + na_Enh1 + '.' + na_Enh2
        #
        #     # 创建存放增强后语音数据的文件
        #     test_Enh_path = os.path.join(TT_IRM_OUT_PATH, 'model', '{}'.format(model_name),
        #                                  'Enh_%s.wav' % na_Enh)
        #     create_folder(os.path.dirname(test_Enh_path))
        #     # 写入语音数据至上一步创建的文件内
        #     sf.write(file=test_Enh_path, data=est_pcm, samplerate=16000)
        #
        #     # CLEAN_PATH = DATA_PATH + '/{}_data/'.format(data_type)
        #
        #     test_clean_path = os.path.join(DATA_PATH, 'test_data', '{}.wav'.format(na_Enh1))
        #
        #
        #     test_mix_path = os.path.join(DATA_PATH, 'new', 'mix_test_wav', '{}.wav'.format(na_mix_tmp))
        #
        #     est_speech_audio, _ = sf.read(test_Enh_path)
        #     mix_speech_audio, _ = sf.read(test_mix_path)
        #     cle_speech_audio, _ = sf.read(test_clean_path)
        #

        mix_speech_ = feature_.mix_feat_b[0]
        mix_speech_tmp = torch.squeeze(mix_speech_)
        mix_speech = mix_speech_tmp.cuda().data.cpu().detach().numpy()
        clean_speech_ = feature_.label_mask_b[0]
        clean_speech_tmp = torch.squeeze(clean_speech_)
        clean_speech = clean_speech_tmp.cuda().data.cpu().detach().numpy()
        # n_sample = len(clean_speech)
        # n_sample = feature_.nsample_b[0]
        # nframe = (n_sample - WIN_LEN) // WIN_OFFSET + 1
        # nframe = (n_sample - WIN_LEN) // WIN_OFFSET + 1
        # n_sample = (nframe + 1) * WIN_OFFSET

        # clean_speech_re = clean_speech[0:n_sample]
        # mix_speech_re = mix_speech[0:n_sample]
        # est_speech_numpy_re = est_speech_numpy[0:n_sample]
        pesq_score_est = pesq(clean_speech, est_speech_numpy, SAMPLE_RATE)
        pesq_score_mix = pesq(clean_speech, mix_speech, SAMPLE_RATE)

        stoi_score_est = stoi(clean_speech, est_speech_numpy, SAMPLE_RATE, extended=False)
        stoi_score_mix = stoi(clean_speech, mix_speech, SAMPLE_RATE, extended=False)

        pesq_gap = pesq_score_est - pesq_score_mix
        stoi_gap = stoi_score_est - stoi_score_mix

        pesq_score_est_list.append(pesq_score_est)
        pesq_score_mix_list.append(pesq_score_mix)
        pesq_gap_list.append(pesq_gap)

        stoi_score_est_list.append(stoi_score_est)
        stoi_score_mix_list.append(stoi_score_mix)
        stoi_gap_list.append(stoi_gap)

        # score_est_txt = open('./score_txt/{}_est_score.txt'.format(model_name), 'a')
        score_est_txt = open('{}/{}dB_{}_est_score.txt'.format(score_txt, snr, model_name), 'a')

        # score_mix_txt = open('./score_txt/{}_mix_score.txt'.format(model_name), 'a')
        score_mix_txt = open('{}/{}dB_{}_mix_score.txt'.format(score_txt, snr, model_name), 'a')

        # score_gap_txt = open('./score_txt/{}_gap_score.txt'.format(model_name), 'a')
        score_gap_txt = open('{}/{}dB_{}_gap_score.txt'.format(score_txt, snr, model_name), 'a')

        score_est_txt.write(Enh_wav_name + '     :     ' + str(pesq_score_est) + '          ' + str(stoi_score_est) + '\n')
        score_mix_txt.write(mix_wav_name + '     :     ' + str(pesq_score_mix) + '          ' + str(stoi_score_mix) + '\n')
        score_gap_txt.write('gap_mix_est_{}'.format(mix_wav_name) + '     :     ' + str(pesq_gap) + '          ' + str(stoi_gap) + '\n')

        score_est_txt.flush()
        score_mix_txt.flush()
        score_gap_txt.flush()

    pesq_score_est_mean = np.mean(pesq_score_est_list)
    pesq_score_mix_mean = np.mean(pesq_score_mix_list)
    pesq_score_gap_mean = np.mean(pesq_gap_list)

    stoi_score_est_mean = np.mean(stoi_score_est_list)
    stoi_score_mix_mean = np.mean(stoi_score_mix_list)
    stoi_score_gap_mean = np.mean(stoi_gap_list)
    # print(f'pesq_score_gap_mean: {pesq_score_gap_mean}')
    log.info(f'pesq_score_gap_mean: {pesq_score_gap_mean}')
    # print(f'stoi_score_gap_mean: {stoi_score_gap_mean}')
    log.info(f'stoi_score_gap_mean: {stoi_score_gap_mean}')

    score_est_txt.write('Mean Socre' + '   :   ' + str(pesq_score_est_mean) + '          ' + str(stoi_score_est_mean) + '\n')
    score_mix_txt.write('Mean Socre' + '   :   ' + str(pesq_score_mix_mean) + '          ' + str(stoi_score_mix_mean) + '\n')
    score_gap_txt.write('Mean Socre' + '   :   ' + str(pesq_score_gap_mean) + '          ' + str(stoi_score_gap_mean) + '\n')

    score_est_txt.close()
    score_mix_txt.close()
    score_gap_txt.close()


    pass
def main():
    start_time = time.time()
    # test_model(epoch=182, snr=5, score_txt=SCORE_TXT)
    # test_model(epoch=182, snr=-5, score_txt=SCORE_TXT)
    # test_model(epoch=182, snr=0, score_txt=SCORE_TXT)
    # test_model(epoch=327, snr=-5, score_txt=SCORE_TXT)
    # test_model(epoch=327, snr=0, score_txt=SCORE_TXT)
    # test_model(epoch=327, snr=5, score_txt=SCORE_TXT)
    # test_model(epoch=170, snr=-5, score_txt=SCORE_TXT)
    # test_model(epoch=170, snr=0, score_txt=SCORE_TXT)
    # test_model(epoch=170, snr=5, score_txt=SCORE_TXT)
    # test_model(epoch=168, snr=-5, score_txt=SCORE_TXT)
    # test_model(epoch=168, snr=0, score_txt=SCORE_TXT)
    # test_model(epoch=168, snr=5, score_txt=SCORE_TXT)
    
    test_model(epoch=339, snr=-5, score_txt=SCORE_TXT)
    test_model(epoch=339, snr=0, score_txt=SCORE_TXT)
    test_model(epoch=339, snr=5, score_txt=SCORE_TXT)
    end_time = time.time()
    print('test time :{}'.format(end_time - start_time))


    pass
if __name__ == '__main__':
    main()