# from gen_train_data import TrainDataCreator
from BIGDATA_SNR_PL_LSTM_DATASET.compute_loss import *
import torch.nn as nn
from BIGDATA_SNR_PL_LSTM_DATASET.gen_feat import BatchDataLoader, SpeechMixDataset, FeatureDataCreator
from BIGDATA_SNR_PL_LSTM_DATASET.model_handle import *
from BIGDATA_SNR_PL_LSTM_DATASET.net_lstm import NetLstm
from BIGDATA_SNR_PL_LSTM_DATASET.config import *
from BIGDATA_SNR_PL_LSTM_DATASET.util import gen_list, _cal_log
import os
from torch.autograd.variable import *
import torch
import time
import logging as log
from BIGDATA_SNR_PL_LSTM_DATASET.log import Logger
import subprocess
from BIGDATA_SNR_PL_LSTM_DATASET.progressbar import progressbar as pb


import torch
import inspect

from torchvision import models
from BIGDATA_SNR_PL_LSTM_DATASET.gpu_mem_track import  MemTracker

frame = inspect.currentframe()          # define a frame to track
gpu_tracker = MemTracker(frame)         # define a GPU tracker


# from get_mean_variance import cal_mean_variance, get_mean_variance\
device = torch.device("cuda:" + str(CUDA_ID[0]) if torch.cuda.is_available() else "cpu")
STFT = STFT(WIN_LEN, WIN_OFFSET).cuda(CUDA_ID[0])
def write_log(file,name, train):
    message = ''
    for m, val in enumerate(train):
        message += ' --TRerror%i=%.3f ' % (m, val.data.numpy())
    file.write(name + ' ')
    file.write(message)
    file.write('/n')
def check_point():
    if os.path.exists(BEST_MODEL_PATH + '/' + MODEL_NAME):
        return True
    return False

def to_est_spec(feature_info_):

    feature_ = featureDataCreator(feature_info_)
    input_data_c1 = feature_.mix_feat_b[0].to(device)

    real_part, imag_part = STFT.transformri(input_data_c1)    # 提取STFT特征

    # 计算amplitude
    real_part = real_part.permute(0, 2, 1)
    imag_part = imag_part.permute(0, 2, 1)
    amplitude_x = torch.sqrt(real_part ** 2 + imag_part ** 2)
    # 计算LPS
    lps_x = _cal_log(amplitude_x)

    # # model是我们加载的模型
    # # input是实际中投入的input（Tensor）变量
    #
    # # 利用clone()去复制一个input，这样不会对input造成影响
    # input_ = lps_x.clone()
    # # 确保不需要计算梯度，因为我们的目的只是为了计算中间变量而已
    # input_.requires_grad_(requires_grad=False)
    #
    # mods = list(net_work.modules())
    # out_sizes = []
    #
    # for i in range(1, len(mods)):
    #     m = mods[i]
    #     # 注意这里，如果relu激活函数是inplace则不用计算
    #     if isinstance(m, nn.ReLU):
    #         if m.inplace:
    #             continue
    #     if i == 2:
    #         out, (h, c) = m(input_)
    #         out_sizes.append(np.array(out.size()))
    #         out_sizes.append(np.array(h.size()))
    #         out_sizes.append(np.array(c.size()))
    #     else:
    #         out = m(input_)
    #         out_sizes.append(np.array(out.size()))
    #
    #     input_ = out
    #
    # total_nums = 0
    # for i in range(len(out_sizes)):
    #     s = out_sizes[i]
    #     nums = np.prod(np.array(s))
    #     total_nums += nums
    #
    # # 打印两种，只有 forward 和 foreward、backward的情况
    # print('Model {} : intermedite variables: {:3f} M (without backward)'
    #       .format(net_work._get_name(), total_nums * 4 / 1000 / 1000))
    # print('Model {} : intermedite variables: {:3f} M (with backward)'
    #       .format(net_work._get_name(), total_nums * 4 * 2 / 1000 / 1000))


    # input_feature = torch.stack([STFT_C1,phase_C1],-1)

    # gpu_tracker.track()  # run function between the code line where uses GPU

    est_ = net_work(input_data_c1=lps_x)


    return feature_, est_
def train_one_epoch(epoch, net_work, batch_dataloader, optim, loss_helper):

    net_work.train()
    # for param in net_work.cnn_net.parameters():
    #     param.requires_grad=False
    pbar1 = pb(0, tr_batch_num)
    pbar1.start()
    step = 1
    for i, batch_info in enumerate(batch_dataloader.get_dataloader()):
        tr_start_time = time.time()
        optim.zero_grad()
        train_info, est_spec = to_est_spec(batch_info)
        loss_info_dict = {LOSS_EST_KEY: est_spec,
                          LOSS_LABEL_KEY: train_info.label_mask_b,
                          LOSS_MASK_FOR_LOSS_KEY: train_info.mask_for_loss_b,
                          LOSS_NFRAMES_KEY: train_info.nframe_b,
                          LOSS_MIX_SPEC_KEY: train_info.mix_spec_b}
        loss_info = LossInfo(loss_info_dict)
        loss_ori = loss_helper.gen_loss(loss_info=loss_info)
        loss_ori.backward()
        # loss_list.append(loss_ori)
        optim.step()
        tr_end_time = time.time()
        time_gap = tr_end_time - tr_start_time
        log.info('\ntr_epoch_{}_step_{}___loss:{}  and   train one step spend time: {}'.format(epoch, step, loss_ori, time_gap))
        step += 1

    pbar1.finish()

    # return loss_ori, loss_list
    return loss_ori
def cv_one_epoch(epoch, net_work, batch_dataloader, loss_helper):
    net_work.eval()
    # for param in net_work.cnn_net.parameters():
    #     param.requires_grad=False
    pbar1 = pb(0, cv_batch_num)
    pbar1.start()
    # loss_list = []
    step = 1
    for i, batch_info in enumerate(batch_dataloader.get_dataloader()):
        cv_start_time = time.time()
        train_info, est_spec = to_est_spec(batch_info)
        loss_info_dict = {LOSS_EST_KEY: est_spec,
                          LOSS_LABEL_KEY: train_info.label_mask_b,
                          LOSS_MASK_FOR_LOSS_KEY: train_info.mask_for_loss_b,
                          LOSS_NFRAMES_KEY: train_info.nframe_b,
                          LOSS_MIX_SPEC_KEY: train_info.mix_spec_b}
        loss_info = LossInfo(loss_info_dict)
        loss_ori = loss_helper.gen_loss(loss_info=loss_info)
        cv_end_time = time.time()
        time_gap = cv_end_time - cv_start_time
        log.info('\ncv_epoch_{}_step_{}___loss:{}    and    valid one step spend time: {}'.format(epoch, step, loss_ori, time_gap))
        step += 1

    pbar1.finish()

    # return loss_ori, loss_list
    return loss_ori


if __name__ == '__main__':
    log = Logger(LOG_FILE_NAME, level='info').logger
    log.info('\n\n\n\n\n\nstart.......\n\n\n\n\n\n')
    log.info('\nbatch_size={}'.format(TR_BATCH_SIZE))
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)
    if not os.path.exists(BEST_MODEL_PATH):
        os.makedirs(BEST_MODEL_PATH)
    featureDataCreator = FeatureDataCreator()
    net_work = NetLstm()
    device = torch.device("cuda:" + str(CUDA_ID[0]) if torch.cuda.is_available() else "cpu")
    net_work = nn.DataParallel(net_work, device_ids=CUDA_ID)
    net_work.to(device)
    nb_param_q = sum(p.numel() for p in net_work.parameters() if p.requires_grad)
    log.info("\nNumber of trainable parameters : " + str(nb_param_q))
    loss_helper = LossHelper()
    optim = torch.optim.Adam(net_work.parameters(), lr=LR, amsgrad=True)
    log.info('\nSTART TRAINING...')
    tr_lst = gen_list(TR_PATH, '.wav')
    tr_mix_len = len(tr_lst)         # 1800
    tr_batch_num = tr_mix_len // TR_BATCH_SIZE

    tr_mix_dataset = SpeechMixDataset(TR_PATH, 'tr')
    tr_batch_dataloader = BatchDataLoader(tr_mix_dataset, TR_BATCH_SIZE, is_shuffle=True, workers_num=1)
    if USE_CV:
        cv_lst = gen_list(CV_PATH, '.wav')
        cv_mix_len = len(cv_lst)     # 200
        cv_batch_num = cv_mix_len // CV_BATCH_SIZE

        cv_mix_dataset = SpeechMixDataset(CV_PATH, 'cv')
        cv_batch_dataloader = BatchDataLoader(cv_mix_dataset, CV_BATCH_SIZE, is_shuffle=False, workers_num=1)
    bestmode_full_path = os.path.join(BEST_MODEL_PATH, MODEL_NAME)
    if RESUME_MODEL:
        log.info('\nRESUME PRE MODEL')
        EXIST = check_point()
        if EXIST:
            log.info('\n{}'.format(bestmode_full_path))
            optim_dict, best_loss = resume_model(net_work, bestmode_full_path)
            print(best_loss)
            optim.load_state_dict(optim_dict)
        else:
            log.info('\nMODEL NO EXIST, TRAIN NEW MODEL')
            best_loss = float("inf")
    log.info('\n\n\n\n\n\nSTART TRAINING...\n\n\n\n\n\n')
    # if USE_CV:
    #     [best_loss, _] = cv_one_epoch(net_work, cv_batch_dataloader, loss_helper)
    for epoch in range(max_epochs):
        goodmode_full_paths = os.path.join(GOOD_MODEL_PATHs, 'epoch_{}_{}'.format(epoch + 1, MODEL_NAME))
        start_time = time.time()
        tr_loss = train_one_epoch(epoch + 1, net_work, tr_batch_dataloader, optim, loss_helper)
        end_time = time.time()
        log.info('\ntrain one epoch time: {}'.format(end_time - start_time))
        if USE_CV:
            cv_loss = cv_one_epoch(epoch + 1, net_work, cv_batch_dataloader, loss_helper)
        end_time2 = time.time()
        mode_full_path = os.path.join(MODEL_PATH, 'epoch_{}_{}'.format(epoch + 1, MODEL_NAME))
        if cv_loss < best_loss:
            best_loss = cv_loss
            save_model(net_work, optim, tr_loss, models_path=bestmode_full_path)
            save_model(net_work, optim, tr_loss, models_path=goodmode_full_paths)
        save_model(net_work, optim, tr_loss, models_path=mode_full_path)
        log.info(
            "\nITERATION %d: TRAIN ONE EPOCH LAST LOSS %.6f, BSET_LOSS %.6f, (lrate%e)"
            " , %s (%s), TIME USED: %.2fs" % (
                epoch + 1, tr_loss, best_loss, LR,
                "nnet accepted", 'epoch_{}_{}'.format(epoch + 1, MODEL_NAME),
                (end_time2 - start_time) / 1))