from torch.autograd.variable import *
from BIGDATA_SNR_PL_LSTM_DATASET.config import *
import numpy as np
from BIGDATA_SNR_PL_LSTM_DATASET.target_function import cIRM
from BIGDATA_SNR_PL_LSTM_DATASET.stft_istft import STFT
from BIGDATA_SNR_PL_LSTM_DATASET.util import _cal_log

from BIGDATA_SNR_PL_LSTM_DATASET.speech_processing import wav_segmentation
import scipy.io as sio

LOSS_EST_KEY = 'est_mask'
LOSS_LABEL_KEY = 'label_mask'
LOSS_MASK_FOR_LOSS_KEY = 'mask_for_loss'
LOSS_NFRAMES_KEY = 'nframes'
LOSS_NSAMPLES_KEY = 'nsamples'
LOSS_MIX_SPEC_KEY = 'mix_spec_b'
class LossInfo(object):
    def __init__(self, loss_info_dict):
        super(LossInfo, self).__init__()
        self.est = loss_info_dict[LOSS_EST_KEY]
        self.label = loss_info_dict[LOSS_LABEL_KEY]
        self.mask_for_loss = loss_info_dict[LOSS_MASK_FOR_LOSS_KEY]
        self.nframes = loss_info_dict[LOSS_NFRAMES_KEY]
        self.mix_spec_b = loss_info_dict[LOSS_MIX_SPEC_KEY]

class LossHelper(object):
    def __init__(self):
        super(LossHelper, self).__init__()
        # self.hanning = Variable(torch.FloatTensor(np.hanning(WIN_LEN)).cuda(), requires_grad=False)
        self.STFT = STFT(WIN_LEN, WIN_OFFSET).cuda(CUDA_ID[0])
        # self.Feature_frame = round((Guide_time * SAMPLE_RATE - WIN_LEN) // WIN_OFFSET + 1)
    def gen_loss(self, loss_info):
        return self.mse_loss(loss_info)

    def mse_loss(self, loss_info):
        real_part, imag_part = self.STFT.transformri(loss_info.label[0])
        real_part = real_part.permute([0, 2, 1])
        imag_part = imag_part.permute([0, 2, 1])
        # 计算干净语音的amplitude特征
        amplitude_y = torch.sqrt(real_part ** 2 + imag_part **2)
        # 计算干净语音的LPS特征
        lps_y = _cal_log(amplitude_y)

        est = loss_info.est
        cost1 = torch.pow(est - lps_y, 2)
        # sum_mask0 = torch.sum(loss_info.mask_for_loss[:, :, :FFT_SIZE], dim=1)
        sum_mask0 = torch.sum(loss_info.mask_for_loss[:, :, :FFT_SIZE])
        mask = loss_info.mask_for_loss[:, :, : FFT_SIZE]
        cost0_ = cost1 * mask
        # cost0 = torch.sum(cost0, dim=1) / sum_mask0
        sum_cost0 = torch.sum(cost0_)
        cost0 = sum_cost0 / sum_mask0
        return cost0  # [timedomain, real(CRM), imag(CRM)]

        # sio.savemat('tmp.mat', {'label': label.data.cpu().numpy(), 'est': loss_info.est[0].data.cpu().numpy()})