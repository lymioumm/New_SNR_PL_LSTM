import os
import soundfile as sf
import scipy
import random
import torch
from torch.nn.utils.rnn import *
from scipy import io as sio
import numpy as np
from BIGDATA_SNR_PL_LSTM_DATASET.config import *
from torch.utils.data import Dataset, DataLoader
from torch.autograd.variable import *
from BIGDATA_SNR_PL_LSTM_DATASET.util import gen_list, read_list
import struct
import mmap
TARGET_BATCH_KEY = 'target_batch'
MIX_BATCH_KEY = 'mix_batch'
MASK_FOR_LOSS_BATCH_KEY = 'mask_for_loss'
NFRAMES_BATCH_KEY = 'nframes_batch'
FILENAME_BATCH_KEY = 'filename_batch'
SAMPLE_NUM_BATCH_KEY = 'sample_num_batch'
class SpeechMixDataset(Dataset):
    def __init__(self, wav_dir, kind):
        self.kind = kind
        self.wav_dir = wav_dir
        self.wav_list = gen_list(self.wav_dir, '.wav')
        self.len = len(self.wav_list)

    def __len__(self):
        if self.kind == 'tr':
            return self.len
        else:
            return self.len
    def __getitem__(self, idx):
        wav_name = self.wav_list[idx]
        if self.kind == 'tr':
            clip, _ = sf.read(self.wav_dir + wav_name)

            clean_wav_name = wav_name.split('.')[1] + '.wav'
            # clean, _ = sf.read(CLEAN_TR_PATH + '/' + '%s.wav' % (clean_wav_name))    # DR4_MJDC0_SA2.wav
            clean, _ = sf.read(CLEAN_TR_PATH + '/' + clean_wav_name)  # DR4_MJDC0_SA2.wav
            mix, _ = sf.read(TR_PATH + '/' + wav_name)           # 'train_5dB.DR4_MJDC0_SA2.train_buccaneer1.wav'
            # clean, _ = sf.read(CLEAN_TR_PATH + wav_name[4:])
            noise, _ = sf.read(NOISE_TRAIN_PATH + '/' + wav_name)

        elif self.kind == 'cv':
            clip, _ = sf.read(self.wav_dir + wav_name)
            # clean, _ = sf.read(CLEAN_CV_PATH + wav_name[4:])
            clean_wav_name = wav_name.split('.')[1] + '.wav'

            # clean, _ = sf.read(CLEAN_CV_PATH + wav_name[4:])
            clean, _ = sf.read(CLEAN_CV_PATH + clean_wav_name)
            mix, _ = sf.read(CV_PATH + wav_name)
            noise, _ = sf.read(NOISE_VALID_PATH + wav_name)
        elif self.kind == 'tt':
            clip, _ = sf.read(self.wav_dir + wav_name)
            clean_wav_name = wav_name.split('.')[1] + '.wav'

            # clean, _ = sf.read(CLEAN_TT_PATH + wav_name[4:])
            clean, _ = sf.read(CLEAN_TT_PATH + clean_wav_name)
            mix, _ =sf.read(TT_PATH + wav_name)
            noise, _ = sf.read(NOISE_TEST_PATH + wav_name)

        # speech = Variable(torch.FloatTensor(speech.astype('float32')))
        # noise = Variable(torch.FloatTensor(noise.astype('float32')))
        n_sample = len(clean)
        nframe = (n_sample - WIN_LEN) // WIN_OFFSET + 1
        n_sample = (nframe + 1) * WIN_OFFSET
        clean = clean[0:n_sample]     # 干净语音
        clip = clip[0:n_sample]       # 带噪语音
        data = np.stack((clip, clean), 1)
        mask_for_loss = np.ones((nframe, WIN_LEN), dtype=np.float32)
        sample = (Variable(torch.FloatTensor(data.astype('float32'))),
                  Variable(torch.FloatTensor(mask_for_loss)),
                  wav_name,
                  clean_wav_name,
                  nframe,
                  n_sample
                  )
        return sample

class SMBatchInfo(object):
    def __init__(self, batch_dict):
        super(SMBatchInfo, self).__init__()
        self.target_batch = batch_dict[TARGET_BATCH_KEY]
        self.mask_for_loss_batch = batch_dict[MASK_FOR_LOSS_BATCH_KEY]
        self.filename_batch = batch_dict[FILENAME_BATCH_KEY]
        self.nframe_b = batch_dict[NFRAMES_BATCH_KEY]
        self.sample_num = batch_dict[SAMPLE_NUM_BATCH_KEY]
class BatchDataLoader(object):
    def __init__(self, s_mix_dataset, batch_size, is_shuffle=True, workers_num=16):
        self.dataloader = DataLoader(s_mix_dataset, batch_size=batch_size, shuffle=is_shuffle, num_workers=workers_num,
                                     collate_fn=self.collate_fn, drop_last=True)
    def get_dataloader(self):
        return self.dataloader
    @staticmethod
    def collate_fn(batch):
        batch.sort(key=lambda x: x[0].size()[0], reverse=True)
        data, mask_for_loss, filename1, filename2, nframe, nsample = zip(*batch)
        data = list(data)
        data = pad_sequence(data, batch_first=True)
        pad_mask_for_loss = pad_sequence(mask_for_loss, batch_first=True)
        batch_dict = {TARGET_BATCH_KEY: data,
                      MASK_FOR_LOSS_BATCH_KEY: pad_mask_for_loss,
                      FILENAME_BATCH_KEY: [filename1, filename2],
                      NFRAMES_BATCH_KEY: nframe,
                      SAMPLE_NUM_BATCH_KEY: nsample}
        # print('end get_dataloader')
        return SMBatchInfo(batch_dict)

class TrainDataBatch(object):
    def __init__(self, mix_feat_b, label_mask_b, mix_spec_b, sm_bath_info):
        super(TrainDataBatch, self).__init__()
        self.mix_feat_b = mix_feat_b
        self.label_mask_b = label_mask_b
        self.mix_spec_b = mix_spec_b
        self.mask_for_loss_b = sm_bath_info.mask_for_loss_batch.cuda(CUDA_ID[0])
        # self.mask_for_loss_b = sm_bath_info.mask_for_loss_batch
        self.nframe_b = sm_bath_info.nframe_b
        self.nsample_b = sm_bath_info.sample_num

class FeatureDataCreator(torch.nn.Module):
    def __init__(self):
        super(FeatureDataCreator, self).__init__()

    def forward(self, batch_info):
        data = batch_info.target_batch.cuda(CUDA_ID[0])
        # data = batch_info.target_batch
        clip = data[:, :, 0]         # 干净语音
        clean = data[:, :, 1]        # 带噪语音
        return TrainDataBatch(mix_feat_b=[clip], label_mask_b=[clean],
                              mix_spec_b=[clip], sm_bath_info=batch_info)