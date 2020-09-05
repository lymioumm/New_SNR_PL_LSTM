# # # import torch
# # # # def test():
# # # #     x = torch.randn(4, 5)
# # # #
# # # #     print(f'x:{x}')
# # # #
# # # #     print(f'x.sum(0):{x.sum(0)}')  # 按列求和
# # # #     print(f'x.sum(1):{x.sum(1)}')  # 按行求和
# # # #     print(f'torch.sum(x):{torch.sum(x)}')  # 按列求和
# # # #     print(f'torch.sum(x, dim=0):{torch.sum(x, dim=0)}')  # 按列求和
# # # #     print(f'torch.sum(x, dim=1):{torch.sum(x, dim=1)}')  # 按行求和
# # # #
# # # #     pass
# # # #
# # # #
# # # # def main():
# # # #     test()
# # # #
# # # #     pass
# # # # if __name__ == '__main__':
# # # #     main()
# #
# #
# # import torch
# # import numpy as np
# # import torch.nn.functional as F
# # from scipy.signal import get_window
# # from librosa.util import pad_center, tiny
# # from BIGDATA_SNR_PL_LSTM_DATASET.util import window_sumsquare
# #
# #
# # class STFT(torch.nn.Module):
# #     def __init__(self, filter_length=1024, hop_length=512, win_length=None,
# #                  window='hann'):
# #         """
# #         This module implements an STFT using 1D convolution and 1D transpose convolutions.
# #         This is a bit tricky so there are some cases that probably won't work as working
# #         out the same sizes before and after in all overlap add setups is tough. Right now,
# #         this code should work with hop lengths that are half the filter length (50% overlap
# #         between frames).
# #
# #         Keyword Arguments:
# #             filter_length {int} -- Length of filters used (default: {1024})
# #             hop_length {int} -- Hop length of STFT (restrict to 50% overlap between frames) (default: {512})
# #             win_length {[type]} -- Length of the window function applied to each frame (if not specified, it
# #                 equals the filter length). (default: {None})
# #             window {str} -- Type of window to use (options are bartlett, hann, hamming, blackman, blackmanharris)
# #                 (default: {'hann'})
# #         """
# #         super(STFT, self).__init__()
# #         self.filter_length = filter_length
# #         self.hop_length = hop_length
# #         self.win_length = win_length if win_length else filter_length
# #         self.window = window
# #         self.forward_transform = None
# #         self.pad_amount = int(self.filter_length / 2)
# #         scale = self.filter_length / self.hop_length
# #         fourier_basis = np.fft.fft(np.eye(self.filter_length))
# #
# #         cutoff = int((self.filter_length / 2 + 1))
# #         fourier_basis = np.vstack([np.real(fourier_basis[:cutoff, :]),
# #                                    np.imag(fourier_basis[:cutoff, :])])
# #         forward_basis = torch.FloatTensor(fourier_basis[:, None, :])
# #         inverse_basis = torch.FloatTensor(
# #             np.linalg.pinv(scale * fourier_basis).T[:, None, :])
# #
# #         assert (filter_length >= self.win_length)
# #         # get window and zero center pad it to filter_length
# #         fft_window = get_window(window, self.win_length, fftbins=True)
# #         fft_window = pad_center(fft_window, filter_length)
# #         fft_window = torch.from_numpy(fft_window).float()
# #
# #         # window the bases
# #         forward_basis *= fft_window
# #         inverse_basis *= fft_window
# #
# #         self.register_buffer('forward_basis', forward_basis.float())
# #         self.register_buffer('inverse_basis', inverse_basis.float())
# #
# #     def transform(self, input_data):
# #         """Take input data (audio) to STFT domain.
# #
# #         Arguments:
# #             input_data {tensor} -- Tensor of floats, with shape (num_batch, num_samples)
# #
# #         Returns:
# #             magnitude {tensor} -- Magnitude of STFT with shape (num_batch,
# #                 num_frequencies, num_frames)
# #             phase {tensor} -- Phase of STFT with shape (num_batch,
# #                 num_frequencies, num_frames)
# #         """
# #         num_batches = input_data.shape[0]
# #         num_samples = input_data.shape[-1]
# #
# #         self.num_samples = num_samples
# #
# #         # similar to librosa, reflect-pad the input
# #         input_data = input_data.view(num_batches, 1, num_samples)
# #
# #         input_data = F.pad(
# #             input_data.unsqueeze(1),
# #             (self.pad_amount, self.pad_amount, 0, 0),
# #             mode='reflect')
# #         input_data = input_data.squeeze(1)
# #
# #         forward_transform = F.conv1d(
# #             input_data,
# #             self.forward_basis,
# #             stride=self.hop_length,
# #             padding=0)
# #
# #         cutoff = int((self.filter_length / 2) + 1)
# #         real_part = forward_transform[:, :cutoff, :]
# #         imag_part = forward_transform[:, cutoff:, :]
# #
# #         magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2)
# #         phase = torch.atan2(imag_part.data, real_part.data)
# #
# #         return magnitude, phase
# #
# #     def inverse(self, magnitude, phase):
# #         """Call the inverse STFT (iSTFT), given magnitude and phase tensors produced
# #         by the ```transform``` function.
# #
# #         Arguments:
# #             magnitude {tensor} -- Magnitude of STFT with shape (num_batch,
# #                 num_frequencies, num_frames)
# #             phase {tensor} -- Phase of STFT with shape (num_batch,
# #                 num_frequencies, num_frames)
# #
# #         Returns:
# #             inverse_transform {tensor} -- Reconstructed audio given magnitude and phase. Of
# #                 shape (num_batch, num_samples)
# #         """
# #         recombine_magnitude_phase = torch.cat(
# #             [magnitude * torch.cos(phase), magnitude * torch.sin(phase)], dim=1)
# #
# #         inverse_transform = F.conv_transpose1d(
# #             recombine_magnitude_phase,
# #             self.inverse_basis,
# #             stride=self.hop_length,
# #             padding=0)
# #
# #         if self.window is not None:
# #             window_sum = window_sumsquare(
# #                 self.window, magnitude.size(-1), hop_length=self.hop_length,
# #                 win_length=self.win_length, n_fft=self.filter_length,
# #                 dtype=np.float32)
# #             # remove modulation effects
# #             approx_nonzero_indices = torch.from_numpy(
# #                 np.where(window_sum > tiny(window_sum))[0])
# #             window_sum = torch.from_numpy(window_sum).to(inverse_transform.device)
# #             inverse_transform[:, :, approx_nonzero_indices] /= window_sum[approx_nonzero_indices]
# #
# #             # scale by hop ratio
# #             inverse_transform *= float(self.filter_length) / self.hop_length
# #
# #         inverse_transform = inverse_transform[..., self.pad_amount:]
# #         inverse_transform = inverse_transform[..., :self.num_samples]
# #         inverse_transform = inverse_transform.squeeze(1)
# #
# #         return inverse_transform
# #
# #     def forward(self, input_data):
# #         """Take input data (audio) to STFT domain and then back to audio.
# #
# #         Arguments:
# #             input_data {tensor} -- Tensor of floats, with shape (num_batch, num_samples)
# #
# #         Returns:
# #             reconstruction {tensor} -- Reconstructed audio given magnitude and phase. Of
# #                 shape (num_batch, num_samples)
# #         """
# #         self.magnitude, self.phase = self.transform(input_data)
# #         reconstruction = self.inverse(self.magnitude, self.phase)
# #         return reconstruction
#
#
import pickle

import torch
import inspect

from torchvision import models
import numpy as np
from BIGDATA_SNR_PL_LSTM_DATASET.config import MEAN_STD_PATH, TR_PATH, MEAN_BATCH_SIZE, CUDA_ID, WIN_LEN, WIN_OFFSET
from BIGDATA_SNR_PL_LSTM_DATASET.gen_feat import FeatureDataCreator, SpeechMixDataset, BatchDataLoader
from BIGDATA_SNR_PL_LSTM_DATASET.gpu_mem_track import  MemTracker
from BIGDATA_SNR_PL_LSTM_DATASET.stft_istft import STFT
from BIGDATA_SNR_PL_LSTM_DATASET.util import _cal_log

device = torch.device("cuda:" + str(CUDA_ID[0]) if torch.cuda.is_available() else "cpu")


frame = inspect.currentframe()          # define a frame to track
gpu_tracker = MemTracker(frame)         # define a GPU tracker

gpu_tracker.track()                     # run function between the code line where uses GPU
cnn = models.vgg19(pretrained=True).features.to(device).eval()
gpu_tracker.track()                     # run function between the code line where uses GPU

dummy_tensor_1 = torch.randn(30, 3, 512, 512).float().to(device)  # 30*3*512*512*4/1000/1000 = 94.37M
dummy_tensor_2 = torch.randn(40, 3, 512, 512).float().to(device)  # 40*3*512*512*4/1000/1000 = 125.82M
dummy_tensor_3 = torch.randn(60, 3, 512, 512).float().to(device)  # 60*3*512*512*4/1000/1000 = 188.74M

gpu_tracker.track()

dummy_tensor_4 = torch.randn(120, 3, 512, 512).float().to(device)  # 120*3*512*512*4/1000/1000 = 377.48M
dummy_tensor_5 = torch.randn(80, 3, 512, 512).float().to(device)  # 80*3*512*512*4/1000/1000 = 251.64M

gpu_tracker.track()

dummy_tensor_4 = dummy_tensor_4.cpu()
dummy_tensor_2 = dummy_tensor_2.cpu()
torch.cuda.empty_cache()

gpu_tracker.track()

# from __future__ import print_function
# import os
#
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
# os.environ['GPU_DEBUG'] = '2'
#
# import argparse
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torchvision import datasets, transforms
#
#
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
#         self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#         self.conv2_drop = nn.Dropout2d()
#         self.fc1 = nn.Linear(320, 50)
#         self.fc2 = nn.Linear(50, 10)
#
#     def forward(self, x):
#         x = F.relu(F.max_pool2d(self.conv1(x), 2))
#         x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
#         x = x.view(-1, 320)
#         x = F.relu(self.fc1(x))
#         x = F.dropout(x, training=self.training)
#         x = self.fc2(x)
#         return F.log_softmax(x, dim=1)
#
#
# import sys
# from BIGDATA_SNR_PL_LSTM_DATASET.gpu_profile import gpu_profile
#
#
# def train(args, model, device, train_loader, optimizer, epoch):
#     model.train()
#     for batch_idx, (data, target) in enumerate(train_loader):
#         data, target = data.to(device), target.to(device)
#         optimizer.zero_grad()
#         output = model(data)
#         loss = F.nll_loss(output, target)
#         loss.backward()
#         optimizer.step()
#         if batch_idx % args.log_interval == 0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch, batch_idx * len(data), len(train_loader.dataset),
#                        100. * batch_idx / len(train_loader), loss.item()))
#
#         # gpu_profile(frame=sys._getframe(), event='line', arg=None)
#
#
# def test(args, model, device, test_loader):
#     model.eval()
#     test_loss = 0
#     correct = 0
#     with torch.no_grad():
#         for data, target in test_loader:
#             data, target = data.to(device), target.to(device)
#             output = model(data)
#             test_loss += F.nll_loss(output, target, size_average=False).item()  # sum up batch loss
#             pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
#             correct += pred.eq(target.view_as(pred)).sum().item()
#
#     test_loss /= len(test_loader.dataset)
#     print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#         test_loss, correct, len(test_loader.dataset),
#         100. * correct / len(test_loader.dataset)))
#
#
# def main():
#     # Training settings
#     parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
#     parser.add_argument('--batch-size', type=int, default=64, metavar='N',
#                         help='input batch size for training (default: 64)')
#     parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
#                         help='input batch size for testing (default: 1000)')
#     parser.add_argument('--epochs', type=int, default=10, metavar='N',
#                         help='number of epochs to train (default: 10)')
#     parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
#                         help='learning rate (default: 0.01)')
#     parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
#                         help='SGD momentum (default: 0.5)')
#     parser.add_argument('--no-cuda', action='store_true', default=False,
#                         help='disables CUDA training')
#     parser.add_argument('--seed', type=int, default=1, metavar='S',
#                         help='random seed (default: 1)')
#     parser.add_argument('--log-interval', type=int, default=10, metavar='N',
#                         help='how many batches to wait before logging training status')
#     args = parser.parse_args()
#     use_cuda = not args.no_cuda and torch.cuda.is_available()
#
#     torch.manual_seed(args.seed)
#
#     device = torch.device("cuda" if use_cuda else "cpu")
#
#     kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
#     train_loader = torch.utils.data.DataLoader(
#         datasets.MNIST('./data', train=True, download=True,
#                        transform=transforms.Compose([
#                            transforms.ToTensor(),
#                            transforms.Normalize((0.1307,), (0.3081,))
#                        ])),
#         batch_size=args.batch_size, shuffle=True, **kwargs)
#     test_loader = torch.utils.data.DataLoader(
#         datasets.MNIST('./data', train=False, transform=transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.1307,), (0.3081,))
#         ])),
#         batch_size=args.test_batch_size, shuffle=True, **kwargs)
#
#     model = Net().to(device)
#     optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
#
#     for epoch in range(1, args.epochs + 1):
#         train(args, model, device, train_loader, optimizer, epoch)
#         test(args, model, device, test_loader)
#
#
# if __name__ == '__main__':
#     sys.settrace(gpu_profile)
#     main()


# import soundfile as sf
# from pystoi import stoi
#
# clean, fs = sf.read('path/to/clean/audio')
# denoised, fs = sf.read('path/to/denoised/audio')
#
# # Clean and den should have the same length, and be 1D
# d = stoi(clean, denoised, fs, extended=False)

STFT = STFT(WIN_LEN, WIN_OFFSET).cuda(CUDA_ID[0])


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


    tr_total_con = np.concatenate(tr_total, axis=1)  # 横向拼接
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