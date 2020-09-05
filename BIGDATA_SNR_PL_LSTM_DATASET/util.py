import librosa
import numpy as np
import os
import re
import torch
from torch.autograd import Variable
import soundfile as sf

# def expandWindow(data, left, right):
#     data = data.detach().cpu().numpy()
#     sp = data.shape
#     idx = 0
#     exdata = np.zeros([sp[0], sp[1], sp[2] * (left + right + 1)])
#     for i in range(-left, right+1):
#         exdata[:, :, sp[2] * idx : sp[2] * (idx + 1)] = np.roll(data, shift=-i, axis=1)
#         idx = idx + 1
#     return Variable(torch.FloatTensor(exdata)).cuda(CUDA_ID[0])

# def context_window(data, left, right):
#     sp = data.data.shape
#     exdata = torch.zeros(sp[0], sp[1], sp[2] * (left + right + 1)).cuda(CUDA_ID[0])
#     for i in range(1, left + 1):
#         exdata[:, i:, sp[2] * (left - i) : sp[2] * (left - i + 1)] = data.data[:, :-i,:]
#     for i in range(1, right+1):
#         exdata[:, :-i, sp[2] * (left + i):sp[2]*(left+i+1)] = data.data[:, i:, :]
#     exdata[:, :, sp[2] * left : sp[2] * (left + 1)] = data.data
#     return Variable(exdata)
def read_list(list_file):
    f = open(list_file, "r")
    lines = f.readlines()
    list_sig = []
    for x in lines:
        list_sig.append(x[:-1])
    f.close()
    return list_sig
def gen_list(wav_dir, append):
    l = []
    lst = os.listdir(wav_dir)
    lst.sort()
    for f in lst:
        if re.search(append, f):
            l.append(f)
    return l

def write_log(file,name, train, validate):
    message = ''
    for m, val in enumerate(train):
        message += ' --TRerror%i=%.3f ' % (m, val.data.numpy())
    for m, val in enumerate(validate):
        message += ' --CVerror%i=%.3f ' % (m, val.data.numpy())
    file.write(name + ' ')
    file.write(message)
    file.write('/n')

def read_audio(path):
    (audio, _) = sf.read(path)
    return audio

def write_audio(path, audio, sample_rate):
    sf.write(file=path, data=audio, samplerate=sample_rate)

def _calc_alpha(SNR, speech, noise):
    alpha = np.sqrt(np.sum(speech **2.0) / (np.sum(noise **2.0) * (10.0 ** (SNR / 10.0))))
    return alpha
def _calc_stft(data):
    feature = librosa.stft(data, n_fft=320, win_length=320, hop_length=160)
    return feature
def _calc_irm(speech, noise):
    _s_square = np.square(np.absolute(speech))                            # np.square(x) ： 计算数组各元素的平方；np.absolute: 计算绝对值，np.absolute(a) 或者 np.abs(a)，对于非复数的数组，np.fabs 速度更快
    _n_square = np.square(np.absolute(noise))
    irm_mask = np.sqrt(np.divide(_s_square, (_s_square + _n_square)))     # np.divide(a, b): 两个数组元素一一对应相除.
    return irm_mask
def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)      # 这里若使用os.mkdir(fd)的话，则不能创建多级目录
def _cal_log(x):
    return torch.log(x + 1.0)



import numpy as np
from scipy.signal import get_window
import librosa.util as librosa_util

def window_sumsquare(window, n_frames, hop_length=200, win_length=800,
                     n_fft=800, dtype=np.float32, norm=None):
    """
    # from librosa 0.6
    Compute the sum-square envelope of a window function at a given hop length.
    This is used to estimate modulation effects induced by windowing
    observations in short-time fourier transforms.
    Parameters
    ----------
    window : string, tuple, number, callable, or list-like
        Window specification, as in `get_window`
    n_frames : int > 0
        The number of analysis frames
    hop_length : int > 0
        The number of samples to advance between frames
    win_length : [optional]
        The length of the window function.  By default, this matches `n_fft`.
    n_fft : int > 0
        The length of each analysis frame.
    dtype : np.dtype
        The data type of the output
    Returns
    -------
    wss : np.ndarray, shape=`(n_fft + hop_length * (n_frames - 1))`
        The sum-squared envelope of the window function
    """
    if win_length is None:
        win_length = n_fft

    n = n_fft + hop_length * (n_frames - 1)
    x = np.zeros(n, dtype=dtype)

    # Compute the squared window at the desired length
    win_sq = get_window(window, win_length, fftbins=True)
    win_sq = librosa_util.normalize(win_sq, norm=norm)**2
    win_sq = librosa_util.pad_center(win_sq, n_fft)

    # Fill the envelope
    for i in range(n_frames):
        sample = i * hop_length
        x[sample:min(n, sample + n_fft)] += win_sq[:max(0, min(n_fft, n - sample))]
    return x