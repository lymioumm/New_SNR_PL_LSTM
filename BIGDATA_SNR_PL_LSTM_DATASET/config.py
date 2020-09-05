
TRAIN = True
DECODE = True
GPU_NUM = 2
mode = 'online'

CUDA_ID = [0] #[0, 1]
NUM_TRAIN_SPEECH = 300000
NUM_CV_SPEECH = 3000
SAMPLE_RATE = 16000
MAX_WAV_LEN = 7 * SAMPLE_RATE
WIN_LEN = 320
FFT_SIZE = WIN_LEN // 2 + 1
WIN_OFFSET = 160
num_basis = 256
SPLIT_TRAIN = 1 #20
LR = 0.001
DROPOUT = 0.0
start_halving_impr = 0.003
halving_factor = 0.7 #0.9
end_halving_impr = 0.01
WAV_LOOP_TIMES = 1000
max_epochs = 500000
min_epochs = 200000
# label dim

EPSILON = 1e-7

# INPUT_FEAT_SIZE = (FEAT_LEFT_SIDE + FEAT_RIGHT_SIDE + 1) * WIN_LEN

# for bark data
# BARK_FFT_WEIGHT_PATH = '../data/bark_fft_wt.mat'
import os
path = os.getcwd()
spath = path.split('/')

from BIGDATA_SNR_PL_LSTM_DATASET.get_IP import getip
import re
import platform
import socket
import datetime

sysstr = platform.system()
if sysstr == "Windows":
    ip_address = socket.gethostbyname(socket.gethostname())
    print("Windows @ " + ip_address)
elif sysstr == "Linux":
    ip_address = getip()
    print("Linux @ " + ip_address)
elif sysstr == "Darwin":
    ip_address = socket.gethostbyname(socket.gethostname())
    print("Mac @ " + ip_address)
else:
    print("Other System @ some ip")
ip = ip_address
idx = [i.start() for i in re.finditer('\.',ip)]
# print(idx)
IP = ip[idx[-1]+1:]
print(ip)
# IP = '244'
USE_CV = True
IP = '25'
if IP == '244':
    OUT_PATH = os.path.join('/mnt/hd0/lihao', spath[-2], spath[-1])
    TR_PATH = '/mnt/hd0/lihao/tr05_org_train/'
    TT_PATH = '/mnt/hd0/lihao/single_channel_speech_enhancement/DATA/TEST/'
elif IP == '243':
    OUT_PATH = os.path.join('/mnt/hd0/lihao', spath[-2], spath[-1])
    TR_PATH = '/mnt/hd0/lihao/tr05_org_train/'
    TT_PATH = '/mnt/hd0/lihao/single_channel_speech_enhancement/DATA/TEST/'
elif IP == '88':
    OUT_PATH = os.path.join('/mnt/raid2/userspace/lihao', spath[-2], spath[-1])
    DATA_PATH = '/mnt/raid2/userspace/lihao/SPEECH_ENHANCE_DATA'
    TR_PATH = DATA_PATH + '/tr/'
    CV_PATH = DATA_PATH + '/cv/'if USE_CV else ''
    TT_PATH = DATA_PATH + '/tt/'
elif IP == '253':
    OUT_PATH = os.path.join('/home/ZhangXueLiang/HeShuLin', spath[-2], spath[-1])
    DATA_PATH = '/mnt/raid2/userspace_hdd/zhonghua/heshulin/SPEECH_ENHANCE_DATA/'
    TR_PATH = DATA_PATH + '/tr/'
    CV_PATH = DATA_PATH + '/cv/'if USE_CV else ''
    TT_PATH = DATA_PATH + '/tt/'
elif IP == '61':
    workspace = './'
    OUT_PATH = os.path.join('/home/ZhangXueLiang/LiMiao/pycharmProjects/', spath[-2],
                            spath[-1])  # '/data02/limiao/pycharmProjects/New_SNR_PL_LSTM/BIGDATA_SNR_PL_LSTM'
    # DATA_PATH = '/data02/limiao/data/BIGDATA_SNR_PL_LSTM'
    DATA_PATH = '/data01/limiao/BIGDATA_SNR_PL_LSTM/'

    NOISE_PATH = DATA_PATH + '/noise'
    CLEAN_TR_PATH = DATA_PATH + '/train_data/'
    NOISE_TR_PATH = DATA_PATH + '/noise_train/'

    CLEAN_TT_PATH = DATA_PATH + '/test_data/'
    NOISE_TT_PATH = DATA_PATH + '/noise_test/'

    CLEAN_CV_PATH = DATA_PATH + '/valid_data/'
    NOISE_CV_PATH = DATA_PATH + '/noise_valid/'

    # 混合了-5dB、0dB和5dB信噪比的语音数据
    # /data01/limiao/BIGDATA_SNR_PL_LSTM/mix_train_wav
    TR_PATH = DATA_PATH + 'mix_train_wav/'  # 总数据是：1800 * 5 * 3 = 27000条
    CV_PATH = DATA_PATH + 'mix_valid_wav/' if USE_CV else ''   # 总数据是：200 * 7 * 3 = 4200条
    TT_PATH = DATA_PATH + 'mix_test_wav/'   # 总数据是：192 * 7 * 3 = 4032条

    # 对应上面一定信噪比的噪声语音数据，用于后面生成IRM mask
    NOISE_TRAIN_PATH = DATA_PATH + 'noise_train_wav/'   # 总数据是：1800 * 5 * 3 = 27000条
    NOISE_VALID_PATH = DATA_PATH + 'noise_valid_wav/'   # 总数据是：200 * 7 * 3 = 4200条
    NOISE_TEST_PATH = DATA_PATH + 'noise_test_wav/'     # 总数据是：192 * 7 * 3 = 4032条

elif IP == '25':
    workspace = './'
    OUT_PATH = os.path.join('/home/limiao/pycharmProjects/', spath[-2],
                            spath[-1])  # '/data02/limiao/pycharmProjects/New_SNR_PL_LSTM/BIGDATA_SNR_PL_LSTM'
    DATA_PATH = '/data02/limiao/data/BIGDATA_SNR_PL_LSTM'


    NOISE_PATH = DATA_PATH + '/noise'
    CLEAN_TR_PATH = DATA_PATH + '/train_data/'
    NOISE_TR_PATH = DATA_PATH + '/noise_train'

    CLEAN_TT_PATH = DATA_PATH + '/test_data/'
    NOISE_TT_PATH = DATA_PATH + '/noise_test/'

    CLEAN_CV_PATH = DATA_PATH + '/valid_data/'
    NOISE_CV_PATH = DATA_PATH + '/noise_valid/'

    # 混合了-5dB、0dB和5dB信噪比的语音数据
    TR_PATH = DATA_PATH + '/mix_train_wav/'  # '/data02/limiao/data/BIGDATA_SNR_PL_LSTM/mix_train_wav'
    CV_PATH = DATA_PATH + '/mix_valid_wav/' if USE_CV else ''
    TT_PATH = DATA_PATH + '/mix_test_wav/'

    # 对应上面一定信噪比的噪声语音数据，用于后面生成IRM mask
    # /data02/limiao/data/BIGDATA_SNR_PL_LSTM/noise_train_wav
    NOISE_TRAIN_PATH = DATA_PATH + '/noise_train_wav/'
    NOISE_VALID_PATH = DATA_PATH + '/noise_valid_wav/'
    NOISE_TEST_PATH = DATA_PATH + '/noise_test_wav/'


# LOG_FILE_NAME = os.path.join(OUT_PATH, 'Train_log', 'train.log')
LOG_FILE_NAME = os.path.join(DATA_PATH, 'Train_log', f'{datetime.datetime.now():%d-%b-%y-%H:%M:%S}-train.log')
LOG_FILE_NAME_TEST = os.path.join(DATA_PATH, 'Test_log', f'{datetime.datetime.now():%d-%b-%y-%H:%M:%S}-test.log')
if not os.path.exists(os.path.dirname(LOG_FILE_NAME_TEST)):
    os.makedirs(os.path.dirname(LOG_FILE_NAME_TEST))
# f'{datetime.datetime.now():%d-%b-%y-%H:%M:%S}-gpu_mem_track.txt'
# DATA_PATH = '/mnt/raid/data/lihao/dual_speech_enhancement/data/'
MODEL_NAME = 'nnet_model.pickle'
USE_MEAN_VARIANCE = False
NUM_CAL_MEAN_VARIANCE = 500 #used NUM_MV utterances to cal mean and std
PATH_MEAN_VARIANCE = os.path.join(OUT_PATH, '/mean_var.mat')

MODEL_PATH = os.path.join(OUT_PATH, 'model')                 # 存放所有的model
BEST_MODEL_PATH = os.path.join(OUT_PATH, 'best_model')       # 存放最好的一次model
GOOD_MODEL_PATHs = os.path.join(OUT_PATH, 'good_models')     # 存放只要cv_loss低于tr_loss的model
MEAN_STD_PATH = os.path.join(DATA_PATH, 'mean_std_pickle', 'mean_std.p')
SCORE_TXT = os.path.join(DATA_PATH, 'score_txt/')
if not os.path.exists(SCORE_TXT):
    os.makedirs(SCORE_TXT)
if not os.path.exists(os.path.dirname(MEAN_STD_PATH)):
    os.makedirs(os.path.dirname(MEAN_STD_PATH))
RESUME_MODEL = True
import os
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)
if not os.path.exists(GOOD_MODEL_PATHs):
    os.makedirs(GOOD_MODEL_PATHs)

# CV_PATH = DATA_PATH + '/cv/'
# TT_PATH = DATA_PATH + '/tt/'

TR_BATCH_SIZE = 128
TR_BATCH_SIZE = len(CUDA_ID) * TR_BATCH_SIZE
CV_BATCH_SIZE = TR_BATCH_SIZE
# TR_NIMI_BATCH = 200
TT_BATCH_SIZE = 1
MEAN_BATCH_SIZE = 1

TT_OUT_PATH = os.path.join(DATA_PATH, 'result_data/output/')
TT_OUT_PATH_FORDOWNLOAD = os.path.join(DATA_PATH, 'result_data/output_for_download')