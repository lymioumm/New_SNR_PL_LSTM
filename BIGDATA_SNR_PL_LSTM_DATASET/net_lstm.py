import torch.nn as nn
from torch.nn.utils.rnn import *
from torch.nn.functional import *
import torch.autograd.variable
from BIGDATA_SNR_PL_LSTM_DATASET.config import *
from torch.autograd.variable import *
import torch
import numpy as np
from BIGDATA_SNR_PL_LSTM_DATASET.stft_istft import STFT
# class NetLstm(nn.Module):
#     def __init__(self):
#         super(NetLstm, self).__init__()
#         self.lstm_input_size = 256 * 4
#         self.lstm_layers = 2
#         self.conv1 = nn.Conv2d(in_channels=2, out_channels=16, kernel_size=(3, 3), stride=(1, 2), padding=(1, 0))
#         self.conv1_relu = nn.ELU()
#         self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(1, 2), padding=(1, 0))
#         self.conv2_relu = nn.ELU()
#         self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 2), padding=(1, 0))
#         self.conv3_relu = nn.ELU()
#         self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 2), padding=(1, 0))
#         self.conv4_relu = nn.ELU()
#         self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 2), padding=(1, 0))
#         self.conv5_relu = nn.ELU()
#         self.lstm = nn.LSTM(input_size=self.lstm_input_size,
#                             hidden_size=self.lstm_input_size,
#                             num_layers=self.lstm_layers,
#                             batch_first=True)
#
#         self.conv5_t = nn.ConvTranspose2d(in_channels=512, out_channels=128, kernel_size=(3, 3), stride=(1, 2), padding=(1, 0))
#         self.conv5_t_relu = nn.ELU()
#         self.conv4_t = nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=(3, 3), stride=(1, 2), padding=(1, 0))
#         self.conv4_t_relu = nn.ELU()
#         self.conv3_t = nn.ConvTranspose2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 2), padding=(1, 0))
#         self.conv3_t_relu = nn.ELU()
#         self.conv2_t = nn.ConvTranspose2d(in_channels=64, out_channels=16, kernel_size=(3, 3),  stride=(1, 2), output_padding=(0, 1), padding=(1, 0))
#         self.conv2_t_relu = nn.ELU()
#         self.conv1_t = nn.ConvTranspose2d(in_channels=32, out_channels=2, kernel_size=(3, 3), stride=(1, 2), padding=(1, 0))
#         self.conv1_t_relu = nn.ELU()
#         self.hamming = Variable(torch.FloatTensor(np.hamming(WIN_LEN)).cuda(), requires_grad=False)
#
#         self.conv1_bn = nn.BatchNorm2d(16)
#         self.conv2_bn = nn.BatchNorm2d(32)
#         self.conv3_bn = nn.BatchNorm2d(64)
#         self.conv4_bn = nn.BatchNorm2d(128)
#         self.conv5_bn = nn.BatchNorm2d(256)
#
#         self.conv5_t_bn = nn.BatchNorm2d(128)
#         self.conv4_t_bn = nn.BatchNorm2d(64)
#         self.conv3_t_bn = nn.BatchNorm2d(32)
#         self.conv2_t_bn = nn.BatchNorm2d(16)
#         self.conv1_t_bn = nn.BatchNorm2d(2)
#
#
#         self.STFT = STFT(WIN_LEN, WIN_OFFSET).cuda()
#
#         self.conv1_result = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=(3, 3), stride=(1, 2), padding=(1, 0))
#         self.conv1_result_relu = nn.ELU()
#         self.conv1_result_bn = nn.BatchNorm2d(16)
#         self.conv1_result_t = nn.ConvTranspose2d(in_channels=16, out_channels=2, kernel_size=(3, 3), stride=(1, 2),
#                                           padding=(1, 0))
#
#
#         self.device = torch.device("cuda:" + str(CUDA_ID[0]) if torch.cuda.is_available() else "cpu")
#
#     def forward(self, input_data_c1,feature_):
#
#
#
#
#
#         input_data_c1 = input_data_c1.permute(0, 3, 1, 2)
#         e1 = self.conv1_relu(self.conv1_bn(self.conv1(input_data_c1)))
#         e2 = self.conv2_relu(self.conv2_bn(self.conv2(e1)))
#         e3 = self.conv3_relu(self.conv3_bn(self.conv3(e2)))
#         e4 = self.conv4_relu(self.conv4_bn(self.conv4(e3)))
#         e5 = self.conv5_relu(self.conv5_bn(self.conv5(e4)))
#
#         out_real = e5.contiguous().transpose(1, 2)
#         out_real = out_real.contiguous().view(out_real.size(0), out_real.size(1), -1)
#         lstm_out, _ = self.lstm(out_real)
#         lstm_out_real = lstm_out.contiguous().view(lstm_out.size(0), lstm_out.size(1), 256, 4)
#         lstm_out_real = lstm_out_real.contiguous().transpose(1, 2)
#
#         t5 = self.conv5_t_relu(self.conv5_t_bn(self.conv5_t(torch.cat((lstm_out_real, e5), dim=1))))
#         t4 = self.conv4_t_relu(self.conv4_t_bn(self.conv4_t(torch.cat((t5, e4), dim=1))))
#         t3 = self.conv3_t_relu(self.conv3_t_bn(self.conv3_t(torch.cat((t4, e3), dim=1))))
#         t2 = self.conv2_t_relu(self.conv2_t_bn(self.conv2_t(torch.cat((t3, e2), dim=1))))
#         t1 = self.conv1_t_relu(self.conv1_t_bn(self.conv1_t(torch.cat((t2, e1), dim=1))))
#
#         result1 = self.conv1_result_relu(self.conv1_result_bn(self.conv1_result(torch.cat((input_data_c1, t1), dim=1))))
#         result2 = self.conv1_result_t(result1)
#
#         out = torch.squeeze(result2, 1)
#         return [out]
#
#
#
#     def init_lstm1(self, batch_size):
#         return (Variable(
#             torch.zeros(self.lstm_layers, batch_size, self.lstm_input_size)).cuda(CUDA_ID[0]),
#                 Variable(torch.zeros(self.lstm_layers, batch_size, self.lstm_input_size)).cuda(CUDA_ID[0]))



class NetLstm(nn.Module):
    def __init__(self):


        super().__init__()
        self.input_size = 161  # input_size：161
        self.hidden_size = 382  # hidden_size：192
        self.num_layers = 3  # num_layers：3
        self.batch_size = TT_BATCH_SIZE  # batch_size_lstm：128    训练以及验证batch：32， 测试batch：1
        self.rnn = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)  # LSTM（161， 192， 3）
        self.fc = nn.Linear(self.hidden_size, 161)  # output_dim：161

        self.device = torch.device("cuda:" + str(CUDA_ID[0]) if torch.cuda.is_available() else "cpu")

    def forward(self, input_data_c1):
        h_0 = c_0 = torch.rand(self.num_layers, self.batch_size, self.hidden_size).to(self.device)
        rnn_o, (h_1, c_1) = self.rnn(input_data_c1, (h_0, c_0))
        # out_pad, out_len = rnn_utils.pad_packed_sequence(rnn_o, batch_first=True)
        fc_o = self.fc(rnn_o)

        fc_o_sig = torch.sigmoid(fc_o)
        return fc_o_sig



    def init_lstm1(self, batch_size):
        return (Variable(
            torch.zeros(self.lstm_layers, batch_size, self.lstm_input_size)).cuda(CUDA_ID[0]),
                Variable(torch.zeros(self.lstm_layers, batch_size, self.lstm_input_size)).cuda(CUDA_ID[0]))
