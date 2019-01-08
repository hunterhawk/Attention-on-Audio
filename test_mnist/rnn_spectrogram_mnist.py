import sys
sys.path.append('..')

import torch
from torch.autograd import Variable
from torch import nn
from torch.utils.data import DataLoader


from torchvision import transforms as tfs
from torchvision.datasets import MNIST

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 定义数据
data_tf = tfs.Compose([
    tfs.ToTensor(),
    tfs.Normalize([0.5], [0.5]) # 标准化
])

train_set = MNIST('./data', train=True, transform=data_tf)
test_set = MNIST('./data', train=False, transform=data_tf)

train_data = DataLoader(train_set, 64, True, num_workers=4)
test_data = DataLoader(test_set, 128, False, num_workers=4)


# 定义模型
class rnn_classify(nn.Module):
    def __init__(self, in_feature=28, hidden_feature=100, num_class=10, num_layers=2):
        super(rnn_classify, self).__init__()
        self.rnn = nn.LSTM(in_feature, hidden_feature, num_layers)  # 使用两层 lstm
        self.classifier = nn.Linear(hidden_feature, num_class)  # 将最后一个 rnn 的输出使用全连接得到最后的分类结果

    def forward(self, x):
        '''
        x 大小为 (batch, 1, 28, 28)，所以我们需要将其转换成 RNN 的输入形式，即 (28, batch, 28)
        '''
        x = x.squeeze()  # 去掉 (batch, 1, 28, 28) 中的 1，变成 (batch, 28, 28)
        x = x.permute(2, 0, 1)  # 将最后一维放到第一维，变成 (28, batch, 28)
        out, _ = self.rnn(x)  # 使用默认的隐藏状态，得到的 out 是 (28, batch, hidden_feature)
        out = out[-1, :, :]  # 取序列中的最后一个，大小是 (batch, hidden_feature)
        out = self.classifier(out)  # 得到分类结果
        return out

net = rnn_classify()
criterion = nn.CrossEntropyLoss()

optimzier = torch.optim.Adadelta(net.parameters(), 1e-1)
# 开始训练
from utils import train
train(net, train_data, test_data, 10, optimzier, criterion)