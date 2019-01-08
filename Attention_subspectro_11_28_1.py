import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch.autograd.variable as variable
from torch.autograd import Variable as variable
import torch.optim as optim
import numpy as np 

import torchvision
from torchvision import datasets, transforms, utils

import matplotlib.pyplot as plt

import datetime
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
data_train = torchvision.datasets.ImageFolder('./sox_single_data/train/', transform = transforms.Compose([transforms.Scale((128, 128)), transforms.ToTensor()]))
data_test = torchvision.datasets.ImageFolder('./sox_single_data/test/', transform = transforms.Compose([transforms.Scale((128, 128)), transforms.ToTensor()]))

# 数据装载
data_loader_train = torch.utils.data.DataLoader(dataset = data_train, batch_size=32, shuffle=True)
data_loader_test = torch.utils.data.DataLoader(dataset = data_test, batch_size=32, shuffle=True)

# 搭建模型
class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.base_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size = 5, stride = 1, padding = 2, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(stride = 2, kernel_size = 2),

            nn.Conv2d(64, 128, kernel_size = 5, stride = 1, padding = 2, bias = False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(stride = 2, kernel_size = 2)
        )

        self.att_conv = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size = 5, stride = 1, padding = 2, bias = False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(stride = 2, kernel_size = 2),

            nn.Conv2d(256, 512, kernel_size = 5, stride = 1, padding = 2, bias = False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(stride = 2, kernel_size = 2)
        )

        # self.avg_layer = nn.AvgPool2d(16,16)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.att_fc = nn.Sequential(
            nn.Linear(512, 512 // 16),
            nn.ReLU(inplace=True),
            nn.Linear(512 // 16, 512),
            nn.Sigmoid()
        )

        self.fc = nn.Sequential(
            nn.Linear(32768, 4096),
            nn.ReLU(),

            nn.Linear(4096, 512),
            nn.ReLU(),

            nn.Linear(512, 10)
        )

    def forward(self, x, num_spectrogram):
        in_size = x.size(0)

        # 划分子频段
        x_sub = x.chunk(num_spectrogram, dim = -2)
        
        '''
            基本模块
        '''
        # 全局特征卷积
        global_feature = self.base_conv(x)

        # 局部特征卷积
        local_feature = []
        for i in range(num_spectrogram):
            local_feature.append(self.base_conv(x_sub[i]))
        
        # 局部特征拼接
        i = 1
        local_cat = local_feature[0]
        while i < num_spectrogram:
            local_cat = torch.cat((local_cat, local_feature[i]), 2)
            i += 1
        
        base_feature = global_feature + local_cat


        '''
            注意力模块
        '''
        # 全局特征卷积
        att_global_feature = self.att_conv(base_feature)

        # 划分子频段特征
        att_sub = base_feature.chunk(num_spectrogram, dim = -2)

        att_local_feature = []
        for i in range(num_spectrogram):
            att_local_feature.append(self.att_conv(att_sub[i]))

        att_global_p = att_global_feature.chunk(num_spectrogram, dim = -2)

        re_cat = []
        for i in range(num_spectrogram):
            for j in range(num_spectrogram):
                temp = self.avg_pool(torch.cat((att_global_p[i], att_local_feature[j]), 2)).view(in_size, 512)
                re_cat.append((self.att_fc(temp)).view(in_size, 512, 1, 1))

        A = []
        for i in range(num_spectrogram):
            temp = att_global_p[0] * re_cat[(i * num_spectrogram)]
            for j in range(1, num_spectrogram):
                temp = torch.cat((temp, (att_global_p[j] * re_cat[(i * num_spectrogram + j)])), 2)
            A.append(temp)
        output = att_global_feature
        for i in range(num_spectrogram):
            output = output + A[i]
        
        output = output.view(in_size, -1)
        output = self.fc(output)
        output = F.log_softmax(output)
        # return output
        return output

is_cuda = torch.cuda.is_available()
# is_cuda = False
if is_cuda:
    model = Model().cuda()
else:
    model = Model()

# 损失函数
cost = torch.nn.CrossEntropyLoss()
# 参数优化
optimizer = optim.Adam(model.parameters())

# 查看模型的完整结构
# print(model)

# 进行模型训练和参数优化

n_epochs = 20
loss_plt = []
epoch_plt = []
train_acc_plt = []
test_acc_plt = []
start_time = datetime.datetime.now()
for epoch in range(n_epochs):
    running_loss = 0.0
    running_correct = 0.0
    print('Epoch {} / {}'.format(epoch, n_epochs))
    print('-' * 10)
    i = 0
    
    for data in data_loader_train:
        #print('i:{}'.format(i))
        x_train, y_train = data
        if is_cuda:
            x_train, y_train = variable(x_train).cuda(), variable(y_train).cuda()
        else:
            x_train, y_train = variable(x_train), variable(y_train)
        # print(x_train.shape)
        outputs = model(x_train, 8)
        # print(outputs.type())
        # _ -- 最大值， pred -- 最大值序号
        _, pred = torch.max(outputs.data, 1)
        # print(pred.type())

        optimizer.zero_grad()
        
        loss = cost(outputs, y_train)
        # print(loss.type())

        loss.backward()
        optimizer.step()

        running_loss += loss.data
        # print(running_loss.type())

        # 返回输入向量input所有元素的和
        running_correct += torch.sum(pred == y_train.data)
        # print(running_correct.type())

        batch_correct = torch.sum(pred == y_train.data)
        if is_cuda:
            batch_correct = batch_correct.type(torch.cuda.FloatTensor)
        else:
            batch_correct = batch_correct.type(torch.FloatTensor)
        # print(batch_correct.type())
        i += 1
        if i % 10 == 0:
            print('i:{} loss: {} correct: {}'.format(i, loss.data, batch_correct / 32))
    
    train_endtime = datetime.datetime.now()
    testing_correct = 0.0
    i = 0
    for data in data_loader_test:
        #print('i:{}: Testing data:{}, the shape is:{}, the type is:{}'.format(i, data, len(data), type(data)))
        x_test, y_test = data
        if is_cuda:
            x_test, y_test = variable(x_test).cuda(), variable(y_test).cuda()
        else:
            x_test, y_test = variable(x_test), variable(y_test)
        outputs = model(x_test, 8)
        _, pred = torch.max(outputs.data, 1)
        testing_correct += torch.sum(pred == y_test.data)

        batch_correct = torch.sum(pred == y_test.data)
        if is_cuda:
            batch_correct = batch_correct.type(torch.cuda.FloatTensor)
        else:
            batch_correct = batch_correct.type(torch.FloatTensor)

        i += 1
        if i % 10 == 0:
            print('i:{} loss: {} correct: {}'.format(i, loss.data, batch_correct / 32))
        
    test_endtime = datetime.datetime.now()
    # print('training time is: {}'.format(train_endtime - start_time))
    # print('testing time is: {}'.format(test_endtime - train_endtime))
    # print('all time is: {}'.format(test_endtime - start_time))
    # print("running_correct: ", running_correct, type(running_correct), running_correct.type())
    # print("len(data_train): ", len(data_train), type(len(data_train)))
    # print("testing_correct: ", testing_correct, type(testing_correct), testing_correct.type())
    # print("len(data_test): ", len(data_test), type(len(data_test)))
    running_correct = running_correct.type(torch.cuda.FloatTensor)
    testing_correct = testing_correct.type(torch.cuda.FloatTensor)
    print('Loss is:{}, Train Accuracy is:{:.6f}%, Test Accuracy is:{:.6f}%'.format(running_loss / len(data_train), 100.0 * running_correct/len(data_train), 100.0 * testing_correct / len(data_test)))

    temp_loss = running_loss / len(data_train)
    temp_train_acc = 100 * running_correct/len(data_train)
    temp_test_acc = 100 * testing_correct / len(data_test)
    loss_plt.append(temp_loss)
    train_acc_plt.append(temp_train_acc)
    test_acc_plt.append(temp_test_acc)
    epoch_plt.append(epoch)

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(epoch_plt, loss_plt, 'r', label = 'loss')
ax1.legend(loc = 'upper left')
ax2 = ax1.twinx()
ax2.plot(epoch_plt, train_acc_plt, 'g', label = 'train Accuracy')
ax2.plot(epoch_plt, test_acc_plt, 'b', label = 'test Accuracy')

plt.legend(loc = 4)
plt.savefig('./result.jpg')
plt.show()
