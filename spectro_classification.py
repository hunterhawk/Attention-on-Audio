import torch
import torch.nn as nn
import torch.autograd.variable as variable
import torch.optim as optim
import numpy as np 

import torchvision
from torchvision import datasets, transforms, utils

import matplotlib.pyplot as plt

import datetime

data_train = torchvision.datasets.ImageFolder('./sox_single_data/train/', transform = transforms.Compose([transforms.Scale((64, 64)), transforms.ToTensor()]))
data_test = torchvision.datasets.ImageFolder('./sox_single_data/test/', transform = transforms.Compose([transforms.Scale((64, 64)), transforms.ToTensor()]))

# 数据装载
data_loader_train = torch.utils.data.DataLoader(dataset = data_train, batch_size=64, shuffle=True)
data_loader_test = torch.utils.data.DataLoader(dataset = data_test, batch_size=64, shuffle=True)

# 选取一个批次的数据进行预览
'''
images, labels = next(iter(data_loader_train))
img = torchvision.utils.make_grid(images)

img = img.numpy().transpose(1, 2, 0)

print([labels[i] for i in range(64)])
plt.imshow(img)

plt.show()
'''

# 搭建模型
'''
    卷积层：nn.Conv2d类方法搭建
    激活层：nn.ReLU()类方法搭建
    池化层：nn.MaxPool2d类方法搭建
    全连接层：nn.Linear类方法搭建
'''
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            # 输入通道数、输出通道数、卷积核大小、移动步长、padding值
            # 0 -- 不进行边界像素的填充
            # 1 -- 进行边界像素的填充
            nn.Conv2d(3, 64, kernel_size = 5, stride = 1, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(stride = 2, kernel_size = 2),

            nn.Conv2d(64, 128, kernel_size = 5, stride = 1, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(stride = 2, kernel_size = 2), # 池化窗口大小、移动步长、Padding值

            nn.Conv2d(128, 256, kernel_size = 5, stride = 1, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(stride = 2, kernel_size = 2),

            nn.Conv2d(256, 512, kernel_size = 5, stride = 1, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(stride = 2, kernel_size = 2)
        )

        self.dense = nn.Sequential(
            nn.Linear(2*2*512, 4096),
            nn.ReLU(),
            nn.Dropout(p = 0.5),

            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p = 0.5),

            nn.Linear(4096, 10)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1, 2 * 2 * 512)
        x = self.dense(x)
        return x

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

n_epochs = 50
'''
print(type(data_loader_train))
print(type(data_train))
print(len(data_train))
print(len(data_test))
'''
loss_plt = []
epoch_plt = []
train_acc_plt = []
test_acc_plt = []
start_time = datetime.datetime.now()
for epoch in range(n_epochs):
    running_loss = 0.0
    running_correct = 0
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
        outputs = model(x_train)
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
            print('i:{} loss: {} correct: {}'.format(i, loss.data, batch_correct / 64))
        '''
        if i == 3:
            break
        '''
    train_endtime = datetime.datetime.now()
    testing_correct = 0
    i = 0
    for data in data_loader_test:
        #print('i:{}: Testing data:{}, the shape is:{}, the type is:{}'.format(i, data, len(data), type(data)))
        x_test, y_test = data
        if is_cuda:
            x_test, y_test = variable(x_test).cuda(), variable(y_test).cuda()
        else:
            x_test, y_test = variable(x_test), variable(y_test)
        outputs = model(x_test)
        _, pred = torch.max(outputs.data, 1)
        testing_correct += torch.sum(pred == y_test.data)

        batch_correct = torch.sum(pred == y_test.data)
        if is_cuda:
            batch_correct = batch_correct.type(torch.cuda.FloatTensor)
        else:
            batch_correct = batch_correct.type(torch.FloatTensor)

        i += 1
        if i % 10 == 0:
            print('i:{} loss: {} correct: {}'.format(i, loss.data, batch_correct / 64))
        '''
        if i == 3:
            break
        '''
    '''    
    cc = 100 * running_correct / len(data_train)
    bb = 100 * testing_correct / len(data_test)
    '''
    test_endtime = datetime.datetime.now()
    print('training time is: {}'.format(train_endtime - start_time))
    print('testing time is: {}'.format(test_endtime - train_endtime))
    print('all time is: {}'.format(test_endtime - start_time))
    print('Loss is:{:.4f}, Train Accuracy is:{:.4f}%, Test Accuracy is:{:.4f}%'.format(running_loss / len(data_train), 100 * running_correct/len(data_train), 100 * testing_correct / len(data_test)))

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

plt.legend(loc = 'upper right')
plt.savefig('./result.jpg')
plt.show()