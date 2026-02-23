'''


'''
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from PIL import Image
import pandas as pd
import random
from torch import optim
from torch.optim import lr_scheduler
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import torchvision
import torchvision.datasets as datasets
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import time

# 设置plot中文字体
font = {'family': 'MicroSoft YaHei',
        'weight': 'bold',
        'size': '10'}
matplotlib.rc("font", **font)


# 设定超参数及常数
learning_rate = 0.0005     #学习率
batch_size = 1           #批处理量
epochs_num = 50            #训练迭代次数
use_gpu = 1                #CUDA GPU加速  1:使用  0:禁用
is_train = 0               #训练模型  1:重新训练     0:加载现有模型

ROOT_DIR = './dataset/'
TRAIN_DIR = 'ETH3x100/'
VAL_DIR = 'ETH3x100/'
TEST_DIR = 'ETH3x100/'
TRAIN_ANNO = 'Species_train_annotation.csv'
VAL_ANNO = 'Species_val_annotation.csv'
TEST_ANNO = 'Species_test_annotation.csv'



class MyDataset():

    def __init__(self, root_dir, annotations_file, transform=None):

        self.root_dir = root_dir
        self.annotations_file = annotations_file
        self.transform = transform

        if not os.path.isfile(self.annotations_file):
            print(self.annotations_file + 'does not exist!')
        self.file_info = pd.read_csv(annotations_file, index_col=0)
        self.size = len(self.file_info)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        image_path = self.file_info['path'][idx]
        if not os.path.isfile(image_path):
            print(image_path + '  does not exist!')
            return None

        image = Image.open(image_path).convert('RGB')
        label_species = int(self.file_info.iloc[idx]['species'])

        sample = {'image': image, 'species': label_species}
        if self.transform:
            sample['image'] = self.transform(image)
        return sample

# 原图像是256x256的
train_transforms = transforms.Compose([transforms.Resize((68, 68)),
                                        transforms.RandomHorizontalFlip(p=0.5),   # 水平翻转
                                        transforms.RandomVerticalFlip(p=0.5),     # 上下翻转
                                        transforms.ToTensor(),
                                    ])
val_transforms = transforms.Compose([transforms.Resize((68, 68)),
                                     transforms.ToTensor()
                                     ])
test_transforms = transforms.Compose([transforms.Resize((68, 68)),
                                     transforms.ToTensor()
                                     ])


train_dataset = MyDataset(root_dir= ROOT_DIR + TRAIN_DIR,
                          annotations_file= TRAIN_ANNO,
                          transform=train_transforms)
val_dataset = MyDataset(root_dir= ROOT_DIR + VAL_DIR,
                         annotations_file= VAL_ANNO,
                         transform=val_transforms)
test_dataset = MyDataset(root_dir=ROOT_DIR + TEST_DIR,
                         annotations_file=TEST_ANNO,
                         transform=test_transforms)


train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size , shuffle=True)
val_loader = DataLoader(dataset=val_dataset)
test_loader = DataLoader(dataset=test_dataset)


data_loaders = {'train': train_loader, 'val': val_loader, 'test':test_loader}


# 初始化卷积神经网络
class ETH_Network(nn.Module):
    def __init__(self):
        super(ETH_Network, self).__init__()#3x68x68

        self.conv1 = nn.Conv2d(3,12,kernel_size = 5,padding=0)  # 卷积层12x64x64
        self.relu1 = nn.ReLU()                                  # 激活函数ReLU
        self.pool1 = nn.MaxPool2d(2,stride=2)                   # 最大池化层12x32x32

        self.conv2 = nn.Conv2d(12,14,kernel_size = 5,padding=0) # 卷积层14x28x28
        self.relu2 = nn.ReLU()                                  # 激活函数ReLU
        self.pool2 = nn.MaxPool2d(2,stride=2)                   # 最大池化层14x14x14

        self.conv3 = nn.Conv2d(14,8,kernel_size = 5,padding=0) # 卷积层16x10x10
        self.relu3 = nn.ReLU()                                  # 激活函数ReLU
        self.pool3 = nn.MaxPool2d(2,stride=2)                   # 最大池化层16x5x5


        self.fc4 = nn.Linear(5*5*8,3)                           # 全连接层
        self.softmax4 = nn.Softmax(dim=1)                       # Softmax层

    # 前向传播
    def forward(self, input1):
        x = self.conv1(input1)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        x = x.view(x.size()[0], -1)

        x = self.fc4(x)
        x = self.softmax4(x)
        return x

# 初始化神经网络
net = ETH_Network()
if use_gpu:           #CUDA GPU加速
    net = net.cuda()


if is_train:
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)  # 使用adam算法进行训练

    counter = []
    loss_history = []
    train_correct_history = []
    val_correct_history = []
    correct_cnt = 0
    counter_temp = 0
    # 多次迭代训练网络
    for epoch in range(0, epochs_num):
        net.train()
        for i, data in enumerate(data_loaders['train'],0):
            img = data['image']
            label = data['species']
            if use_gpu:   #CUDA GPU加速
                img, label = img.cuda(), label.cuda()

            optimizer.zero_grad()            # 清除网络状态
            output = net(img)                # 前向传播

            loss = criterion(output, label)  # 计算损失函数
            loss.backward()                  # 反向传播
            optimizer.step()                 # 参数更新

            _, predict = torch.max(output, 1)
            correct_cnt += (predict == label).sum()  # 预测值与实际值比较

            # 存储损失值与精度
        counter_temp =  epoch + 1
        counter.append(counter_temp)
        loss_history.append(loss.item())
        train_correct_history.append(correct_cnt.float().item()/(len(train_dataset)))
        correct_cnt = 0
        print("迭代次数 {}\n 当前训练集准确率 {}".format(epoch + 1, train_correct_history[-1]))

        net.eval()
        correct_val = 0
        with torch.no_grad():
            for val_data in data_loaders['val']:
                val_img = val_data['image']
                val_label = val_data['species']
                if use_gpu:  # CUDA GPU加速
                    val_img, val_label = val_img.cuda(), val_label.cuda()
                val_output = net(val_img)
                _, val_pred = torch.max(val_output, 1)
                correct_val += (val_pred == val_label).sum()
        val_acc = correct_val.float().item() / (len(val_dataset))
        val_correct_history.append(val_acc)
        print(" 当前验证集准确率 {}\n".format(val_correct_history[-1]))


    # 绘制损失函数与精度曲线
    plt.figure(figsize=(20, 10), dpi=80)
    plt.subplot(311)
    plt.plot(counter, loss_history)
    plt.ylabel('损失函数值')
    plt.title('损失函数曲线')
    plt.subplot(312)
    plt.plot(counter, train_correct_history)
    plt.ylabel('精确度')
    plt.title('训练集精确度曲线')
    plt.subplot(313)
    plt.plot(counter, val_correct_history)
    plt.ylabel('精确度')
    plt.title('验证集精确度曲线')
    plt.show()
    # 存储模型参数
    state = {'net':net.state_dict()}
    torch.save(net.state_dict(),'.\modelpara.pth')  # 这个模型参数保存了用300张图片训练一次后的模型参数

# 加载模型参数
if use_gpu:
    net.load_state_dict(torch.load('.\modelpara.pth'))
else:
    net.load_state_dict(torch.load('.\modelpara.pth', map_location='cpu'))


# 测试
correct = 0
for i,data in enumerate(test_loader, 0):
    img = data['image']
    label = data['species']
    if use_gpu:             #CUDA GPU加速
        img, label = img.cuda(), label.cuda()
    output = net(img)       # 前向传播
    _,predict = torch.max(output,1)
    correct += (predict==label).sum()  # 预测值与实际值比较

# 输出测试集准确率
print('测试集识别准确率= {:.2f}'.format(correct.cpu().numpy()/len(test_dataset)*100)+'%')
