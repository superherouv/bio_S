'''
这个代码用来将单个文件（也就是单张图片）转换成脉冲序列
'''
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import pickle
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from PIL import Image
import re

# 设置plot中文字体
font = {'family': 'MicroSoft YaHei',
        'weight': 'bold',
        'size': '10'}
matplotlib.rc("font", **font)


# 初始化卷积神经网络
class ETH_Network(nn.Module):
    def __init__(self):
        super(ETH_Network, self).__init__()  # 3x68x68

        self.conv1 = nn.Conv2d(3, 12, kernel_size=5, padding=0)  # 卷积层12x64x64
        self.relu1 = nn.ReLU()  # 激活函数ReLU
        self.pool1 = nn.MaxPool2d(2, stride=2)  # 最大池化层12x32x32

        self.conv2 = nn.Conv2d(12, 14, kernel_size=5, padding=0)  # 卷积层14x28x28
        self.relu2 = nn.ReLU()  # 激活函数ReLU
        self.pool2 = nn.MaxPool2d(2, stride=2)  # 最大池化层14x14x14

        self.conv3 = nn.Conv2d(14, 8, kernel_size=5, padding=0)  # 卷积层16x10x10
        self.relu3 = nn.ReLU()  # 激活函数ReLU
        self.pool3 = nn.MaxPool2d(2, stride=2)  # 最大池化层16x5x5

        self.fc4 = nn.Linear(5 * 5 * 8, 3)  # 全连接层
        self.softmax4 = nn.Softmax(dim=1)  # Softmax层

    # 前向传播
    def forward(self, input1):  # input1=(1,1,28,28)
        x = self.conv1(input1)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        print(x.shape)

        return x


class DelayPhaseEncoder:

    def __init__(self, n_rf, t_max=1., alpha=1., amp=1., freq=40., phi_0=0., delta_phi=None):
        self.n_rf = n_rf
        self.t_max = t_max
        self.alpha = alpha
        self.amp = amp
        self.freq = freq
        self.phi_o = phi_0
        if delta_phi is None:
            self.delta_phi = 2 * np.pi / n_rf
        else:
            self.delta_phi = delta_phi

    @property
    def n_rf(self):
        return self._n_rf

    @n_rf.setter
    def n_rf(self, new_nrf):
        assert isinstance(new_nrf, (int, float)), "'n_rf' must be of type int or float."
        self._n_rf = new_nrf

    @property
    def t_max(self):
        return self._t_max

    @t_max.setter
    def t_max(self, new_tmax):
        assert isinstance(new_tmax, float), "'t_max' must be of type float."
        self._t_max = new_tmax

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, new_alpha):
        assert isinstance(new_alpha, float), "'alpha' must be of type float."
        self._alpha = new_alpha

    @property
    def amp(self):
        return self._amp

    @amp.setter
    def amp(self, new_amp):
        assert isinstance(new_amp, float), "'amp' must be of type float."
        self._amp = new_amp

    @property
    def freq(self):
        return self._freq

    @freq.setter
    def freq(self, new_freq):
        assert isinstance(new_freq, (int, float)), "'freq' must be of type int or float."
        self._freq = new_freq

    @property
    def phi_0(self):
        return self._phi_0

    @phi_0.setter
    def phi_0(self, new_phi0):
        assert isinstance(new_phi0, float), "'phi_0' must be of type float"
        self._phi_0 = new_phi0

    @property
    def delta_phi(self):
        return self._delta_phi

    @delta_phi.setter
    def delta_phi(self, new_deltaphi):
        assert isinstance(new_deltaphi, float), "'delta_phi' must be of type float."
        self._delta_phi = new_deltaphi

    def encode(self, input_array):

        assert isinstance(input_array, np.ndarray), "'input_array' must be of type np.ndarray"
        assert input_array.ndim == 1, "'input_array' must be 1-d tensor"  # 秩或维度的个数必須為1
        assert input_array.shape[0] % self.n_rf == 0, "'input_array' dimensionality should match with n_rf."
        assert input_array.dtype == np.float64, "'input_array' must be of dtype np.float64"
        g_cell = GanglionCell(n_rf=self.n_rf, t_max=self.t_max, alpha=self.alpha, amp=self.amp,
                              freq=self.freq, phi_0=self.phi_o, delta_phi=self.delta_phi)
        n_input = input_array.shape[0]  # n_input表示所有像素的数量
        ganshouyedeshumu = (n_input / self.n_rf)  # 感受野的数目，即神经元的数目
        fields = np.split(input_array, ganshouyedeshumu)  # 均等分割
        encoded = list()
        for row in fields:
            encoded.append(g_cell.encode(stimulation=row))

        # 找到整个大列表中的最小值
        min_value = min(min(item) for item in encoded)
        # 对大列表中的每个元素进行操作
        encoded = [[(x - min_value) for x in item] for item in encoded]  # 让时间序列没有负的值

        # 找到整个大列表中的最大值
        max_value = max(max(item) for item in encoded)
        # 对大列表中的每个元素进行操作
        encoded = [[(x / max_value) * 1000 * 2 for x in item] for item in encoded]  # 我在这里乘以了两倍，也就是tmax

        arithmetic_sequence = [0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800]
        for sublist_index, sublist in enumerate(encoded):
            # 遍历小列表中的每个元素并替换
            for i, element in enumerate(sublist):
                closest_value = min(arithmetic_sequence, key=lambda x: abs(x - element))
                encoded[sublist_index][i] = closest_value

        list1 = []  # 8x100 这个list1是由0和1组成的矩阵
        for i in range(len(encoded)):
            list1.append(arithmetic_sequence.copy())
            for item_index, item in enumerate(list1[i]):
                if item in encoded[i]:
                    list1[i][item_index] = 1
                else:
                    list1[i][item_index] = 0

        for sublist in encoded:
            for i in range(len(sublist)):
                sublist[i] = np.float64(sublist[i])  # 转化为保留两位小数

        # 去除每个小列表中的重复元素
        encoded = [list(set(small_list)) for small_list in encoded]

        plt.figure(figsize=(10, 5))

        plt.style.use('ggplot')
        plt.eventplot(encoded, linelengths=0.8, linewidths=2.5, colors='blue')  # 绘制相同的平行线
        plt.yticks(np.arange(0, len(encoded), 1))  # 指定坐标轴的刻度,往下移-0.3
        plt.xticks(np.arange(0, 2001, 200))  # 指定坐标轴的刻度,1000毫秒
        plt.xlabel('time(ms)')
        plt.ylabel('neuron number')
        # plt.title('编码后的脉冲刺激序列')
        plt.grid(color='black', axis='y')

        plt.show()
        return encoded, list1


class ImageProcessing:
    def __init__(self, n_rf, m, n):
        self.n_rf = n_rf
        self.m = m
        self.n = n

    def ImageProcess(self, Image):

        """将图像分成各个感受野RF，

        RF中像素数量为n_rf

        """
        assert (Image.shape[0] * Image.shape[
            1]) % self.n_rf == 0, "'input_array' dimensionality should match with n_rf."
        Image_1d = list()
        for i in range(0, Image.shape[0], self.m):
            for j in range(0, Image.shape[1], self.n):
                for s in range(0, self.m):
                    Image_1d.extend(Image[i + s, j:j + self.n])

        return np.array(Image_1d) / 255


class GanglionCell:
    def __init__(self, n_rf, t_max=1., alpha=1., amp=1., freq=40., phi_0=0., delta_phi=None):
        self.n_rf = n_rf
        self.t_max = t_max
        self.alpha = alpha
        self.amp = amp
        self.freq = freq
        self.omega = 2 * np.pi * freq
        self.phi_0 = phi_0
        if delta_phi is None:
            self.delta_phi = 2 * np.pi / n_rf
        else:
            self.delta_phi = delta_phi

    def encode(self, stimulation):  # 输入的是单个感受野的所有像素点
        """Encode input stimulation.
        Args
        ----
        stimulation (:obj: np.ndarray): Must be of of shape (n_rf,) .
        :returns :obj: np.array of shape(n_rf,) and dtype int64
        """
        receptive_field = [PhotoReceptor(t_max=self.t_max, alpha=self.alpha) for _ in range(self.n_rf)]
        # 整个列表推导式的目的是创建一个包含self.n_rf个PhotoReceptor对象的列表，并将这些对象存储在receptive_field变量中。
        out_spike_times = np.zeros(self.n_rf, dtype=np.float64)
        for (ind, intensity) in enumerate(stimulation):  # enumerate()可以同时获得索引和值
            pr_spike_time = (receptive_field[ind].get_spike_time(intensity=intensity) * 1000).astype(np.int64)  # 以毫秒为单位
            out_spike_times[ind] = pr_spike_time

        # --------时间偏移-----------
        for (id, spike) in enumerate(out_spike_times):
            if id != 0:
                out_spike_times[id] = spike + 10 / self.n_rf * id
            if id == 0:
                out_spike_times[id] = spike
        # -----------------------------------------

        return out_spike_times


class PhotoReceptor:
    def __init__(self, t_max, alpha):
        """
        unit.

        Args
        ----
        t_max (float): Max output interval is seconds.
        alpha (float): Scaling factor used in logarithmic transformation function.

        """
        assert isinstance(t_max, float)
        assert isinstance(alpha, float)
        self.t_max = t_max
        self.alpha = alpha

    def get_spike_time(self, intensity):  # intensity表示输入的一个像素值
        """以秒为单位返回尖峰时间.
        intensity (float): 浮点数并且标准化分布.
        :以秒为单位返回浮点型的峰值时间.
        """
        assert isinstance(intensity, float), "'intensity' must be of type float"  # 检查对象是否是另一个对象的子类
        spike_time = self.t_max * (1 - np.arctan(1.557 * intensity))
        return spike_time


# 初始化神经网络
net = ETH_Network()

net.load_state_dict(torch.load('.\modelpara.pth', map_location='cpu'))

# 选取的 9 张图片文件名为： apple_13  apple_26  apple_65  car_51  car_56  car_78  cup_14  cup_41  cup_100
#          论文中命名为： apple_1    apple_2   apple_3   car_1   car_2   car_3    cup1   cup2     cup3
file_path = "./dataset/ETH3x100/apple/apple_65.png"
# ETH测试集准确率测试
img = Image.open(file_path)
img_resized = img.resize((68, 68))

input_data = transforms.ToTensor()(img_resized).unsqueeze(0)  # Convert to tensor and add batch dimension

n_rf = 25  # 表示感受野中的像素数量，即m*n
m = 5  # 表示感受野的行数
n = 5  # 表示感受野列数

output = net(input_data)

print("Predicted Class:", output)

lazhi = output.view(-1)
numpy_array = lazhi.detach().numpy()
numpy_array = numpy_array.astype(np.float64)

encoder = DelayPhaseEncoder(n_rf)
encoding, channelstatus = encoder.encode(input_array=numpy_array)

match = re.search(r'/([^/]+)\.png$', file_path)

if match:
    filename = match.group(1)
    print(filename)  # 输出: car_344

# with open(filename +'_ChannelStatus.pkl', 'wb') as f:
#     pickle.dump(channelstatus, f)
