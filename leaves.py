import d2l.torch
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# This is for the progress bar.

# 看看label文件长啥样
labels_dataframe = pd.read_csv('./data/train.csv')
# labels_dataframe.head(5)
leaves_labels = sorted(list(set(labels_dataframe['label'])))
n_classes = len(leaves_labels)
class_to_num = dict(zip(leaves_labels, range(n_classes)))
num_to_class = {v: k for k, v in class_to_num.items()}


# 继承pytorch的dataset，创建自己的
class LeavesData(Dataset):
    def __init__(self, csv_path, file_path, mode='train', valid_ratio=0.2, resize_height=256, resize_width=256):
        """
        Args:
            csv_path (string): csv 文件路径
            img_path (string): 图像文件所在路径
            mode (string): 训练模式还是测试模式
            valid_ratio (float): 验证集比例
        """

        # 需要调整后的照片尺寸，我这里每张图片的大小尺寸不一致#
        self.resize_height = resize_height
        self.resize_width = resize_width

        self.file_path = file_path
        self.mode = mode

        # 读取 csv 文件
        # 利用pandas读取csv文件
        self.data_info = pd.read_csv(
            csv_path, header=None)  # header=None是去掉表头部分
        # 计算 length
        self.data_len = len(self.data_info.index) - 1
        self.train_len = int(self.data_len * (1 - valid_ratio))

        if mode == 'train':
            # 第一列包含图像文件的名称
            # self.data_info.iloc[1:,0]表示读取第一列，从第二行开始到train_len
            self.train_image = np.asarray(
                self.data_info.iloc[1:self.train_len, 0])
            # 第二列是图像的 label
            self.train_label = np.asarray(
                self.data_info.iloc[1:self.train_len, 1])
            self.image_arr = self.train_image
            self.label_arr = self.train_label
        elif mode == 'valid':
            self.valid_image = np.asarray(
                self.data_info.iloc[self.train_len:, 0])
            self.valid_label = np.asarray(
                self.data_info.iloc[self.train_len:, 1])
            self.image_arr = self.valid_image
            self.label_arr = self.valid_label
        elif mode == 'test':
            self.test_image = np.asarray(self.data_info.iloc[1:, 0])
            self.image_arr = self.test_image

        self.real_len = len(self.image_arr)

        print('Finished reading the {} set of Leaves Dataset ({} samples found)'
              .format(mode, self.real_len))

    def __getitem__(self, index):
        # 从 image_arr中得到索引对应的文件名
        single_image_name = self.image_arr[index]

        # 读取图像文件
        img_as_img = Image.open(self.file_path + single_image_name)

        # 如果需要将RGB三通道的图片转换成灰度图片可参考下面两行
        #         if img_as_img.mode != 'L':
        #             img_as_img = img_as_img.convert('L')

        # 设置好需要转换的变量，还可以包括一系列的nomarlize等等操作
        if self.mode == 'train':
            transform = transforms.Compose([
                # transforms.RandomResizedCrop((224,224), scale=(0.8, 1), ratio=(0.8, 1.2)), #随机剪裁
                # transforms.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5), #颜色亮度色调
                # transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转 选择一个概率
                transforms.RandomVerticalFlip(p=0.5),  # 随机水平翻转 选择一个概率
                transforms.ToTensor()
            ])
        else:
            # valid和test不做数据增强
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])

        img_as_img = transform(img_as_img)

        if self.mode == 'test':
            return img_as_img
        else:
            # 得到图像的 string label
            label = self.label_arr[index]
            # number label
            number_label = class_to_num[label]

            return img_as_img, number_label  # 返回每一个index对应的图片数据和对应的label

    def __len__(self):
        return self.real_len


train_path = './data/train.csv'
test_path = './data/test.csv'
# csv文件中已经images的路径了，因此这里只到上一级目录
img_path = './data/'

train_dataset = LeavesData(train_path, img_path, mode='train')
val_dataset = LeavesData(train_path, img_path, mode='valid')
test_dataset = LeavesData(test_path, img_path, mode='test')

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=0)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0)
device = torch.device('cpu')
criterion = nn.CrossEntropyLoss()
lr = 0.9


class resnet18_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.stride = stride
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # 若输入通道与输出通道不一样，则使用side 1x1卷积块
        if (in_channels != out_channels):
            self.conv_side = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
            self.bn_side = nn.BatchNorm2d(out_channels)
        else:
            self.conv_side = None
            self.bn_side = None

    def forward(self, X):
        out1 = self.relu(self.bn1(self.conv1(X)))
        out2 = self.bn2(self.conv2(out1))
        if self.conv_side:
            X = self.bn_side(self.conv_side(X))
        return self.relu(out2 + X)


class ResNet18(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels=64, kernel_size=7, padding=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        )
        self.layer2 = nn.Sequential(
            resnet18_block(64, 64),
            resnet18_block(64, 128, stride=2),
        )
        self.layer3 = nn.Sequential(
            resnet18_block(128, 128),
            resnet18_block(128, 256, stride=2)
        )
        self.layer4 = nn.Sequential(
            resnet18_block(256, 256),
            resnet18_block(256, 512, stride=2)
        )
        self.layer5 = nn.Sequential(
            resnet18_block(512, 512),
            resnet18_block(512, 1024, stride=2)
        )
        self.layer6 = nn.Sequential(
            resnet18_block(1024, 1024),
            resnet18_block(1024, 2048, stride=2)
        )
        self.layer7 = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(2048, num_classes),
            nn.Dropout(p=0.5),
            nn.Sigmoid()
        )

    def forward(self, X):
        X1 = self.layer1(X)
        X2 = self.layer2(X1)
        X3 = self.layer3(X2)
        X4 = self.layer4(X3)
        X5 = self.layer5(X4)
        X6 = self.layer6(X5)
        X7 = self.layer7(X6)

        return X7


net = ResNet18(3, n_classes)


def train(train_loader, val_loader, net, lr):
    loss_arr = []
    net.to(device)
    optim_my = optim.SGD(params=net.parameters(), lr=lr)
    for (train_X, train_Y) in train_loader:
        train_X, train_Y = train_X.to(device),train_Y.to(device)
        optim_my.zero_grad()
        Y_hat = net(train_X)
        loss = criterion(Y_hat, train_Y)
        loss.to(device)
        loss.backward()
        loss_arr.append(loss)
        optim_my.step()
    plt.scatter(loss_arr)
    return loss_arr


train(train_loader, val_loader, net, lr)
