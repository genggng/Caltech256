# -*- coding: utf-8 -*-
import argparse
import os

import torch
import torchvision.transforms as transforms
from torch import nn, optim
from torch.utils.data import DataLoader

from flyai_sdk import FlyAI, DataHelper, MODEL_PATH
from vgg import vgg11
from PIL import Image
from torch.utils.data import Dataset
from my_transform import get_train_transform
from warm_lr import adjust_learning_rate_cosine,adjust_learning_rate_step

from build_efficientnet import Efficientnet


'''
样例代码仅供参考学习，可以自己修改实现逻辑。
模版项目下载支持 PyTorch、Tensorflow、Keras、scikit-learn等机器学习框架
常见问题：https://doc.flyai.com/question.html
遇到问题不要着急，添加小姐姐微信，扫描项目里面的：FlyAI小助手二维码-小姐姐在线解答您的问题.png
'''

if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# 项目的超参数

# image_mean = [0.38753143, 0.36847523, 0.27735737]
# image_std = [0.25998375, 0.23844026, 0.2313706]
epochs = 200
batch_size = 256
train_csv_path = os.path.join(".", "data", "input", "Caltech256", "train.csv")
learn_rate = 0.001


# 判断GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_gpu = torch.cuda.is_available()

# 创建保存模型的文件夹
save_dir = './trained'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


class MyDataset(Dataset):
    def __init__(self, txt_path, transform=None, target_transform=None):
        fh = open(txt_path, 'r')
        imgs = []
        i = 0
        for line in fh:
            if i == 0:
                i = 1
                continue
            else:
                line = line.rstrip()
                words = line.split(',')
                imgs.append((words[0], int(words[1])))

        self.imgs = imgs  # 最主要就是要生成这个list， 然后DataLoader中给index，通过getitem读取图片数据
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        fn = os.path.join('data', 'input', 'Caltech256', fn)
        img = Image.open(fn).convert('RGB')  # 像素值 0~255，在transfrom.totensor会除以255，使像素值变成 0~1

        if self.transform is not None:
            img = self.transform(img)  # 在这里做transform，转为tensor等等

        return img, label

    def __len__(self):
        return len(self.imgs)


class Main(FlyAI):
    '''
    项目中必须继承FlyAI类，否则线上运行会报错。
    '''

    def __init__(self):
        self.train_trans = get_train_transform(size=300)


    def init_net(self):
        self.net = Efficientnet('efficientnet-b7',num_classes=256)
        self.net.to(device)
        self.loss_function = nn.CrossEntropyLoss()

    def download_data(self):
        # 根据数据ID下载训练数据

        data_helper = DataHelper()
        data_helper.download_from_ids("Caltech256")

    def deal_with_data(self):
        '''
        处理数据，没有可不写。
        :return:
        '''
        # 构建MyDataset 实例
        self.train_data = MyDataset(txt_path=train_csv_path, transform=self.train_trans)
        self.train_loader = DataLoader(dataset=self.train_data, batch_size=batch_size, shuffle=True, drop_last=True)

    def train(self):
        '''
        训练模型，必须实现此方法
        :return:
        '''
        
        self.init_net()
        self.deal_with_data()
        optimizer = optim.SGD(self.net.parameters(), lr=learn_rate,
                      momentum=0.9, weight_decay=0.001)
        
        criterion = self.loss_function
        epoch_size = len(self.train_data) // batch_size
        epoch = 0
        max_iter = epochs * epoch_size

        global_step = 0
        # step 学习率调整参数
        stepvalues = (10 * epoch_size, 20 * epoch_size, 30 * epoch_size)
        step_index = 0

       
       
        for iteration in range(max_iter):
            print("开始训练")

            global_step += 1

            ##更新迭代器
            if iteration % epoch_size == 0:
                # create batch iterator
                batch_iterator = iter(self.train_loader)
                loss = 0
                epoch += 1
                if epoch > 1:
                    pass

                self.net.train()
            

            if iteration in stepvalues:
                step_index += 1
            lr = adjust_learning_rate_step(optimizer, learn_rate, 0.1, epoch, step_index, iteration, epoch_size)

            ## 获取image 和 label
            images, labels = next(batch_iterator)
            images, labels = images.cuda(), labels.cuda()

            out = self.net(images)
            loss = criterion(out, labels)

            optimizer.zero_grad()  # 清空梯度信息，否则在每次进行反向传播时都会累加
            loss.backward()  # loss反向传播
            optimizer.step()  ##梯度更新

            prediction = torch.max(out, 1)[1]
            train_correct = (prediction == labels).sum()
            ##这里得到的train_correct是一个longtensor型，需要转换为float
            # print(train_correct.type())
            train_acc = (train_correct.float()) / batch_size
            

            if iteration % 10 == 0:
                print('Epoch:' + repr(epoch) + ' || epochiter: ' + repr(iteration % epoch_size) + '/' + repr(epoch_size)
                    + '|| Totel iter ' + repr(iteration) + ' || Loss: %.6f||' % (loss.item()) + 'ACC: %.3f ||' %(train_acc * 100) + 'LR: %.8f' % (lr))

        torch.save(self.net, save_dir + '/trained_model.pth')

if __name__ == '__main__':
    main = Main()
    main.download_data()
    main.train()
