import os
import numpy as np
from numpy import mean
from torchvision import transforms
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from . import dataloader_for_RECOG
import torch.optim as optim
from torch.autograd import Variable
import torch
from .models.MLP import MLP
from .models.ResNet import ResNet
from .models.DeepSVDD import DeepSVDDNetwork
from .models.HNet_base import HNet_base
import datetime
from tqdm import tqdm


# 目标检测未用到
# 雷达单模训练
def train_hrrp(hrrps_path='', label_path='', model2_name='', class_name=None,
               num_classes=5, epochs=100, batch_size=16, learning_rate=0.001):
    if class_name is None:
        class_name = []

    if not os.path.exists('./weights'):
        os.makedirs('./weights')

    print('开始进行雷达信号的单模训练...')

    data_train = dataloader_for_RECOG.HRRPDataset(hrrps_path, label_path, class_name)

    # data_test = dataloader_for_RECOG.HRRPDataset(data_path, hrrp_path, test_file)

    train_loader = DataLoader(data_train,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=4
                              )

    # test_loader = DataLoader(data_test,
    #                          batch_size=batch_size,
    #                          shuffle=False,
    #                          num_workers=0
    #                          )

    use_gpu = torch.cuda.is_available()

    model = None
    save_path = ''

    if model2_name == 'hnet':
        model = HNet_base(num_classes)
        save_path = './weights/hrrps_models/model_hnet.pth'

    if torch.cuda.device_count() > 1:
        print("检测到", torch.cuda.device_count(), "块GPU，使用GPU运行!")
        model = nn.DataParallel(model)
    elif torch.cuda.device_count() == 1:
        print("检测到一块GPU，使用GPU运行！")
    else:
        print("该系统未检测到GPU，使用CPU运行！")

    if use_gpu:
        model = model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    # scheduler_lr = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Accuracy = []
    # cnt = 0
    for epoch in range(epochs):

        epoch_count = 0
        total_training_loss = 0.0

        for iter, traindata in enumerate(train_loader):

            train_inputs, train_labels = traindata

            if use_gpu:
                train_inputs, train_labels = torch.autograd.Variable(train_inputs.cuda()), torch.autograd.Variable(
                    train_labels.cuda())
            else:
                train_inputs, train_labels = Variable(train_inputs), Variable(train_labels)

            optimizer.zero_grad()

            train_outputs = model(train_inputs)

            loss = criterion(train_outputs, train_labels)

            loss.requires_grad_()
            loss.backward()
            optimizer.step()

            # total += train_labels.size(0)
            total_training_loss += loss.item()
            epoch_count += 1

        print('Training Phase: Epoch: [%2d/%2d]\tIteration Loss: %.6f' %
              (epoch, epochs, total_training_loss / epoch_count))

        # scheduler_lr.step()

    torch.save(model.state_dict(), save_path)
    print(f'雷达单模训练完成！权重文件{save_path}已保存!')
    print('请继续操作！')
    model.eval()

    Accuracy = []

    for iter, testdata in enumerate(train_loader):
        cnt = 0
        test_inputs, test_labels = testdata
        if use_gpu:
            test_inputs, test_labels = Variable(test_inputs.cuda()), Variable(test_labels.cuda())
        else:
            test_inputs, test_labels = Variable(test_inputs), Variable(test_labels)

        test_outputs = model(test_inputs)

         # test_outputs = torch.argmax(test_outputs, dim=1)
         # print(test_outputs[:3])
         # print(test_labels[:3])

        for i in range(len(test_outputs)):
            if torch.argmax(test_outputs[i]) == test_labels[i]:
                # print('*******************')
                # print(torch.argmax(test_outputs[i]))
                # print(torch.argmax(test_labels[i]))
                # print('*******************')
                cnt += 1

        # test_loss = criterion(test_outputs, test_labels)
        # print('Testing Phase: Epoch: [%2d][%2d/%2d]\tIteration Loss: %.3f' %
        #        (iter, epoch, epochs, test_loss.item()))
        # print("=======================================================")
        Accuracy.append((cnt / batch_size) * 100)

         # print(cnt)
         # print(len(test_loader) * batch_size)
    Accuracy = mean(Accuracy)
    print("Avg Accuracy: %.2f %%" % Accuracy)


if __name__ == '__main__':
    # train_hrrp('./datasets/data_for_RECOG/hrrps', './datasets/data_for_RECOG/label.csv', 'deepsvdd')
    train_hrrp('./test_data/3_4/hrrps.mat', './test_data/3_4/label.csv',
               'hnet', ['car', 'building'],
               num_classes=2)
