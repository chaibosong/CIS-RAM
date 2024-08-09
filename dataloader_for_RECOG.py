import torch
import os
from PIL import Image
from torch.utils.data.dataset import Dataset
import numpy as np
import scipy.io as scio
from torchvision import datasets, models, transforms


# 目标检测未用到
# 双模数据集的加载
class MyDataset(Dataset):
    def __init__(self, image_path='', hrrp_path='', label_path='', class_name=None):

        if class_name is None:
            class_name = []

        self.img_path = image_path
        self.hrrp_path = hrrp_path
        self.label_path = label_path

        self.images = []
        self.labels = []
        self.indexs = []

        self.hrrps = scio.loadmat(hrrp_path)['raw_data']

        with open(self.label_path, 'r') as fp:
            next(fp)

            for f in fp:
                self.images.append(f.split(',')[1].strip() + '.jpg')

                for index, value in enumerate(class_name):
                    if f.split(',')[3].strip() == value:
                        self.labels.append(np.array(index))

                self.indexs.append(int(f.split(',')[2].strip()))

        self.transform = {
            'train': transforms.Compose([
                transforms.Resize((224, 224)),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
                #               transforms.Normalize([0.656,0.487,0.411], [1., 1., 1.])
            ]),
            'val': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
                #               transforms.Normalize([0.656,0.487,0.411], [1., 1., 1.])
            ]),
            'test': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
                #               transforms.Normalize([0.656,0.487,0.411], [1., 1., 1.])
            ])
        }

    def __getitem__(self, index):
        # 根据当前行号获得雷达数据

        idx = self.indexs[index]

        # print(index, idx)
        # print('*****************')
        # 读取一项HRRP数据
        img = Image.open(os.path.join(self.img_path, self.images[index]))
        # print(os.path.join(self.img_path, self.images[index]))
        img = self.transform['train'](img)

        hrrp = self.hrrps[idx]

        hrrp = normalization(hrrp)

        hrrp = np.float32(hrrp)

        label = torch.from_numpy(self.labels[index]).long()

        return img, hrrp, label

        # return self.images[index], self.hrrps[index]

    def __len__(self):
        return len(self.labels)


# 加载红外图像数据集
class ImageDataset(Dataset):
    def __init__(self, image_path='', label_path='', class_name=None):
        if class_name is None:
            class_name = []

        self.img_path = image_path

        self.images = []
        self.labels = []
        self.indexs = []

        self.label_path = label_path

        with open(self.label_path, 'r') as fp:
            next(fp)

            for f in fp:
                self.images.append(f.split(',')[1].strip() + '.jpg')

                for index, value in enumerate(class_name):
                    if f.split(',')[3].strip() == value:
                        self.labels.append(np.array(index))

        self.transform = {
            'train': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
                # transforms.Normalize([0.656,0.487,0.411], [1., 1., 1.])
            ]),
            'val': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
                # transforms.Normalize([0.656,0.487,0.411], [1., 1., 1.])
            ]),
        }

    def __getitem__(self, index):

        # 读取一张红外图像
        img = Image.open(os.path.join(self.img_path, self.images[index]))

        img = self.transform['train'](img)

        label = torch.from_numpy(self.labels[index]).long()

        return img, label

    def __len__(self):
        return len(self.images)


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


# 加载HRRP数据集
class HRRPDataset(Dataset):
    def __init__(self, hrrp_path='', label_path='', class_name=None):
        if class_name is None:
            class_name = []

        self.hrrp_path = hrrp_path

        try:
            self.hrrps = scio.loadmat(hrrp_path)['aa']
        except Exception as e:
            self.hrrps = scio.loadmat(hrrp_path)['raw_data']

        self.labels = []
        self.indexs = []

        # reading hrrp file from file
        self.label_path = label_path

        with open(self.label_path, 'r') as fp:
            next(fp)

            for f in fp:
                for index, value in enumerate(class_name):
                    if f.split(',')[4].strip() == value:
                        # print(value)
                        self.labels.append(np.array(index))

                self.indexs.append(int(f.split(',')[2].strip()))

    def __getitem__(self, index):
        # 读取一项HRRP数据
        # 获取index对应的行号，从行号获取雷达数据
        idx = self.indexs[index]

        hrrp = self.hrrps[idx]

        hrrp = normalization(hrrp)

        hrrp = np.float32(hrrp)
        label = torch.from_numpy(self.labels[index]).long()

        return hrrp, label

    def __len__(self):
        return len(self.labels)


if __name__ == '__main__':
    hrrp = HRRPDataset('./test_data/MMData2/hrrps.mat',
                       './test_data/MMData2/label.csv',
                       ['airplane', 'building', 'car', 'tank', 'truck']).__getitem__(0)
    print(hrrp)
    #
    # image = ImageDataset('./temp/images/', './test_data/label.csv', ['vehicle', 'building']).__getitem__(0)
    # print(image)

    # data = MyDataset('./temp/images/', './test_data/hrrps.mat', './test_data/label.csv', ['vehicle', 'building']).__getitem__(0)
    # print(data)
