import datetime
import os

import torch
import scipy.io as scio
from PIL import Image
from torchvision import transforms

import numpy as np
from .dataloader_for_RECOG import MyDataset
from .dataloader_for_RECOG import normalization
from .models.DeepSVDD import DeepSVDDNetwork
from .models.ResNet import *
from .models.MLP import *
from .models.VGG import *
from .models.MultiModel import *
from .models.HNet_base import *

# 目标检测未用到
# 目标识别的三种测试：雷达单模，红外单模，雷达-红外双模
def test_hrrp(hrrp_path='', model2='', train_weight='', num_classes=0, class_name=None):

    # results_txt = './results/result-hrrp-' + model2_name + '-' \
    #               + datetime.date.today().strftime('%Y-%m-%d') + '.txt'
    if not os.path.exists('./results'):
        os.makedirs('./results')

    if class_name is None:
        class_name = []

    results_txt = './results/results.txt'

    if os.path.exists(results_txt):
        os.remove(results_txt)

    print('正在进行雷达单模识别...')

    model = None
    if model2 == 'hnet':
        model = HNet_base(num_classes)
    if model2 == 'deepsvdd':
        model = DeepSVDDNetwork(num_classes)
    elif model2 == 'mlp':
        model = MLP(num_classes)

    model.load_state_dict(torch.load(train_weight))

    model.eval()

    # print(model.state_dict())

    with torch.no_grad():
        init_hrrp = torch.zeros(1, 128)

        model(init_hrrp)

        with open(results_txt, 'w') as txt:
            txt.write('文件路径\t类别\t置信度\n')

            print('正在识别：%s' % hrrp_path)

            hrrp = scio.loadmat(hrrp_path)['data'][0]

            hrrp = normalization(hrrp)
            hrrp = torch.from_numpy(hrrp).float()
            hrrp = hrrp.view(1, 128)

            out = model(hrrp)

            res = torch.softmax(out[0], dim=0)

            # print(res)
            pred_prob, pred_label = torch.max(res, dim=0)

            print(res)
            for index, value in enumerate(class_name):
                if pred_label == index:
                    str_res = value + '\t{:.2}'.format(float(res[pred_label]))

            print(str_res)

            line = hrrp_path + '\t' + str_res + '\n'
            txt.write(line)

    print('雷达单模识别结束，结果已保存至' + results_txt)

# 目标识别 雷达单模测试
def test_hrrps(hrrps_path='', model2='', train_weight='', num_classes=2, class_name=None):

    if not os.path.exists('./results'):
        os.makedirs('./results')

    if class_name is None:
        class_name = []

    results_txt = './results/results.txt'

    if os.path.exists(results_txt):
        os.remove(results_txt)

    print('正在进行雷达单模识别...')
    model = None

    if model2 == 'hnet':
        model = HNet_base(num_classes)
    if model2 == 'deepsvdd':
        model = DeepSVDDNetwork(num_classes)
    elif model2 == 'mlp':
        model = MLP(num_classes)

    model.load_state_dict(torch.load(train_weight))

    model.eval()

    hrrps = scio.loadmat(hrrps_path)['raw_data']

    with torch.no_grad():
        init_hrrp = torch.zeros(1, 128)

        model(init_hrrp)

        with open(results_txt, 'w') as txt:
            txt.write('文件路径\t类别\t置信度\n')




            for hrrp in hrrps:
                hrrp = hrrp_dir_path + '/' + hrrp

                print('正在识别：%s' % hrrp)

                hrrp = scio.loadmat(hrrp)['data'][0]
                hrrp = normalization(hrrp)
                hrrp = torch.from_numpy(hrrp).float()

                hrrp = hrrp.view(1, 128)

                out = model(hrrp)

                res = torch.softmax(out[0], dim=0)

                pred_prob, pred_label = torch.max(res, dim=0)

                print(res)
                for index, value in enumerate(class_name):
                    if pred_label == index:
                        str_res = value + '\t{:.2}'.format(float(res[pred_label]))

                print(str_res)

                line = hrrp_dir_path + '\t' + str_res + '\n'
                txt.write(line)

    print('雷达单模识别结束，结果已保存至' + results_txt)


# 目标识别 图像单模测试
def test_image(image_path='', model1_name='', train_weight='', num_classes=0):

    # results_txt = './results/result-image-' + model1_name + '-' \
    #               + datetime.date.today().strftime('%Y-%m-%d') + '.txt'
    results_txt = './results/results.txt'

    if os.path.exists(results_txt):
        os.remove(results_txt)

    print('正在进行红外图像单模识别...')

    model = None

    if model1_name == 'resnet50':
        model = resnet50(num_classes=num_classes, model_path='')
    elif model1_name == 'resnet18':
        model = resnet18(num_classes=num_classes, model_path='')
    elif model1_name == 'vgg16':
        model = vgg16(num_classes=num_classes, model_path='')

    model.load_state_dict(torch.load(train_weight))

    model.eval()

    with torch.no_grad():
        init_img = torch.zeros(1, 3, 224, 224)
        model(init_img)
        # print(image.shape)
        images = os.listdir(image_path)

        with open(results_txt, 'a') as txt:
            txt.write('image_name cls prob\n')
            for image in images:
                image_name = image
                image = os.path.join(image_path, image)
                print('正在识别：%s' % image)

                image = Image.open(image)

                data_transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor()
                ])

                image = data_transform(image)
                image = image.view(1, 3, 224, 224)

                res = model(image)

                # pred_prob, pred_label = torch.max(res, dim=1)
                pred_prob, pred_label = torch.softmax(res, dim=1)

                if pred_label == 0:
                    str_res = 'car {:.2}'.format(float(res[0][pred_label]))
                elif pred_label == 1:
                    str_res = 'person {:.2}'.format(float(res[0][pred_label]))

                print(str_res)

                line = image_name + ' ' + str_res + '\n'
                txt.write(line)

    print('红外图像单模识别结束，结果已保存至' + results_txt)

# 目标识别 多模态测试
def test_mm(image_path='', hrrp_path='', model1_name='', model2_name='', label_path='', train_weight='', class_name=None, num_classes=2):
    # results_txt = './results/result-mm-' + model1_name + '-' + model2_name + '-' \
    #               + datetime.date.today().strftime('%Y-%m-%d') + '.txt'
    if class_name is None:
        class_name = []
    results_txt = './results/results.txt'
    if os.path.exists(results_txt):
        os.remove(results_txt)

    model = None
    model = MultiModel(model1_name, model2_name, num_classes)

    model.load_state_dict(torch.load(train_weight))

    # print(model.state_dict())
    model.eval()

    predict_dataset = MyDataset(image_path=image_path, hrrp_path=hrrp_path, label_path=label_path, class_name=class_name)

    print('正在进行红外/雷达双模态识别...')

    with torch.no_grad():
        init_img = torch.zeros(1, 3, 224, 224)
        init_hrrp = torch.zeros(1, 128)

        model(init_img, init_hrrp)

        # print(image.shape)
        images = predict_dataset.images
        hrrps = predict_dataset.hrrps
        i = 0
        indexs = predict_dataset.indexs
        with open(results_txt, 'a') as txt:
            txt.write('image_name cls prob\n')
            for image in images:
                image_name = image
                image = os.path.join(image_path, image)
                print('正在识别：%s' % image)

                image = Image.open(image)

                data_transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor()
                ])

                image = data_transform(image)
                image = image.view(1, 3, 224, 224)

                idx = indexs[i]
                i = i + 1
                hrrp = hrrps[idx]
                hrrp = normalization(hrrp)
                hrrp = torch.from_numpy(hrrp).float()

                hrrp = hrrp.view(1, 128)

                res = model(image, hrrp)

                pred_prob, pred_label = torch.max(res, dim=1)
                if pred_label == 0:
                    str_res = 'vehicle {:.2}'.format(float(res[0][pred_label]))
                elif pred_label == 1:
                    str_res = 'building {:.2}'.format(float(res[0][pred_label]))

                print(str_res)

                line = image_name + ' ' + str_res + '\n'
                txt.write(line)

    print(f'红外/雷达双模识别结束，结果已保存至' + results_txt)


def predict_image(num_classes=2):
    model = resnet50(num_classes=num_classes)
    train_weight = "./weights/model-resnet50.pth"
    model.load_state_dict(torch.load(train_weight))

    # print(model.state_dict())

    image = Image.open("./test/1121R-101_1.jpg")

    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    image = data_transform(image)
    image = image.view(1, 3, 224, 224)

    model.eval()

    with torch.no_grad():
        init_img = torch.zeros(1, 3, 224, 224)
        model(init_img)
        # print(image.shape)
        res = model(image)
        print("+++++++++++++++++++ 红外图像识别结果 ++++++++++++++++++++")
        print(res)
        if res[0][0] > res[0][-1]:
            print('识别结果：car')
        else:
            print('识别结果：person')
        print("+++++++++++++++++++++++++++++++++++++++++++++++++")


if __name__ == '__main__':
    # test_hrrp('./test_data/building.mat', 'hnet', train_weight='./weights/model-hrrp-hnet-2022-11-06.pth',
    #           num_classes=2, class_name=['vehicle', 'building'])

    test_hrrp('./test_data/building.mat', 'hnet', train_weight='./weights/model-hrrp-hnet-2023-03-29.pth',
              num_classes=5, class_name=['airplane', 'building', 'car', 'tank', 'truck'])

    # test_image('./temp/images', 'resnet50', train_weight='./weights/model-image-resnet50-2022-04-26.pth', num_classes=2)
    # test_mm('./temp/images', './test_data/test_hrrps', 'resnet50', 'deepsvdd',
    #         train_weight='./weights/model-mm-resnet50-deepsvdd-2022-05-05.pth', num_classes=2)
