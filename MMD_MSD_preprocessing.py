import os
import cv2
import numpy as np
from tqdm import tqdm


# 目标检测未用到
# 目标识别的数据预处理
# 单张图像分割
def image_cut(img_path, label_path, split_images_path):

    coor = []
    print(label_path)
    with open(label_path, 'r') as csv:
        next(csv)

        lines = csv.readlines()

        print(lines)

        for line in lines:
            print(line)
            image_name = line.split(',')[0]
            tar_image_name = line.split(',')[1] + '.jpg'

            # image_path = os.path.join(parser_data.img_path, image_name)
            print(img_path)
            img = cv2.imread(img_path)

            x_min = int(line.split(',')[4])
            y_min = int(line.split(',')[5])
            x_max = int(line.split(',')[6])
            y_max = int(line.split(',')[7])

            coor.append([x_min, y_min])
            coor.append([x_max, y_min])
            coor.append([x_min, y_max])
            coor.append([x_max, y_max])
            # print(coor)
            src_pts = np.float32(coor)
            coor = []

            width = x_max - x_min
            height = y_max - y_min

            tar_pts = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

            # print(tar_pts)
            matrix = cv2.getPerspectiveTransform(src_pts, tar_pts)

            tar_image = cv2.warpPerspective(img, matrix, (width, height))

            tar_image_path = os.path.join(split_images_path, tar_image_name)

            print(tar_image_path)
            cv2.imwrite(tar_image_path, tar_image)


# 文件夹下的所有图像分割
def images_cut(img_dir_path, label_path, split_images_path):

    coor = []

    img_list = os.listdir(img_dir_path)

    # print(img_list)

    # print(src_pts)
    with open(label_path, 'r') as csv:
        next(csv)

        lines = csv.readlines()
        # p = progressbar.ProgressBar()
        lines = tqdm(lines)
        for line in lines:
            # print(line)
            image_name = line.split(',')[0]

            if image_name in img_list:

                tar_image_name = line.split(',')[1] + '.jpg'
                # print('正在生成图像：' + tar_image_name)

                image_path = os.path.join(img_dir_path, image_name)
                # print(image_path)
                img = cv2.imread(image_path)

                x_min = int(line.split(',')[4])
                y_min = int(line.split(',')[5])
                x_max = int(line.split(',')[6])
                y_max = int(line.split(',')[7])

                coor.append([x_min, y_min])
                coor.append([x_max, y_min])
                coor.append([x_min, y_max])
                coor.append([x_max, y_max])
                # print(coor)
                src_pts = np.float32(coor)
                coor = []

                width = x_max - x_min
                height = y_max - y_min

                tar_pts = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

                # print(tar_pts)
                matrix = cv2.getPerspectiveTransform(src_pts, tar_pts)

                tar_image = cv2.warpPerspective(img, matrix, (width, height))
                tar_image_path = os.path.join(split_images_path, tar_image_name)
                # print(tar_image_path)
                cv2.imwrite(tar_image_path, tar_image)


if __name__ == '__main__':
    # image_cut('../tes/t/1121R-101.jpg', '../test/label_image.csv')
    images_cut('./test_data/images', './test_data/label.csv', './temp/images')