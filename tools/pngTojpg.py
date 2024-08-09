import cv2
import os

input_folder = '/home/lll/software/fixed/MMDetector/test_data/ShangHai_4/VOCdevkit/VOC2007/images'
output_folder = '/home/lll/software/fixed/MMDetector/test_data/ShangHai_4/VOCdevkit/VOC2007/JPEGImages'

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 遍历输入文件夹中的所有文件
for filename in os.listdir(input_folder):
    if filename.endswith('.png'):  # 仅处理以 .png 结尾的文件
        # 构建输入和输出文件的完整路径
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + '.jpg')

        # 读取红外图像
        infrared_image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

        # 将红外图像保存为JPEG格式
        cv2.imwrite(output_path, infrared_image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

print("转换完成！")