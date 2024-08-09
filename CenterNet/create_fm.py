import os
import cv2
from centernet import CenterNet
from PIL import Image
from nets.centernet import CenterNet_HourglassNet, CenterNet_Resnet50
import torch
import numpy as np
from utils.utils import (cvtColor, get_classes, preprocess_input, resize_image,
                         show_config)


from torchvision.utils import make_grid, save_image

if __name__ == '__main__':
    
    net = CenterNet_Resnet50(num_classes=2, pretrained=False).cuda()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_path = '/home/lll/software/fixed/MMDetector/weights/multimodelbest_epoch_weights.pth'

    net.load_state_dict(torch.load(model_path, map_location=device))

    img = 'imgs/00037.jpg'

    image = Image.open(img)

    # r_image = CenterNet.create_fm(image, crop=crop, count=count)
    
    # r_image.save(os.path.join(dir_save_path, "result.png"), quality=95, subsampling=0)
    
    image_shape = np.array(np.shape(image)[0:2])
    # ---------------------------------------------------------#
    #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
    #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
    # ---------------------------------------------------------#
    image = cvtColor(image)
    # ---------------------------------------------------------#
    #   给图像增加灰条，实现不失真的resize
    #   也可以直接resize进行识别
    # ---------------------------------------------------------#
    image_data = resize_image(image, (512, 512), False)
    # -----------------------------------------------------------#
    #   图片预处理，归一化。获得的photo的shape为[1, 512, 512, 3]
    # -----------------------------------------------------------#
    image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

    images = torch.from_numpy(np.asarray(image_data)).type(torch.FloatTensor)

    images = images.cuda()
    # ---------------------------------------------------------#
    #   将图像输入网络当中进行预测！
    # ---------------------------------------------------------#
    # outputs = self.net(images)
    # for name, layer in self.net.named_modules():
    #     print(name)

    FEATURE_FOLDER = "./imgs_out/features"

    if not os.path.exists(FEATURE_FOLDER):
        os.mkdir(FEATURE_FOLDER)

    # three global vatiable for feature image name
    feature_list = list()
    count = 0
    idx = 0
    
    def get_image_path_for_hook(module): 
        global count  
        image_name = feature_list[count] + ".png"
        count += 1
        image_path = os.path.join(FEATURE_FOLDER, image_name)
        return image_path
    
    def hook_func(module, input, output):
        image_path = get_image_path_for_hook(module)
        data = output.clone().detach()
        global idx
        print(idx, "->", data.shape)
        idx+=1
        data = data.data.permute(1, 0, 2, 3)

        # data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        print('----------------')
        print(type(data))
        print('----------------')

        save_image(data, image_path, normalize=False)

    for name, module in net.named_modules():
        
        if isinstance(module, torch.nn.Conv2d):
            print(name)
            feature_list.append(name)
            module.register_forward_hook(hook_func)
    
    out = net(images)
