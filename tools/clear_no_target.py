import os
from xml.etree import ElementTree as ET
import shutil


path = '../test_data/3_4/VOCdevkit/VOC2007/Annotations'
path1 = '../test_data/3_4/VOCdevkit/VOC2007/Ann'

cnt = 0

remove_list = []


for file in os.listdir(path):
    full_path = os.path.join(path,file)
    root = ET.parse(full_path).getroot()
    for obj in root.findall('object'):
        name_obj = obj.find('name')
        if name_obj.text == 'notruck':
            try:
                shutil.move(os.path.join(path,file),os.path.join(path1,file))
            except:
                continue


