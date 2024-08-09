import os

folder_path = '/home/lll/software/fixed/MMDetector/test_data/3_4/VOCdevkit/VOC2007/Annotations'

files = os.listdir(folder_path)

for filename in files:
    if filename.endswith('.xml'):
        old_path = os.path.join(folder_path,filename)
        new_filename = '0' + filename
        new_path = os.path.join(folder_path,new_filename)
        os.rename(old_path,new_path)
print("end")