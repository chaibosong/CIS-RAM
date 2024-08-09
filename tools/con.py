import os
import shutil
d_f = '/home/lll/software/fixed/MMDetector/test_data/ShangHai_4/VOCdevkit/VOC2007/JPEGImages'
# s_f2 = '/home/lll/software/fixed/MMDetector/test_data/3_4/VOCdevkit/VOC2007/Annotations'
s_f1 = '/home/lll/software/fixed/MMDetector/test_data/3_4/VOCdevkit/VOC2007/JPEGImages'

file_list1 = os.listdir(s_f1)

for file_name in file_list1:
    s_file = os.path.join(s_f1,file_name)
    # d_file = os.path.join(d_f,file_name)
    if len(file_name) == 9 :
        shutil.copy(s_file, d_f)

# file_list2 = os.listdir(s_f2)
# for file_name in file_list2:
#     s_file = os.path.join(s_f2,file_name)
#     d_file = os.path.join(d_f,file_name)
#     shutil.copy(s_file,d_file)