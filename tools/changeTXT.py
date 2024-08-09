input_file = '/home/lll/software/fixed/MMDetector/test_data/3_4/VOCdevkit/VOC2007/ImageSets/Main/test.txt'
output_file = '/test_data/3_4/VOCdevkit/VOC2007/ImageSets/Main/test.txt'
with open(input_file,'r') as file_in,open(output_file,'w') as file_out:
    for line in file_in:
        modefied_line = '0' + line
        file_out.write(modefied_line)