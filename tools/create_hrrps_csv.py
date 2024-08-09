import os
import argparse
import pandas as pd


'''
    利用label.txt生成.csv格式的标注文件
'''


def txt2csv(parser_data):
    files = []
    with open(os.path.join(parser_data.src_path, 'label.txt'), 'r') as txt:
        lines = txt.readlines()
        for line in lines:
            file_name = line.split(' ')[1].split('.')[0]
            label = line.split(' ')[2]
            files.append([file_name, label])

    file_df = pd.DataFrame(files, columns=['file', 'class'], index=None)
    file_df.to_csv(os.path.join(parser_data.tar_path, 'label.csv'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # 设置源路径
    parser.add_argument('--src_path', default='../datasets/data_all/hrrp_type_2', help='source path')
    # 设置目标路径
    parser.add_argument('--tar_path', default='../datasets/data_for_RECOG', help='source path')

    args = parser.parse_args()
    print(args)

    txt2csv(args)