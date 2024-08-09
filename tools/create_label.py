import pandas as pd


def create_label(src_csv_path='', dst_csv_path=''):
    with open(src_csv_path, 'r') as src_csv:
        next(src_csv)
        lines = src_csv.readlines()
        data_txt = []
        with open(dst_csv_path, 'w') as dst_csv:
            for line in lines:
                x_min = int(line.split(',')[4])
                y_min = int(line.split(',')[5])
                x_max = int(line.split(',')[6])
                y_max = int(line.split(',')[7])

                x = str((x_min + x_max) // 2)
                y = str((y_min + y_max) // 2)

                # new_line = line.split(',')[0] + ' ' + line.split(',')[1] + ' ' + line.split(',')[2] + ' ' + \
                #            line.split(',')[3] + ' ' + x + ' ' + y
                input_image = line.split(',')[0]
                ouput_image = input_image.replace('.png','.jpg')
                data_txt.append([ouput_image, line.split(',')[1], line.split(',')[2], line.split(',')[1] + '_' + line.split(',')[2], line.split(',')[3], x, y])

        data_txtDF = pd.DataFrame(data_txt, columns=['image', 'img_id', 'hrrp_id', 'id', 'cls', 'x', 'y'],
                                  index=None)
        data_txtDF.to_csv(dst_csv_path, index=False)


if __name__ == '__main__':
    create_label('/media/lll/One Touch/dd/4/VOCdevkit/VOC2007/label_1.csv', '../test_data/label.csv')
