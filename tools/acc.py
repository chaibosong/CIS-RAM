import os


def acc(label_path, result_path):
    cnt = 0
    with open(label_path, 'r') as labels:
        next(labels)
        lab_lines = labels.readlines()

        with open(result_path, 'r') as results:
            next(results)
            res_lines = results.readlines()
            for res_line in res_lines:
                file_name1, cls1, prob1 = res_line.split(' ')
                # print(file_name1)
                for lab_line in lab_lines:
                    file_name2 = lab_line.split(',')[1]
                    cls2 = lab_line.split(',')[2]
                    # print(file_name2)

                    # if file_name2 == file_name1:
                    #     print(file_name2)
                    #
                    if file_name1 == file_name2 and cls1 == cls2:
                        cnt += 1

    return cnt


if __name__ == '__main__':
    label_path = '../test_data/new_data/test.csv'
    result_path = '../results/results.txt'

    cnt = acc(label_path, result_path)
    print(cnt / 268)