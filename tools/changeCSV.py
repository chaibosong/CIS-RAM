import csv
input_file = '/home/lll/software/fixed/MMDetector/test_data/3_4/label.csv'
output_file = '/test_data/3_4/label.csv'
with open(input_file,'r') as file:
    reader = csv.reader(file)
    rows = list(reader)

for i in range(len(rows)):
    row = rows[i]
    row[0] = '0' + row[0]
    row[1] = '0' + row[1]
    row[2] = str(19994 + int(row[2]))
    row[3] = row[1] + '_' +row[2]
with open(output_file,'w',newline='') as file:
    writer = csv.writer(file)
    writer.writerows(rows)

print('end')