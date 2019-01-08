import numpy as np
import csv
import math
import os
import shutil

def get_imginfo(filepath):
    csv_file = open(filepath, "r", encoding='UTF-8')
    reader = csv.reader(csv_file)

    img_name = []
    # img_label = []

    i = 0
    for item in reader:
        # 忽略第一行
        if reader.line_num == 1:
            continue
        i = i+1
        img_name.append((item[0]+'.png'))
        if not os.path.exists(('img_train/' + item[1])):
            os.mkdir(('img_train/' + item[1]))
        shutil.move(('img_train/' + item[0] + '.png'), ('img_train/' + item[1] + '/' + item[0] + '.png'))
    print(i)
    csv_file.close()
    
if __name__ == '__main__':
    get_imginfo("train.csv")