import os
import sys
from shutil import copyfile
import csv

base_path = '/home/voqtuyen/Documents/'
in_dir = 'Asian_crop'
out_dir = 'data'

in_path = os.path.join(base_path, in_dir)
out_path = os.path.join(base_path, out_dir)

FILES = os.listdir(in_path)

input_image_filename_list = [i for i in FILES if i.endswith('.jpg')]
with open('age_gender_utk.csv', 'w') as f:
    writer = csv.writer(f)

    for path in input_image_filename_list:
        age,gender,race,rest = path.split('_')
        name,_,_,ext = rest.split('.')
        file_tmp = os.path.join(in_path, path)
        file_name = name + '.' + ext
        print(path)
        lbl = ''
    # 0 = Male
        if gender == '0':
            if int(age) >= 0 and int(age) < 10:
                lbl = '10100000000'
            elif int(age) >= 10 and int(age) < 19:
                lbl = '10010000000'
            elif int(age) >= 20 and int(age) < 29:
                lbl = '10001000000'
            elif int(age) >= 30 and int(age) < 39:
                lbl = '10000100000'
            elif int(age) >= 40 and int(age) < 49:
                lbl = '10000010000'
            elif int(age) >= 50 and int(age) < 59:
                lbl = '10000001000'
            elif int(age) >= 60 and int(age) < 69:
                lbl = '10000000100'
            elif int(age) >= 70 and int(age) < 79:
                lbl = '10000000010'
            else:
                lbl = '10000000001'
        # 1 = Female
        else:
            if int(age) >= 0 and int(age) < 10:
                lbl = '01100000000'
            elif int(age) >= 10 and int(age) < 19:
                lbl = '01010000000'
            elif int(age) >= 20 and int(age) < 29:
                lbl = '01001000000'
            elif int(age) >= 30 and int(age) < 39:
                lbl = '01000100000'
            elif int(age) >= 40 and int(age) < 49:
                lbl = '01000010000'
            elif int(age) >= 50 and int(age) < 59:
                lbl = '01000001000'
            elif int(age) >= 60 and int(age) < 69:
                lbl = '0000000100'
            elif int(age) >= 70 and int(age) < 79:
                lbl = '01000000010'
            else:
                lbl = '01000000001'
        line = [lbl, 'images/' + path]
        writer.writerow(line)
f.close()