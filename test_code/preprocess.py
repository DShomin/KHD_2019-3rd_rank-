import pandas as pd
import numpy as np
import os
import cv2


from pathlib import Path
from glob import glob

def image_preprocessing(img_path, resize):
    ## 이미지 크기 조정 및 픽셀 범위 재설정
    # current image size h, w, c = 3900, 3072, 3
    nh, nw = resize
    # print(im.shape)
    im = cv2.imread(img_path)
    res = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_AREA)
    return res

def preprocess(data_path, output_path, resize):
    '''
    data들을 preprocess와 label csv를 만들어주는 함수
    current dataset path : 예제 path/class1/img_1
    data_path : 위에서의 path를 의미
    resize : (h, w) is must be a tupple
    '''
    # create output directory 
    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    # read label list
    path_file_list = glob(data_path + '/*')
    label_df = pd.DataFrame(columns=['file', 'label'])
    for dir in path_file_list:
        if os.path.isdir(dir):
            label = dir.split('/')[-1].split('\\')[-1]
            print(label)
            file_list = glob(dir + '/*.jpg')

            for file in file_list:
                img = image_preprocessing(file, resize)
                file_name = label + '_' + file.split('\\')[-1]
                cv2.imwrite(output_path + '/' + file_name, img)
                label_df = label_df.append(pd.DataFrame([(file_name, label)], columns=['file', 'label']), ignore_index=True)
    label_df.to_csv(output_path + '/label.csv', index=False)