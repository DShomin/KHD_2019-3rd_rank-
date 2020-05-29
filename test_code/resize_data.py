import os
import pickle
import cv2
import pandas as pd
import numpy as np

import nsml

from nsml import DATASET_PATH
from glob import glob

def cen_crop_img(path, margine, size):
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    w, h, _ = img.shape
    x = w // 2
    y = h // 2
    croped_img = img[x - margine // 2 : x + margine // 2, y - margine // 2  : y + margine // 2 ]
    croped_img = cv2.resize(croped_img, dsize=size, interpolation=cv2.INTER_AREA)
    print(croped_img.shape)

    return croped_img

def make_csv(folder_path):
    folder_list = os.listdir(folder_path)

    result_df = pd.DataFrame(columns=['path', 'label'])
    for folder in folder_list:
        file_list = glob(folder_path + '/' + folder + '/*jpg')

        for file in file_list:
            result_df = result_df.append(pd.DataFrame([(file, folder)], columns=['path', 'label']), ignore_index=True)

    result_df['LR'] = result_df.path.apply(lambda x : x.split('/')[-1].split('.')[0][-1])

    result_df['LR_label_com'] = result_df['LR'].astype(str) + '_' + result_df['label'].astype(str)
    return result_df

def bind_data(array):
    def data_save(dir_name):

        os.makedirs(dir_name, exist_ok=True)
        with open(os.path.join(dir_name, 'class.pkl'), 'wb') as fp:
            pickle.dump(array, fp) # array는 이 함수내에서 참조가 가능해야함 예를들어 global로 선언한 경우.
        print('crop resize  model saved!')

    def data_load(dir_name):
        with open(os.path.join(dir_name, 'class.pkl'), 'rb') as fp:
            array = pickle.load(fp) # 여기서 마찬가지로 array는 global변수같은걸로 선언해서 값을 덮어씌워줘야 합니다.
        print('dataset loaded!')
    nsml.bind(save=data_save, load=data_load)

def bind_label(array):
    def label_save(dir_name):

        os.makedirs(dir_name, exist_ok=True)
        with open(os.path.join(dir_name, 'class.pkl'), 'wb') as fp:
            pickle.dump(array, fp) # array는 이 함수내에서 참조가 가능해야함 예를들어 global로 선언한 경우.
        print('crop resize  model saved!')

    def label_load(dir_name):
        with open(os.path.join(dir_name, 'class.pkl'), 'rb') as fp:
            array = pickle.load(fp) # 여기서 마찬가지로 array는 global변수같은걸로 선언해서 값을 덮어씌워줘야 합니다.
        print('dataset loaded!')
    nsml.bind(save=label_save, load=label_load)

def clahe(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    if img.dtype != np.uint8:
        raise TypeError("clahe supports only uint8 inputs")

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    if len(img.shape) == 2:
        img = clahe.apply(img)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        img[:, :, 0] = clahe.apply(img[:, :, 0])
        img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)

    return img

def image_preprocessing(im, rescale, resize_factor):
    ## 이미지 크기 조정 및 픽셀 범위 재설정
    im = im[400:3500, 300:2700]

    im = clahe(im)

    # h, w, c = 3072, 3900, 3
    h, w, c = 2400, 3100, 3
    nh, nw = int(h//resize_factor), int(w//resize_factor)
    # print(im.shape)

    res = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_AREA)

    if rescale == True:
        res = res / 255.

    return res
def Label2Class(label):     # one hot encoding (0-3 --> [., ., ., .])

    resvec = [0, 0, 0, 0]
    if label == 'AMD':      cls = 1;    resvec[cls] = 1
    elif label == 'RVO':   cls = 2;    resvec[cls] = 1
    elif label == 'DMR':   cls = 3;    resvec[cls] = 1
    else:               cls = 0;    resvec[cls] = 1      # Normal

    return resvec

if __name__ == '__main__':

    global array

    img_list = []
    df = make_csv(DATASET_PATH + '/train/train_data')
    df['label'] = df.label.map(Label2Class)
    
    for path in df['path']:
        # crop_resize_img = cen_crop_img(path, 2000, (224, 224))
        
        im = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        im = image_preprocessing(im, rescale=True, resize_factor=10)
        
        img_list.append(im)
    array = np.array(img_list)
    # print('dataset array', array.shape)
    print('data shape : ', array.shape)
    bind_data(array)
    nsml.save('save data')

    label_list = []
    for label in df['label']:
        # crop_resize_img = cen_crop_img(path, 2000, (224, 224))
        label_list.append(np.array(label))
    array = np.array(label_list)
    print('label shape : ', array.shape)
    bind_label(array)
    nsml.save('save label')