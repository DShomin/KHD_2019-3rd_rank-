import os 
from glob import glob
import pandas as pd
import numpy as np
import cv2

from nsml import DATASET_PATH


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

def cen_crop_img(path, margine, size):
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    w, h, _ = img.shape
    x = w // 2
    y = h // 2
    croped_img = img[x - margine // 2 : x + margine // 2, y - margine // 2  : y + margine // 2 ]
    cv2.resize(croped_img, dsize=size, interpolation=cv2.INTER_AREA)
    print(croped_img.shape)
    cv2.imwrite(path, croped_img)
    

if __name__ == '__main__':
    df = make_csv(DATASET_PATH + '/train/train_data')

    for path in df['path']:
        cen_crop_img(path, 2000, (224, 224))
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        print('after image size : ', img.shape)
