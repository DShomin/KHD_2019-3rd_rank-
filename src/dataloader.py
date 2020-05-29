import cv2
import os
import numpy as np
import pathlib
import pandas as pd
import pickle as pkl
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from tqdm import tqdm
from nsml import DATASET_PATH

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


class TestDataset(Dataset):
    def __init__(self, image_array, transform=None):
        self.image_array = image_array
        self.transform = transform


    def __len__(self):
        return len(self.image_array)

    def __getitem__(self, idx):

        # img_name = os.path.join(self.image_dir , str(self.meta_data['package_id'].iloc[idx]) , str(self.meta_data['sticker_id'].iloc[idx]) + '.png')
        # png = Image.open(img_name).convert('RGBA')
        # png.load() # required for png.split()

        # new_img = Image.new("RGB", png.size, (255, 255, 255))
        # new_img.paste(png, mask=png.split()[3]) # 3 is the alpha channel

        image = self.image_array[idx]
        image = clahe(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)
        
        return image


class Eyes_Dataset(Dataset):
    def __init__(self, df, transform1=None, transform2=None):
        self.df = df
        self.transform1 = transform1
        self.transform2 = transform2

        self.image_list = []

        for path in self.df['path']:
            image = cv2.imread(path)
            image = clahe((cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
            image = Image.fromarray(image)
            image = self.transform1(image)
            self.image_list.append(image)


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        # img_path = self.df.iloc[idx, 0]
        # print(img_path)
        # image = cv2.imread(img_path)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = Image.fromarray(image)

        # if self.transform:
        #     image = self.transform(image)
        image = self.image_list[idx]
        
        if self.transform2:
            image = self.transform2(image)

        label = self.df['label'][idx]
        
        return image, label
        

def make_loader(df, transforms1, transforms2, batch_size=256, num_workers=4):

    dataset = Eyes_Dataset(df, transforms1, transforms2)

    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=True, 
                        num_workers=num_workers, 
                        pin_memory=True)

    return loader



