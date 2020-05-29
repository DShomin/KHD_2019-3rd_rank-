import os
import gc
import cv2
from PIL import Image

import math
import numpy as np
import pandas as pd
import time
import argparse 

import torch
from torch import nn, cuda
import torch.nn.functional as F

from model import Baseline, Resnet18, Resnet50, Resnext50, Resnext101
import torchvision.models as models
from dataloader import make_loader, TestDataset
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transforms import get_transform

from utils import count_parameters, seed_everything, AdamW, CosineAnnealingWithRestartsLR
from torchvision.transforms import ColorJitter

import nsml
from nsml import DATASET_PATH

def model_infer(data):

    batch_size = 1
    num_workers = 4
    target_size = (384, 384)

    # brightness = 0.05
    # contrast = 0.05
    test_transforms = transforms.Compose([
                transforms.CenterCrop(2000), 
                transforms.Resize(target_size),
                transforms.RandomHorizontalFlip(),
                # ColorJitter(
                #     brightness=brightness,
                #     contrast=contrast,
                #     saturation=0,
                #     hue=0,
                # ),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    test_dataset = TestDataset(data, test_transforms)
    test_loader = DataLoader(test_dataset,
                                batch_size=batch_size,
                                shuffle=False, 
                                num_workers=num_workers, 
                                pin_memory=True)

    load_list = (('team012/KHD2019_FUNDUS/307', 'best_acc_4'),
    # ('team012/KHD2019_FUNDUS/321', 'best_acc_4')
    )
    
    last_pred = []
    for load_session, load_checkpoint in load_list:
        try:
            nsml.load(checkpoint=load_checkpoint, session=load_session)
        except:
            print('load cancel')
        model.to(device)
        model.eval()

        TTA_result = []
        TTA = 3
        for _ in range(TTA):
            preds = np.zeros((len(test_loader.dataset), args.num_classes))
            with torch.no_grad():
                for i, image in enumerate(test_loader):
                    image = image.to(device)
                    output = model(image) # output shape (batch_num, num_classes)
                    
                    preds[i*batch_size: (i+1)*batch_size] = output.detach().cpu().numpy()
            TTA_result.append(preds)
            
        last_pred.append(np.mean(TTA_result, axis=0))
    last_pred = np.mean(last_pred, axis=0)
    predictions = np.argmax(last_pred, axis=1)
    print(predictions)
    return predictions

def bind_model(model):

    def save(dir_name, **kwargs):
        state = {
            'model': model.state_dict(),
        }
        torch.save(state, os.path.join(dir_name, 'model.pt'))

    def load(dir_name):
        state = torch.load(os.path.join(dir_name, 'model.pt'))
        model.load_state_dict(state['model'])
        print('Model loaded')

    def infer(data):

        predictions  = model_infer(data) 
        print(predictions)
        return predictions

    nsml.bind(save=save, load=load, infer=infer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parser')
    arg = parser.add_argument

    # reserved for nsml
    arg("--mode", type=str, default="train")
    arg("--iteration", type=str, default='0')
    arg("--pause", type=int, default=0)

    arg('--num_classes', type=int, default=4, help='num tags to predict') # Fixed
    arg('--model', type=str, default='Resnet50')
    arg('--input_size', type=int, default=296)
    arg('--test_augments', default='resize, horizontal_flip', type=str)
    arg('--augment_ratio', default=0.5, type=float, help='probability of implementing transforms')
    arg('--device', type=int, default=0)
    arg('--hidden_size', type=int, default=128)
    args = parser.parse_args()

    device = args.device
    use_gpu = cuda.is_available()

    SEED = 2019
    seed_everything(SEED)

    global model
    model = models.densenet201(pretrained=False)
    model.classifier = nn.Linear(1920, args.num_classes)
    bind_model(model)
    if args.mode == 'train':
        nsml.save('last')

    if use_gpu:
        model = model.to(device)

    
    if args.pause:
        nsml.paused(scope=locals())