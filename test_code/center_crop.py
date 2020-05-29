import cv2
import os

from glob import glob

def cen_crop_img(path, margine, size):
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    w, h, _ = img.shape
    x = w // 2
    y = h // 2
    croped_img = img[x - margine // 2 : x + margine // 2, y - margine // 2  : y + margine // 2 ]
    cv2.resize(croped_img, dsize=size, interpolation=cv2.INTER_AREA)
    cv2.imwrite(path, croped_img)
    
    