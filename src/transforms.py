import random
import math

from PIL import Image, ImageOps
from torchvision.transforms import Compose, ToTensor, Normalize, RandomResizedCrop, RandomApply, Resize, CenterCrop, RandomAffine
from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip, ColorJitter, RandomGrayscale, RandomRotation


class RandomErase(object):
    def __init__(self, prob, sl, sh, r):
        self.prob = prob
        self.sl = sl
        self.sh = sh
        self.r = r

    def __call__(self, img):
        if random.uniform(0, 1) > self.prob:
            return img

        while True:
            area = random.uniform(self.sl, self.sh) * img.size[0] * img.size[1]
            ratio = random.uniform(self.r, 1/self.r)

            h = int(round(math.sqrt(area * ratio)))
            w = int(round(math.sqrt(area / ratio)))

            if h < img.size[0] and w < img.size[1]:
                x = random.randint(0, img.size[0] - h)
                y = random.randint(0, img.size[1] - w)
                img = np.array(img)
                if len(img.shape) == 3:
                    for c in range(img.shape[2]):
                        img[x:x+h, y:y+w, c] = random.uniform(0, 1)
                else:
                    img[x:x+h, y:y+w] = random.uniform(0, 1)
                img = Image.fromarray(img)

                return img


class KeepAsepctResize:
    def __init__(self, target_size=(288, 288)):
        self.target_size = target_size
    
    def __call__(self, img):
        width, height = img.size
        long_side = max(width, height)
        
        delta_w = long_side - width
        delta_h = long_side - height
        padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
        img = ImageOps.expand(img, padding)
        img = img.resize(self.target_size)
        return img
        


def get_transform(
        target_size=(256, 256),
        transform_list='horizontal_flip', # random_crop | keep_aspect
        augment_ratio=0.5,
        is_train=True,
        ):
    transform = list()
    transform_list = transform_list.split(', ')
    augments = list()

    for transform_name in transform_list:
        if transform_name == 'random_crop':
            scale = (0.6, 1.0) if is_train else (0.8, 1.0)
            transform.append(RandomResizedCrop(target_size, scale=(0.8, 1.0)))
        # elif transform_name == 'resize':
        #     transform.append(Resize(target_size))
        elif transform_name == 'keep_aspect':
            transform.append(KeepAsepctResize(target_size))


        # elif transform_name == 'random_erase':
        #     transform.append(
        #         RandomErase(
        #             prob=args.random_erase_prob if args.random_erase else 0,
        #             sl=args.random_erase_sl,
        #             sh=args.random_erase_sh,
        #             r=args.random_erase_r)
        #     )
        # random_erase', default=False
        # random_erase_prob', default=0.5)
        # random_erase_sl', default=0.02)
        # random_erase_sh', default=0.4)
        # random_erase_r', default=0.3)

        elif transform_name == 'Affine':
            augments.append(RandomAffine(degrees=(-180, 180),
                                         scale=(0.8889, 1.0),
                                         shear=(-36, 36)))
        elif transform_name == 'centor_crop':
            augments.append(CenterCrop(target_size))
        elif transform_name == 'horizontal_flip':
            augments.append(RandomHorizontalFlip())
        elif transform_name == 'vertical_flip':
            augments.append(RandomVerticalFlip())
        elif transform == 'random_grayscale':
            p = 0.5 if is_train else 0.25
            transform.append(RandomGrayscale(p))
        elif transform_name == 'random_rotate':
            augments.append(RandomRotation(15))
        elif transform_name == 'color_jitter':
            brightness = 0.1 if is_train else 0.05
            contrast = 0.1 if is_train else 0.05
            augments.append(ColorJitter(
                brightness=brightness,
                contrast=contrast,
                saturation=0,
                hue=0,
            ))
    transform.append(RandomApply(augments, p=augment_ratio))   
    transform.append(ToTensor())
    transform.append(Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return Compose(transform)
