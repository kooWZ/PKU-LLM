import sys

import numpy as np
import torchvision.transforms as T
from mask_utils import *
from utils import *


def mask_image(img, level=None, position="rand", mask_type="r", mask_size=(200, 200)):
    img_size = img.size  # width, height
    new_mask_size = [
        min(mask_size[sz], (0.3 * img_size[sz])) for sz in range(len(mask_size))
    ]
    position = load_position(position, img_size, new_mask_size)
    new_image = generate_mask_pil(img, mask_type, new_mask_size, position)
    return new_image


def blur_image(img, level=5):
    k1 = sample_odd_level(level)
    k2 = sample_odd_level(level)
    transform = T.GaussianBlur(kernel_size=(k1, k2))
    new_image = transform(img)
    return new_image


def flip_image(img, level=1.0):
    p = sample_float_level(level)
    transform = T.RandomHorizontalFlip(p=p)
    new_image = transform(img)
    return new_image


def vflip_image(img, level=1.0):
    p = sample_float_level(level)
    transform = T.RandomVerticalFlip(p=p)
    new_image = transform(img)
    return new_image


def resize_crop_image(img, level=500):
    size = img.size
    s1 = int(max(sample_int_level(level), 0.8 * size[0]))
    s2 = int(max(sample_int_level(level), 0.8 * size[1]))
    transform = T.RandomResizedCrop((s2, s1), scale=(0.9, 1))
    new_image = transform(img)
    return new_image


def gray_image(img, level=1):
    rate = sample_float_level(level)
    if rate >= 0.5:
        transform = T.Grayscale(num_output_channels=len(img.split()))
        new_image = transform(img)
    else:
        new_image = img
    return new_image


def rotation_image(img, level=180):
    rate = sample_float_level(level)
    transform = T.RandomRotation(degrees=(0, rate))
    new_image = transform(img)
    return new_image


def colorjitter_image(img, level1=1, level2=0.5):
    rate1 = sample_float_level(level1)
    rate2 = sample_float_level(level2)
    transform = T.ColorJitter(brightness=rate1, hue=rate2)
    new_image = transform(img)
    return new_image


def solarize_image(img, level=200):
    rate = sample_float_level(level)
    transform = T.RandomSolarize(threshold=rate)
    new_image = transform(img)
    return new_image


def posterize_image(img, level=3):
    rate = sample_int_level(level)
    transform = T.RandomPosterize(bits=rate)
    new_image = transform(img)
    return new_image


def policy_aug_image(img, level="0.34-0.45-0.21", pool="RR-BL-RP"):
    mutator_list = [img_aug_dict[_mut] for _mut in pool.split("-")]
    probability_list = [float(_value) for _value in level.split("-")]
    probability_list = [sum(probability_list[:i]) for i in range(len(level))]
    randnum = np.random.random()
    index = find_index(probability_list, randnum)

    new_image = mutator_list[index](img)
    return new_image


img_aug = [
    mask_image,
    blur_image,
    flip_image,
    resize_crop_image,
    gray_image,
    rotation_image,
    colorjitter_image,
    vflip_image,
    solarize_image,
    posterize_image,
]
img_aug_dict = {
    "RM": mask_image,
    "BL": blur_image,
    "HF": flip_image,
    "CR": resize_crop_image,
    "GR": gray_image,
    "RR": rotation_image,
    "CJ": colorjitter_image,
    "VF": vflip_image,
    "RS": solarize_image,
    "RP": posterize_image,
    "PL": policy_aug_image,
}
