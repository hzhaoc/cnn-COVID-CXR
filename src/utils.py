#!/usr/bin/env python
# https://stackoverflow.com/questions/2429511/why-do-people-write-usr-bin-env-python-on-the-first-line-of-a-python-script

"""
utils.py: utility function storage
"""

__author__ = "Hua Zhao"

import cv2
import numpy as np
import pydicom as dicom
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time
import torch
import torchvision.transforms as transforms
import torchvision


def crop_top(img, percent=0.15):
    offset = int(img.shape[0] * percent)
    return img[offset:]


def central_crop(img):
    size = min(img.shape[0], img.shape[1])
    offset_h = int((img.shape[0] - size) / 2)
    offset_w = int((img.shape[1] - size) / 2)
    return img[offset_h:offset_h + size, offset_w:offset_w + size]


def process_image_file(fn, top_percent, size):
    """process imgage file (.png, .jpeg, .jpg, .dcm) in to size*size*3 array"""
    if fn[-3:] == 'dcm':  # .dcm
        ds = dicom.dcmread(fn)
        img = ds.pixel_array
        img = cv2.merge((img,img,img))  # most CXR images are gray scale (3 depths have very similar or same values); checked
    else:  # .png., .jpeg, .jpg
        img = cv2.imread(fn)
    img = crop_top(img, percent=top_percent)
    img = central_crop(img)
    img = cv2.resize(img, (size, size))
    return img


def torch_make_weights_for_balanced_classes(images, nclasses):                        
    count = [0] * nclasses                                                      
    for item in images:                                                         
        count[item[1]] += 1                                                     
    weight_per_class = [0.] * nclasses                                      
    N = float(sum(count))                                                   
    for i in range(nclasses):                                                   
        weight_per_class[i] = N/float(count[i])                                 
    weight = [0] * len(images)                                              
    for idx, val in enumerate(images):                                          
        weight[idx] = weight_per_class[val[1]]                                  
    return weight 


def random_ratio_resize(img, prob=0.3, delta=0.1):
    if np.random.rand() >= prob:
        return img
    ratio = img.shape[0] / img.shape[1]
    ratio = np.random.uniform(max(ratio - delta, 0.01), ratio + delta)

    if ratio * img.shape[1] <= img.shape[1]:
        size = (int(img.shape[1] * ratio), img.shape[1])
    else:
        size = (img.shape[0], int(img.shape[0] / ratio))

    dh = img.shape[0] - size[1]
    top, bot = dh // 2, dh - dh // 2
    dw = img.shape[1] - size[0]
    left, right = dw // 2, dw - dw // 2

    if size[0] > 480 or size[1] > 480:
        print(img.shape, size, ratio)

    img = cv2.resize(img, size)
    img = cv2.copyMakeBorder(img, top, bot, left, right, cv2.BORDER_CONSTANT,
                             (0, 0, 0))

    # if img.shape[0] != 480 or img.shape[1] != 480:
    #     raise ValueError(img.shape, size)
    return img


def timmer(flagtime=True, flagname=True, flagdoc=True):
    def wrapper(f):
        def inner(*args, **kwargs):
            # print(f"executing function {f.__name__}\n")

            if flagname:
                print(f"function in execution: {f.__name__}\n")

            if flagdoc:
                print(f"         document: {f.__doc__}\n")

            if flagtime:
                t0 = time.time()
                ret = f(*args, **kwargs)
                t1 = time.time()
                print(f"         execution time: {round(t1-t0, 2)} s\n")
            else:
                ret = f(*args, **kwargs)

            return ret
        return inner
    return wrapper


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_batch_accuracy(output, target):
    """Computes the accuracy for a batch"""
    with torch.no_grad():

        batch_size = target.size(0)
        _, pred = output.max(1)
        correct = pred.eq(target).sum()

        return correct * 100.0 / batch_size