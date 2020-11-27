#!/usr/bin/env python

"""
transform.py: image transform prior to or during tensorflow/pytorch model training 
"""

__author__ = "Hua Zhao"

import torchvision.transforms as transforms
from tensorflow.keras.preprocessing.image import ImageDataGenerator


#  tensorflow augumentation
_tf_augmentation_transform = ImageDataGenerator(
    featurewise_center=False,
    featurewise_std_normalization=False,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    brightness_range=(0.9, 1.1),
    zoom_range=(0.85, 1.15),
    fill_mode='constant',
    cval=0.,
)

# pytorch augumentation, no need to use transforms.Normalize for [TResNet], 
# see https://github.com/mrT23/TResNet/issues/5#issuecomment-608440989
_pytorch_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(hue=.1, saturation=.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor()]
)