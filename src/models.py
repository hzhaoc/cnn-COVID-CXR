#!/usr/bin/env python
# https://stackoverflow.com/questions/2429511/why-do-people-write-usr-bin-env-python-on-the-first-line-of-a-python-script

"""
models.py: custom pytorch models
"""

__author__ = "Hua Zhao"

from src import *
import torchvision as tv
import torch.nn as nn


_models = {'resnet50': tv.models.resnet50, 
			'resnet18': tv.models.resnet18, 
			'vgg19': tv.models.vgg19, 
			'vgg11': tv.models.vgg11
			}


def pytorch_model(architect='resnet18', pretrained=False):
    model = _models.get(architect)(pretrained=pretrained)
    if model is None:
        raise ValueError(f'wrong model architect {architect}')
    if 'resnet' in architect.lower():
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, len(labelmap))  # last layer output classe number is set to 3 obviously
        if params['model']['torch']['in_channel'] == 1:  # default 3
            model.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    if 'vgg' in architect.lower():
        num_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_ftrs, len(labelmap))
        if params['model']['torch']['in_channel'] == 1:  # default 3
            model.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))  # last layer output classe number is set to 3 obviously
    return model