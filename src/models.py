#!/usr/bin/env python
# https://stackoverflow.com/questions/2429511/why-do-people-write-usr-bin-env-python-on-the-first-line-of-a-python-script

"""
models.py: custom pytorch models
"""

__author__ = "Hua Zhao"

import torchvision as tv


_models = {'resnet50': tv.models.resnet50, 
			'resnet18': tv.models.resnet18, 
			'vgg19': tv.models.vgg19, 
			'vgg11': tv.models.vgg11
			}


def pytorch_model(architect='resnet18'):
    model = _models.get(architect)(pretrained=params['model']['torch']['transfer_learning'])
    if model is None:
        raise ValueError(f'wrong model architect {architect}')
    # last layer output classe number is set to 3 obviously
    if 'resnet' in architect.lower():
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, len(labelmap))
    if 'vgg' in architect.lower():
        num_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_ftrs, len(labelmap))
    return model