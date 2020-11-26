#!/usr/bin/env python
# https://stackoverflow.com/questions/2429511/why-do-people-write-usr-bin-env-python-on-the-first-line-of-a-python-script

"""
batch_generator.py: source code to generate tensorflow tensor batches from images for model
                    original: https://github.com/lindawangg/COVID-Net
"""

__author__ = "Hua Zhao"

import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.utils import *
import copy
from src.etl import *


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


class BalancedCovidBatch(keras.utils.Sequence):
    """
    Generates COVID-label-balanced batch data for Keras, 
    weight of COVID class is a hyperparameter input
    """

    def __init__(
            self,
            is_training=True,
            batch_size=32,
            class_weights=dict(),
            batch_weight_covid=.3,
    ):
        # input
        self.is_training = is_training
        self.batch_size = batch_size
        self.batch_weight_covid = batch_weight_covid
        self.class_weights = class_weights
        # inside class
        self.n = 0  # batch index
        META = pickle.load(open(os.path.join(SAVE_PATH,  'meta'), 'rb'))
        if self.is_training:
            META = META[META.train==1]
        else:
            META = META[META.train!=1]
        self.meta_noncovid = META[META.label!=labelmap['covid']]
        self.meta_covid = META[META.label==labelmap['covid']]
        self.n_covid = len(self.meta_covid)
        self.n_class = len(labelmap)

    def __next__(self):
        # print('generating a batch..')
        # genereate one batch of data
        batch_x, batch_y, weights = self.__getitem__(self.n)
        self.n += 1
        # if reach dataset end, start over
        if self.n >= self.__len__():
            self.n = 0
        return batch_x, batch_y, weights

    def __len__(self):
        return int(np.ceil(len(self.meta_noncovid) / self.batch_size))

    def __getitem__(self, idx):
        # list of image file names, it's new data of data type list, not a pointer to the pd.Series
        fns = self.meta_noncovid.imgid.iloc[idx*self.batch_size:(idx+1)*self.batch_size].to_list()
        labels = self.meta_noncovid.label.iloc[idx*self.batch_size:(idx+1)*self.batch_size].to_list() # same order as file names 
        # batch size * length * width * depth
        batch_x = np.zeros((len(fns), params['data']['image_size'], params['data']['image_size'], 3))
        # batch size
        batch_y = np.zeros(len(fns))
        
        # upsample covid cases, first bootstrap from covid data
        _covid_size = max(int(len(fns)*self.batch_weight_covid), 1)
        covid_inds = np.random.choice(np.arange(len(fns)), size=_covid_size, replace=False)
        covid_bootstrap = np.random.choice(np.arange(self.n_covid), size=_covid_size, replace=False)
        
        # then randomly insert into and replace noncovid data in the batch
        for i, j in zip(covid_inds, covid_bootstrap):
            fns[i] = self.meta_covid.imgid.iloc[j]
            labels[i] = self.meta_covid.label.iloc[j]  # same order as file names obviously
        
        # now get [batch # * size * size * 3] image tensor, and [batch #] label tensor
        for i, (fn, label) in enumerate(zip(fns, labels)):
            x = cv2.imread(fn)
            # print('getting a sample in batch..')
            
            if self.is_training:  # augmentate
                x = augmentate(x)
            
            x = x.astype('float32') / 255.0  # RGB ~ [0, 255], normalize
            
            batch_x[i] = x
            batch_y[i] = label

        weights = np.array([self.class_weights[labelmap_inv[i]] for i in batch_y])  # class weights

        return batch_x, keras.utils.to_categorical(batch_y, num_classes=self.n_class), weights


def augmentate(img):
    """img: numpy.ndarray x*y*z"""
    img = random_ratio_resize(img)
    img = _tf_augmentation_transform.random_transform(img)
    return img


class _BalancedCovidBatch(keras.utils.Sequence):
    'Generates COVID-label-balanced batch data for Keras, this deprecated class uses img data as input, not img files'

    def __init__(
            self,
            is_training=True,
            data=None,
            batch_size=32,
            class_weights=dict(),
            batch_weight_covid=.3,
    ):
        # input
        self.data = data
        self.is_training = is_training
        self.batch_size = batch_size
        self.batch_weight_covid = batch_weight_covid
        self.class_weights = class_weights
        # inside class, but mostly form params
        self.n = 0  # batch index
        self.n_covid = len(data['covid']['label'])
        self.n_class = len(labelmap)

    def __next__(self):
        # genereate one batch of data
        batch_x, batch_y, weights = self.__getitem__(self.n)
        self.n += 1
        # if reach dataset end, start over
        if self.n >= self.__len__():
            self.n = 0
        return batch_x, batch_y, weights

    def __len__(self):
        return int(np.ceil(len(self.data['covid']['label']) / self.batch_size))

    def __getitem__(self, idx):
        # batch size * length * width * depth
        batch_x = copy.deepcopy(sellf.data['!covid']['data'][idx*self.batch_size : (idx+1)*self.batch_size])
        # batch size * label
        batch_y = copy.deepcopy(sellf.data['!covid']['label'][idx*self.batch_size : (idx+1)*self.batch_size])
        
        # upsample covid cases, first bootstrap from covid data
        _covid_size = max(int(len(batch_x) * self.batch_weight_covid), 1)
        covid_inds = np.random.choice(np.arange(len(batch_x)), size=_covid_size, replace=False)
        covid_bootstrap = np.random.choice(np.arange(self.n_covid), size=_covid_size, replace=False)
        
        # then randomly insert into and replace noncovid data in the batch
        for i, j in zip(covid_inds, covid_bootstrap):
            x = self.data['covid']['data'][j]
            y = self.data['covid']['label'][j]
                
            if self.is_training:  # augmentate
                x = augmentate(x)
            
            x = x.astype('float32') / 255.0  # RGB ~ [0, 255], normalize
            
            batch_x[i] = x
            batch_y[i] = y

        weights = [self.class_weights[labelmap_inv[i]] for i in batch_y]

        return batch_x, keras.utils.to_categorical(batch_y, num_classes=self.n_class), weights