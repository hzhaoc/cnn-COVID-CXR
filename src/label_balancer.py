#!/usr/bin/env python
# https://stackoverflow.com/questions/2429511/why-do-people-write-usr-bin-env-python-on-the-first-line-of-a-python-script

"""
batch_generator.py: generate tensorflow tensor batches from images for model
                    original: https://github.com/lindawangg/COVID-Net
"""

__author__ = "Hua Zhao"

from src.etl import *
import tensorflow.compat.v1 as tf  # version 2.3.1
"""
NOTICE: 
tensorflow default builds DO NOT include CPU instructions that fasten matrix computation including avx, avx2, etc,.
see:
(https://stackoverflow.com/questions/47068709/your-cpu-supports-instructions-that-this-tensorflow-binary-was-not-compiled-to-u)
to solve this, download tailored wheel from:
(https://github.com/fo40225/tensorflow-windows-wheel/tree/master/2.1.0/py37/CPU%2BGPU/cuda102cudnn76avx2)
then isntall the package with
'pip install --ignore-installed --upgrade /path/target.whl'
"""
from tensorflow import keras
import cv2
from src.transform import Augmentator


class TFBalancedCovidBatch(keras.utils.Sequence):
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
        META = pickle.load(open(os.path.join(CACHE_PATH, 'meta', 'meta'), 'rb'))
        if self.is_training:
            META = META[META.train==1]
        else:
            META = META[META.train!=1]
        self.meta_noncovid = META[META.label!=labelmap['covid']]
        self.meta_covid = META[META.label==labelmap['covid']]
        self.n_covid = len(self.meta_covid)
        self.n_class = len(labelmap)
        self.augmentator = Augmentator(in_channel=3)

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
        batch_x = np.zeros((len(fns), params['etl']['image_size'], params['etl']['image_size'], 3))
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
                x = self.augmentator.tf_augmentate(x)
            
            x = x.astype('float32') / 255.0  # RGB ~ [0, 255], normalize
            
            batch_x[i] = x
            batch_y[i] = label

        weights = np.array([self.class_weights[labelmap_inv[i]] for i in batch_y])  # class weights

        return batch_x, keras.utils.to_categorical(batch_y, num_classes=self.n_class), weights


def pytorch_balanced_covid_samples(images, nclasses, class_map, covid_weight=0.3):
    """
    for the torch model, balance sample weights for COVID class, the weight is a hyperparameter from input, 
    making the sampling same as or close to how tensorflow model weight its samples from tf_batch_generator.py
    which is weighting COVID class, while keeping other two: normal and pneumonia relatively same weight
    ----------------------------------------
    @parameter: 
    :images:
        list of tuples (image path, class index) where class index are 0-indexed integers mapped from sorted classes, 
        see src.utils.class2index
        or https://github.com/pytorch/vision/blob/4ec38d496db69833eb0a6f144ebbd6f751cd3912/torchvision/datasets/folder.py#L57 at _find_classes class method
    @return:
    :weight:
        list
        weights of each sample
        no need to sum up to 1
        see https://pytorch.org/docs/stable/_modules/torch/utils/data/sampler.html#WeightedRandomSampler
    ----------------------------------------
    """
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1          
    balanced_weight_per_class = [0.] * nclasses
    original_weight_per_class = [0.] * nclasses
    N = float(sum(count))
    
    original_covid_weight = count[class_map['covid']]/N
    for i in range(nclasses):
        original_weight_per_class[i] = float(count[i])/N
        if class_map['covid'] == i:
            balanced_weight_per_class[i] = covid_weight/(float(count[i])/N)
        else:
            balanced_weight_per_class[i] = (1-covid_weight)/(1-original_covid_weight)
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = balanced_weight_per_class[val[1]]
    return original_weight_per_class, balanced_weight_per_class, weight