#!/usr/bin/env python
# https://stackoverflow.com/questions/2429511/why-do-people-write-usr-bin-env-python-on-the-first-line-of-a-python-script

"""
glob.py: global declarations
"""

__author__ = "Hua Zhao"

import yaml
import os
from src.utils import class2index
import numpy as np
import pandas as pd
import random 
import shutil
import pickle
from collections import defaultdict, Counter
import copy


discard = [f'COVID-19({x})' for x in ['100', '101', '102', '103', '104', '105', 
                                      '110', '111', '112', '113', '122', '123', 
                                      '124', '125', '126', '217']]
# input parameters
params = yaml.safe_load(open("params.yaml", 'r'))
labelmap = class2index(['covid', 'normal', 'pneumonia'])
labelmap_inv = {v: k for k, v in labelmap.items()}
# to save data
SAVE_PATH = './pkls/'
TRAIN_PATH = os.path.join(SAVE_PATH, 'train')
TEST_PATH = os.path.join(SAVE_PATH, 'test')

# src 0
INPUT_PATH_0_IMG = './data/source/covid-chestxray-dataset-master/images' 
INPUT_PATH_0_META = './data/source/covid-chestxray-dataset-master/metadata.csv'
# src 1
INPUT_PATH_1_IMG = './data/source/Figure1-COVID-chestxray-dataset-master/images'
INPUT_PATH_1_META = './data/source/Figure1-COVID-chestxray-dataset-master/metadata.csv'
# src 2
INPUT_PATH_2_IMG = './data/source/Actualmed-COVID-chestxray-dataset-master/images'
INPUT_PATH_2_META = './data/source/Actualmed-COVID-chestxray-dataset-master/metadata.csv'
# src 3
INPUT_PATH_3_0_IMG = './data/source/COVID-19-Radiography-Database/COVID-19'
INPUT_PATH_3_1_IMG = './data/source/COVID-19-Radiography-Database/NORMAL'
INPUT_PATH_3_2_IMG = './data/source/COVID-19-Radiography-Database/Viral Pneumonia'
INPUT_PATH_3_0_META = './data/source/COVID-19-Radiography-Database/COVID-19.metadata.csv'
INPUT_PATH_3_1_META = './data/source/COVID-19-Radiography-Database/NORMAL.metadata.csv'
INPUT_PATH_3_2_META = './data/source/COVID-19-Radiography-Database/Viral Pneumonia.matadata.csv'
# src 4
INPUT_PATH_4_META_1 = './data/source/rsna-pneumonia-detection-challenge/stage_2_detailed_class_info.csv' 
INPUT_PATH_4_META = './data/source/rsna-pneumonia-detection-challenge/stage_2_train_labels.csv' 
INPUT_PATH_4_IMG = './data/source/rsna-pneumonia-detection-challenge/stage_2_train_images'