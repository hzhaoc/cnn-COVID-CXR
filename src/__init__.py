#!/usr/bin/env python
# https://stackoverflow.com/questions/2429511/why-do-people-write-usr-bin-env-python-on-the-first-line-of-a-python-script

"""
__init__.py: global declarations, like file pathes
--------------
etl.py: do ETL on source images
Source 0: https://github.com/ieee8023/covid-chestxray-dataset
Source 1: https://github.com/agchung/Figure1-COVID-chestxray-dataset
Source 2: https://github.com/agchung/Actualmed-COVID-chestxray-dataset
Source 3: https://www.kaggle.com/tawsifurrahman/covid19-radiography-database
Source 4: https://www.kaggle.com/c/rsna-pneumonia-detection-challenge
--------------
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
from datetime import datetime


discard = [f'COVID-19({x})' for x in ['100', '101', '102', '103', '104', '105', 
                                      '110', '111', '112', '113', '122', '123', 
                                      '124', '125', '126', '217']]
# input parameters
params = yaml.safe_load(open("params.yaml", 'r'))
labelmap = class2index(['covid', 'normal', 'pneumonia'])
labelmap_inv = {v: k for k, v in labelmap.items()}
# to save data
CACHE_PATH = './.pkls/'
FEATURE_PATH = './data/feature/'
OUTPUT_PATH = './output/'
DIAG_PATH = './diagnosis/'
MODEL_PATH = './model/'
TRAIN_PATH = os.path.join(FEATURE_PATH, 'train')
TEST_PATH = os.path.join(FEATURE_PATH, 'test')
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