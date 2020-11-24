#!/usr/bin/env python
# https://stackoverflow.com/questions/2429511/why-do-people-write-usr-bin-env-python-on-the-first-line-of-a-python-script

"""
utils-spark.py: rdd class
"""

__author__ = "Hua Zhao"


from src.etl import *


class CXR:
    'CXR for Spark'
    
    def __init__(
            self,
            pid,
            fn_src,
            label,
            src=-1,
            fn_dst=None,
    ):
        self.pid = pid
        self.fn_src = fn_src
        self.fn_dst = fn_dst
        self.label = label
        self.src = src
        self._proc()
    
    def _proc(self):
        if self.src < 0:
            raise ValueError('src can not be negative')
        elif self.src == 0:
            self.label = src0_label(self.label)
            self.fn_src = os.path.join(INPUT_PATH_0_IMG, self.fn_src)
        elif self.src == 1:
            self.fn_src = os.path.join(INPUT_PATH_1_IMG, src1_imgpath(self.pid))
            self.label = src1_label(self.label)
        elif self.src == 2:
            self.label = src2_label(self.label)
            self.fn_src = os.path.join(INPUT_PATH_2_IMG, self.fn_src)
        elif self.src == 3:
            self.fn_src = _spark_src3_img(self.fn_src, self.label)
        elif self.src == 4:
            self.fn_src = os.path.join(INPUT_PATH_4_IMG, f'{self.pid}.dcm')
            self.label = 'normal' if not int(self.label) else 'pneumonia'
        else:
            raise ValueError('src can not be other numbers than [0, 1, 2, 3, 4]')
    
    def isna(self, how='any'):
        if how == 'any':
            return max([i is None for i in [self.pid, self.fn_src, self.label]])
        if how == 'all':
            return min([i is None for i in [self.pid, self.fn_src, self.label]])

        
def _spark_src3_img(s, label):
    if 'covid' == label:
        fn = os.path.join(INPUT_PATH_3_0_IMG, s)
        _ = os.path.exists(fn)
    if 'normal' == label:
        fn = os.path.join(INPUT_PATH_3_1_IMG, s.replace('-', '(').replace('.', ').'))
        _ = os.path.exists(fn)
    if 'pneumonia' == label:
        fn = os.path.join(INPUT_PATH_3_2_IMG, s.replace('-', '(').replace('.', ').'))
        _ = os.path.exists(fn)
    return fn if _ else ' ('.join(fn.split('('))