#!/usr/bin/env python
# https://stackoverflow.com/questions/2429511/why-do-people-write-usr-bin-env-python-on-the-first-line-of-a-python-script

"""
utils.py: utility function storage
"""

__author__ = "Hua Zhao"

import time


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
    

def class2index(classes):
    """
    to make sure class-index map is the SAME as what torchvision.datasets.folder does, 
    so in thie program in torch model and tensorflow model the map is the SAME
    for details about how torchvision.datasets.folder map class, refer to:
    https://github.com/pytorch/vision/blob/4ec38d496db69833eb0a6f144ebbd6f751cd3912/torchvision/datasets/folder.py#L57
    check '_find_class()' class method
    """
    return {classes[i]: i for i in range(len(sorted(classes)))}