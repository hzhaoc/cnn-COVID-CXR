#!/usr/bin/env python

"""
transform.py: image transform prior to or during tensorflow/pytorch model training 
              has Augmentator in trainig, Preprocessor before training
"""

__author__ = "Hua Zhao"

import cv2
import torchvision.transforms as transforms
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # https://github.com/tensorflow/tensorflow/issues/37649
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pydicom as dicom
import numpy as np
from PIL import Image


class ImgPreprocessor:
    """
    process an image in array to desired form
    -> (adaptive) histogram equlization
    -> crop top
    -> crop center
    -> resize
    -> [optional] segmentation
    ---------------
    @param
    :CLAHE: boolean; use adaptive or global histogram equalizaiton
    :crop_top: float; percent to crop on top
    :size: int; resize
    :(optional) clipLimit: float; contrast limiting in CLAHE
    :(optional) tileGridSize: tuple of float; tile size in CLAHE  
    ---------------
    """

    def __init__(self, CLAHE=False, crop_top=0.02, size=224, use_seg=False, **kwargs):
        self.CLAHE = CLAHE
        self.crop_top_pct = crop_top
        self.size = (size, size)
        for k, v in kwargs.items():
            setattr(self, k, v)
        if (not hasattr(self, 'clipLimit')) and self.CLAHE:
            raise KeyError('clipLimit is not set to use CLAHE')
        if (not hasattr(self, 'tileGridSize')) and self.CLAHE:
            raise KeyError('tileGridSize is not set to use CLAHE')
        # segmentate
        self.threshold = 45
        self.ker = 169
        self.use_seg = use_seg
        print(f"segmentation set to {str(self.use_seg)}")
    
    def __call__(self, img):
        """
        @param
        :img: depth 3 array
        :return: depth 1 or 3 array
        """
        # historgram equlization
        # theory: https://en.wikipedia.org/wiki/Adaptive_histogram_equalization#Contrast_Limited_AHE
        # code impl: https://docs.opencv.org/master/d6/dc7/group__imgproc__hist.html
        # code impl: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_histograms/py_histogram_equalization/py_histogram_equalization.html
        if self.CLAHE:  # adaptive, apply on RGB imge: https://stackoverflow.com/questions/25008458/how-to-apply-clahe-on-rgb-color-images
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            lab_planes = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=self.clipLimit, tileGridSize=self.tileGridSize)
            lab_planes[0] = clahe.apply(lab_planes[0])
            lab = cv2.merge(lab_planes)
            img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else: # global
            img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
            img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        # crop
        img = self.crop_top(img, self.crop_top_pct)
        img = self.central_crop(img)
        # resize
        img = cv2.resize(img, self.size)
        # segmentate
        if self.use_seg:
            img = self.segmentate(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        return img

    def __repr__(self):
        return self.__class__.__name__+'()' + self.__doc__

    def segmentate(self, img):
        """param: 1 depth array, return: 1 depth array"""
        # https://github.com/ilaiw/CXR-lung-segmentation/blob/master/CXR-seg-openCV.ipynb
        img_erased = self.eraseMax(img, draw=False)
        
        kernel = np.ones((self.ker, self.ker), np.uint8)
        blackhat = cv2.morphologyEx(img_erased, cv2.MORPH_BLACKHAT, kernel)
        
        ret, thresh = cv2.threshold(blackhat, self.threshold, 255, 0)
        
        cmask = self.get_cmask(img)
        
        mask = np.multiply(cmask, thresh).astype('uint8')
        
        median = cv2.medianBlur(mask, 23)
        
        contour_mask = self.contourMask(median).astype('uint8')
        
        return img*(contour_mask/255)

    @staticmethod
    def crop_top(img, pct):
        offset = int(img.shape[0] * pct)
        return img[offset:]

    @staticmethod
    def central_crop(img):
        size = min(img.shape[0], img.shape[1])
        offset_h = int((img.shape[0] - size) / 2)
        offset_w = int((img.shape[1] - size) / 2)
        return img[offset_h:offset_h + size, offset_w:offset_w + size]

    @staticmethod
    def eraseMax(img, eraseLineCenter=0, eraseLineWidth=30, draw=False):
        sumpix0=np.sum(img,0)
        if draw:
            plt.plot(sumpix0)
            plt.title('Sum along axis=0')
            plt.xlabel('Column number')
            plt.ylabel('Sum of column')
        max_r2=np.int_(len(sumpix0)/3)+np.argmax(sumpix0[np.int_(len(sumpix0)/3):np.int_(len(sumpix0)*2/3)])
        cv2.line(img,(max_r2+eraseLineCenter,0),(max_r2+eraseLineCenter,512),0,eraseLineWidth)
        return img

    @staticmethod
    def get_cmask(img, maxCorners=3800, qualityLevel=0.001, minDistance=1, Cradius=6):
        corners = cv2.goodFeaturesToTrack(img, maxCorners, qualityLevel, minDistance)
        corners = np.int0(corners)
        cmask = np.zeros(img.shape)
        for corner in corners:
            x,y = corner.ravel()
            cv2.circle(cmask,(x,y),Cradius,1,-1)
        return cmask

    @staticmethod
    def contourMask(image):
        im2,contours,hierc = cv2.findContours(image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 
        area = np.zeros(len(contours))
        for j in range(len(contours)):
            cnt = contours[j]
            area[j] = cv2.contourArea(cnt)
        mask = np.zeros(image.shape)
        cv2.drawContours(mask, contours, np.argmax(area), (255), -1)  # draw largest contour-usually right lung   
        temp = np.copy(area[np.argmax(area)])
        area[np.argmax(area)]=0
        if area[np.argmax(area)] > temp/10:  # make sure 2nd largest contour is also lung, not 2 lungs connected
            cv2.drawContours(mask, contours, np.argmax(area), (255), -1)#draw second largest contour  
        contours.clear() 
        return mask


class Augmentator:
    def __init__(self, in_channel=3):
        self._resizer = RandImgResizer(prob=0.5, delta=0.1)
        self._tf_augmentator = ImageDataGenerator(
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
        self.in_channel = in_channel

    def tf_augmentate(self, img):
        """
        tensorflow augumentation
        @param
        :img: numpy.ndarray
        :return: numpy.ndarray
        """
        img = self._resizer(Image.fromarray(img))
        img = self._tf_augmentator.random_transform(np.array(img))
        return img

    @property
    def pytorch_aumentator(self):
        # pytorch augumentation, no need to use transforms.Normalize for [TResNet]: https://github.com/mrT23/TResNet/issues/5#issuecomment-608440989
        # custom transform: https://discuss.pytorch.org/t/how-to-use-custom-image-transformations-with-torchvision/71469
        _transforms = [
                    self._resizer,  # custom transform only works on first few samples then raise TypeError (np.ndarray instead of PIL), cannot find out why, CONFUSED
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(10),
                    transforms.ColorJitter(hue=.1, saturation=.1),
                    transforms.RandomGrayscale(p=0.1),
                    transforms.ToTensor()]
        if self.in_channel == 1:
            _transforms.insert(0, transforms.Grayscale())
        return transforms.Compose(_transforms)


class RandImgResizer(object):
    """
    Rnadom Ratio Resize image
    ---------------
    @param
    :prob: resize probability
    :delta: resize delta
    ---------------
    """

    def __init__(self, prob=0.5, delta=0.1):
        self.prob = prob
        self.delta = delta

    def __call__(self, img):
        """
        ---------------
        @param
        :img: PIL 
        :return: PIL
        ---------------
        """
        if np.random.rand() >= self.prob:
            return img

        img = np.array(img)

        ratio = img.shape[0] / img.shape[1]
        ratio = np.random.uniform(max(ratio - self.delta, 0.01), ratio + self.delta)

        if ratio * img.shape[1] <= img.shape[1]:
            size = (int(img.shape[1] * ratio), img.shape[1])
        else:
            size = (img.shape[0], int(img.shape[0] / ratio))

        dh = img.shape[0] - size[1]
        top, bot = dh // 2, dh - dh // 2
        dw = img.shape[1] - size[0]
        left, right = dw // 2, dw - dw // 2

        img = cv2.resize(img, size)
        img = cv2.copyMakeBorder(img, top, bot, left, right, cv2.BORDER_CONSTANT, (0, 0, 0))

        return Image.fromarray(img)

    def __repr__(self):
        return self.__class__.__name__+'()'