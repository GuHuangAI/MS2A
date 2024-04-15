#!/usr/bin/env python
# -*- coding: utf-8 -*-


import copy
import inspect
import math
import warnings

import cv2
import mmcv
import numpy as np
from numpy import random
import numpy
import torch
import math
import random
from PIL import Image
from torchvision.transforms import functional as F
from ..builder import PIPELINES


@PIPELINES.register_module()
class Target_aug:

    def __init__(self, img_scale=(640, 640)):
        self.img_scale = img_scale

    def __call__(self, results):
        #print(type(results))
        #for key in results:
        #    print(key, type(results[key]))
        print('file_name', results['img_info'])
        print('filename:', results['filename'], results['ori_filename'])
        print('img_len:', len(results['img']))
        print('gt_bboxes', len(results['gt_bboxes']))
        print('gt_labels', results['gt_labels'])
        return results
    
    
    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'img_scale={self.img_scale}, '
        return repr_str
