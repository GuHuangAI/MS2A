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


from mmdet.core import BitmapMasks, PolygonMasks, find_inside_bboxes
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
from mmdet.utils import log_img_scale
from ..builder import PIPELINES

try:
    from imagecorruptions import corrupt
except ImportError:
    corrupt = None

try:
    import albumentations
    from albumentations import Compose
except ImportError:
    albumentations = None
    Compose = None

import torchvision.transforms as T
from mmdet.datasets.pipelines.auto_augment import BrightnessTransform,\
    ColorTransform,Shear,Rotate
from mmdet.datasets.pipelines.auto_augment import GaussianBlur,RandomImgAugFromAPI




@PIPELINES.register_module()
class SuperMosaic:

    def __init__(self,
                 img_scale=(640, 640),
                 center_ratio_range=(0.5, 1.5),
                 min_bbox_size=0,
                 bbox_clip_border=True,
                 skip_filter=True,
                 pad_val=114,
                 prob=1.0,
                 h_scale=2,
                 w_scale=2,
                 mosaic_type='4part'  # 'default' 4 , '4part', 'randomstack'
                 ):
        assert isinstance(img_scale, tuple)
        assert 0 <= prob <= 1.0, 'The probability should be in range [0,1]. ' \
                                 f'got {prob}.'

        log_img_scale(img_scale, skip_square=True)
        self.img_scale = img_scale # (h,w)格式
        self.center_ratio_range = center_ratio_range
        self.min_bbox_size = min_bbox_size
        self.bbox_clip_border = bbox_clip_border
        self.skip_filter = skip_filter
        self.pad_val = pad_val
        self.prob = prob
        self.type = type
        self.w_scale = w_scale
        self.h_scale = h_scale
        self.mosaic_type = mosaic_type
        self.randomcrop = RandomImgAugFromAPI(type_seq='seq_croppaste')
        from mmdet.datasets.pipelines.transforms import ResizeInput
        self.keepresize = ResizeInput(img_scale=img_scale, keep_ratio=True)

    def __call__(self, results):

        if random.uniform(0, 1) > self.prob:
            return results

        results = self._mosaic_transform(results)
        return results

    def get_indexes(self, dataset):

        indexes = [random.randint(0, len(dataset) - 1) for _ in range(3)]
        return indexes

    def _4part_mosaic(self, results):
        assert 'mix_results' in results
        mosaic_labels = []
        mosaic_bboxes = []
        if len(results['img'].shape) == 3:  # 扩大四倍输出图片
            mosaic_img = np.full(
                (int(self.img_scale[0] * self.h_scale), int(self.img_scale[1] * self.w_scale), 3),
                self.pad_val,
                dtype=results['img'].dtype)
        else:
            mosaic_img = np.full(
                (int(self.img_scale[0] * self.h_scale), int(self.img_scale[1] * self.w_scale)),
                self.pad_val,
                dtype=results['img'].dtype)

        # mosaic center x, y  中心点是随机生成的。
        center_x = int(
            random.uniform(*self.center_ratio_range) * self.img_scale[1])
        center_y = int(
            random.uniform(*self.center_ratio_range) * self.img_scale[0])
        center_position = (center_x, center_y)

        loc_strs = ('top_left', 'top_right', 'bottom_left', 'bottom_right')
        for i, loc in enumerate(loc_strs):  # 贴图四次
            if loc == 'top_left':
                results_patch = copy.deepcopy(results)  # 左上用自己
            else:
                results_patch = copy.deepcopy(results['mix_results'][i - 1])

            img_i = results_patch['img']
            h_i, w_i = img_i.shape[:2]
            # keep_ratio resize
            # 这里是等比的，实际可以四个图随机缩放，提升鲁棒性
            scale_ratio_i = min(self.img_scale[0] / h_i,
                                self.img_scale[1] / w_i)
            img_i = mmcv.imresize(
                img_i, (int(w_i * scale_ratio_i), int(h_i * scale_ratio_i)))

            # compute the combine parameters
            paste_coord, crop_coord = self._mosaic_combine(
                loc, center_position, img_i.shape[:2][::-1])
            x1_p, y1_p, x2_p, y2_p = paste_coord
            x1_c, y1_c, x2_c, y2_c = crop_coord

            # crop and paste image
            mosaic_img[y1_p:y2_p, x1_p:x2_p] = img_i[y1_c:y2_c, x1_c:x2_c]

            # adjust coordinate
            gt_bboxes_i = results_patch['gt_bboxes']
            gt_labels_i = results_patch['gt_labels']

            if gt_bboxes_i.shape[0] > 0:
                padw = x1_p - x1_c
                padh = y1_p - y1_c
                gt_bboxes_i[:, 0::2] = \
                    scale_ratio_i * gt_bboxes_i[:, 0::2] + padw
                gt_bboxes_i[:, 1::2] = \
                    scale_ratio_i * gt_bboxes_i[:, 1::2] + padh

            mosaic_bboxes.append(gt_bboxes_i)
            mosaic_labels.append(gt_labels_i)
        return mosaic_labels, mosaic_bboxes, mosaic_img


    def _randomstack_mosaic(self, results, ):
        assert 'mix_results' in results

        # 4 img  stack
        total_img = []
        total_box = []
        total_label = []
        max_w = 0
        for i in range(0,4):
            if i == 0:
                results_aug = copy.deepcopy(results)  # 左上用自己 ， 必须resize了
            else:
                results_patch = copy.deepcopy(results['mix_results'][i - 1])
                results_patch = self.keepresize(results_patch)
                results_aug = self.randomcrop(results_patch) # 随机处理
            results_img = results_aug['img']
            results_boxes = results_aug['gt_bboxes']
            results_labels = results_aug['gt_labels']
            total_img.append(results_img)
            total_box.append(results_boxes)
            total_label.append(results_labels)
            h,w,c = results_img.shape
            if w > max_w:
                max_w = w
        results_img_pad = []
        for results_img in total_img:
            h, w, c = results_img.shape
            if w < max_w:
                results_img = np.pad(results_img,((0,0),(0,max_w-w),(0,0)),'constant')
            results_img_pad.append(results_img)
        # img
        mosaic_img = np.concatenate(results_img_pad,axis=0)
        # box refine
        mosaic_labels = []
        mosaic_bboxes = []
        start_x = 0
        start_y = 0
        for im,box,label in zip(total_img,total_box,total_label):
            h,w,c = im.shape
            # adjust coordinate
            gt_bboxes_i = box
            gt_labels_i = label
            gt_bboxes_i[:, 0::2] = gt_bboxes_i[:, 0::2] + start_x  # box 先缩放，然后再pad距离w
            gt_bboxes_i[:, 1::2] = gt_bboxes_i[:, 1::2] + start_y
            mosaic_bboxes.append(gt_bboxes_i)
            mosaic_labels.append(gt_labels_i)
            start_y += h
        return mosaic_labels, mosaic_bboxes, mosaic_img

    def _mosaic_transform(self, results):

        if self.mosaic_type == '4part':
            mosaic_labels, mosaic_bboxes, mosaic_img = self._4part_mosaic(results)
        elif self.mosaic_type == 'randomstack':
            mosaic_labels, mosaic_bboxes, mosaic_img = self._randomstack_mosaic(results)
        else:
            print('error  type', self.mix_type)
            raise ('error  type {} '.format( self.mix_type))

        if len(mosaic_labels) > 0:
            mosaic_bboxes = np.concatenate(mosaic_bboxes, 0)
            mosaic_labels = np.concatenate(mosaic_labels, 0)

            if self.bbox_clip_border:
                mosaic_bboxes[:, 0::2] = np.clip(mosaic_bboxes[:, 0::2], 0,
                                                 self.w_scale * self.img_scale[1])
                mosaic_bboxes[:, 1::2] = np.clip(mosaic_bboxes[:, 1::2], 0,
                                                 self.h_scale * self.img_scale[0])

            if not self.skip_filter:
                mosaic_bboxes, mosaic_labels = \
                    self._filter_box_candidates(mosaic_bboxes, mosaic_labels)

        # remove outside bboxes
        inside_inds = find_inside_bboxes(mosaic_bboxes, self.h_scale * self.img_scale[0],
                                         self.w_scale * self.img_scale[1])
        mosaic_bboxes = mosaic_bboxes[inside_inds]
        mosaic_labels = mosaic_labels[inside_inds]

        results['img'] = mosaic_img
        
        #print(mosaic_img.shape)
        #img_array = mosaic_img[0].numpy()
        #img = Image.fromarray(img_array)
        #img.save("/nfs/mosic.jpg")
        
        results['img_shape'] = mosaic_img.shape
        results['gt_bboxes'] = mosaic_bboxes
        results['gt_labels'] = mosaic_labels

        return results

    def _mosaic_combine(self, loc, center_position_xy, img_shape_wh):

        assert loc in ('top_left', 'top_right', 'bottom_left', 'bottom_right')
        if loc == 'top_left':
            # index0 to top left part of image
            x1, y1, x2, y2 = max(center_position_xy[0] - img_shape_wh[0], 0), \
                             max(center_position_xy[1] - img_shape_wh[1], 0), \
                             center_position_xy[0], \
                             center_position_xy[1]
            crop_coord = img_shape_wh[0] - (x2 - x1), img_shape_wh[1] - (
                    y2 - y1), img_shape_wh[0], img_shape_wh[1]

        elif loc == 'top_right':
            # index1 to top right part of image
            x1, y1, x2, y2 = center_position_xy[0], \
                             max(center_position_xy[1] - img_shape_wh[1], 0), \
                             min(center_position_xy[0] + img_shape_wh[0],
                                 self.img_scale[1] * 2), \
                             center_position_xy[1]
            crop_coord = 0, img_shape_wh[1] - (y2 - y1), min(
                img_shape_wh[0], x2 - x1), img_shape_wh[1]

        elif loc == 'bottom_left':
            # index2 to bottom left part of image
            x1, y1, x2, y2 = max(center_position_xy[0] - img_shape_wh[0], 0), \
                             center_position_xy[1], \
                             center_position_xy[0], \
                             min(self.img_scale[0] * 2, center_position_xy[1] +
                                 img_shape_wh[1])
            crop_coord = img_shape_wh[0] - (x2 - x1), 0, img_shape_wh[0], min(
                y2 - y1, img_shape_wh[1])

        else:
            # index3 to bottom right part of image
            x1, y1, x2, y2 = center_position_xy[0], \
                             center_position_xy[1], \
                             min(center_position_xy[0] + img_shape_wh[0],
                                 self.img_scale[1] * 2), \
                             min(self.img_scale[0] * 2, center_position_xy[1] +
                                 img_shape_wh[1])
            crop_coord = 0, 0, min(img_shape_wh[0],
                                   x2 - x1), min(y2 - y1, img_shape_wh[1])

        paste_coord = x1, y1, x2, y2
        return paste_coord, crop_coord

    def _filter_box_candidates(self, bboxes, labels):
        
        bbox_w = bboxes[:, 2] - bboxes[:, 0]
        bbox_h = bboxes[:, 3] - bboxes[:, 1]
        valid_inds = (bbox_w > self.min_bbox_size) & \
                     (bbox_h > self.min_bbox_size)
        valid_inds = np.nonzero(valid_inds)[0]
        return bboxes[valid_inds], labels[valid_inds]

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'img_scale={self.img_scale}, '
        repr_str += f'center_ratio_range={self.center_ratio_range}, '
        repr_str += f'pad_val={self.pad_val}, '
        repr_str += f'min_bbox_size={self.min_bbox_size}, '
        repr_str += f'skip_filter={self.skip_filter})'
        return repr_str
