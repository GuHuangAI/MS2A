#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt

import os
import asyncio
from argparse import ArgumentParser
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from io import BytesIO
from mmdet.datasets.pipelines import Compose
from mmcv.parallel import collate, scatter
import torch
import mmcv
import argparse
from tqdm import tqdm
import cv2
import random

from mmdet.apis import (async_inference_detector, init_detector, show_result_pyplot)


def inference_detector(model, imgs):
    """Inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray] or tuple[str/ndarray]):
           Either image files or loaded images.

    Returns:
        If imgs is a list or tuple, the same length list type results
        will be returned, otherwise return the detection results directly.
    """

    if isinstance(imgs, (list, tuple)):
        is_batch = True
    else:
        imgs = [imgs]
        is_batch = False

    cfg = model.cfg
    device = next(model.parameters()).device  # model device

    if isinstance(imgs[0], np.ndarray):
        cfg = cfg.copy()
        # set loading pipeline type
        cfg.test_pipeline[0].type = 'LoadImageFromWebcam'

    #cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    #test_pipeline = Compose(cfg.data.test.pipeline)
    test_pipeline = Compose(cfg.test_pipeline)

    datas = []
    for img in imgs:
        # prepare data
        if isinstance(img, np.ndarray):
            # directly add img
            data = dict(img=img, env_id=1)
        else:
            # add information into dict
            data = dict(img_info=dict(filename=img), img_prefix=None, env_id=1) #
        # build the data pipeline
        data = test_pipeline(data)
        datas.append(data)

    data = collate(datas, samples_per_gpu=len(imgs))
    # just get the actual data from DataContainer
    data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]
    data['img'] = [img.data[0] for img in data['img']]
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        for m in model.modules():
            assert not isinstance(
                m, RoIPool
            ), 'CPU inference with RoIPool is not supported currently.'

    # forward the model
    with torch.no_grad():
        results = model(return_loss=False, rescale=True, **data)

    if not is_batch:
        return results[0]
    else:
        return results

def save_result(img, result, out_file=None, thickness=3, score_thr=0.3):
    seed_color = 77
    if 'public_data/foggy_cityscapes_COCO_format' in img:
        class_name = ['person', 'car', 'rider', 'bicycle', 'truck', 'motorcycle', 'train', 'bus']
    elif 'public_data/cityscapes_COCO_format' in img:
        class_name = ['car']
    elif 'testset/' in img:
        class_name = ['part']

    # colors = []
    # for _ in class_name:
    #     random.seed(seed_color)
    #     color = [random.randint(0, 255) for _ in range(3)]
    #     colors.append(color)
    #     seed_color = 4 * seed_color + 6
    colors = [[0, 0, 255], [255, 0, 0], [0, 255, 0], [0, 139, 0], [139, 139, 0], [0, 0, 139], [255, 48, 155], [255, 0, 255]]

    img = cv2.imread(img, cv2.IMREAD_COLOR)
    img = img.copy()
    if isinstance(result, tuple):
        bbox_result, segm_result = result
        if isinstance(segm_result, tuple):
            segm_result = segm_result[0]  # ms rcnn
    else:
        bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    scores = bboxes[:, -1]

    inds = scores > score_thr
    bboxes = bboxes[inds, :]
    labels = labels[inds]

    # img = mmcv.bgr2rgb(img)
    img = np.ascontiguousarray(img)
    line_thickness = thickness

    for bbox, score, label_ in zip(bboxes[:, :4], bboxes[:, -1], labels):
        tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
        label = '%s %.2f' % (class_name[int(label_)], score)
        color = colors[int(label_)] or [random.randint(0, 255) for _ in range(3)]
        c1, c2 = (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        if label :
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_TRIPLEX, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(img, text=label, org=(c1[0], c1[1] - 3), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=tl / 3, color=[225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        cv2.imwrite(out_file, img)

def parse_args():
    parser = argparse.ArgumentParser(description='Instruction')
    parser.add_argument('--score', type=float, default=0.7, help='score')
    parser.add_argument('--thickness', type=int, default=2, help='score')
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_args()
    score_list = [
        0.7,
        0.7,
        0.7,
        0.7,
        0.7
    ]
    
    config_dic = {
        # "foggycityscapes": "/home/ml/code/mmdetection/configs/yolox/yolox_fs_x_640x640_50e_cityscapes_8_pretrain_mom_ft.py",
        'kitti': "/home/ml/code/mmdetection/configs/yolox/yolox_fs_x_640x640_50e_kitti_pretrain_mom_ft.py",
        # 'sim10k': "/home/ml/code/mmdetection/configs/yolox/yolox_fs_x_640x640_50e_sim10k_pretrain_mom_ft.py",
        # 'xingbang': "/home/ml/code/mmdetection/configs/yolox/yolox_x_640x640_50e_18_xingbang50_ft_2.py",
        # 'zhonglian': "/home/ml/code/mmdetection/configs/yolox/yolox_x_640x640_50e_18_zhonglian50_ft_3.py"
    }

    checkpoint_dic = {
        # "foggycityscapes": "/home/ml/code/mmdetection/work_dirs/yolox_fs_x_640x640_50e_cityscapes_8_pretrain_mom_ft/best_0_bbox_mAP_epoch_20.pth",
        'kitti': "/home/ml/code/mmdetection/work_dirs/yolox_fs_x_640x640_50e_kitti_pretrain_mom_ft/best_0_bbox_mAP_50_epoch_18.pth",
        # 'sim10k': "/home/ml/code/mmdetection/work_dirs/yolox_fs_x_640x640_50e_sim10k_pretrain_mom_ft/sim10_ft_65.5_best_0_bbox_mAP_50_epoch_15.pth",
        # 'xingbang': "/home/ml/code/mmdetection/work_dirs/yolox_x_640x640_50e_18_xingbang50_ft_2/best_0_bbox_mAP_epoch_18.pth",
        # 'zhonglian': "/home/ml/code/mmdetection/work_dirs/yolox_x_640x640_50e_18_zhonglian50_ft_3/best_0_bbox_mAP_epoch_17.pth"
    }

    img_dic = {
        # "foggycityscapes": [
        #     "/benchmark/持续学习数据集/public_data/foggy_cityscapes_COCO_format/val2014/frankfurt/frankfurt_000000_000576_leftImg8bit_foggy_beta_0.02.png", 
        #     "/benchmark/持续学习数据集/public_data/foggy_cityscapes_COCO_format/val2014/frankfurt/frankfurt_000000_005543_leftImg8bit_foggy_beta_0.02.png",
        #     "/benchmark/持续学习数据集/public_data/foggy_cityscapes_COCO_format/val2014/frankfurt/frankfurt_000000_013382_leftImg8bit_foggy_beta_0.02.png",
        #     "/benchmark/持续学习数据集/public_data/foggy_cityscapes_COCO_format/val2014/frankfurt/frankfurt_000000_020215_leftImg8bit_foggy_beta_0.02.png",
        #     "/benchmark/持续学习数据集/public_data/foggy_cityscapes_COCO_format/val2014/frankfurt/frankfurt_000001_010830_leftImg8bit_foggy_beta_0.02.png",
        #     "/benchmark/持续学习数据集/public_data/foggy_cityscapes_COCO_format/val2014/frankfurt/frankfurt_000001_015768_leftImg8bit_foggy_beta_0.02.png",
        #     "/benchmark/持续学习数据集/public_data/foggy_cityscapes_COCO_format/val2014/frankfurt/frankfurt_000001_054219_leftImg8bit_foggy_beta_0.02.png",
        #     "/benchmark/持续学习数据集/public_data/foggy_cityscapes_COCO_format/val2014/frankfurt/frankfurt_000001_055172_leftImg8bit_foggy_beta_0.02.png",
        #     "/benchmark/持续学习数据集/public_data/foggy_cityscapes_COCO_format/val2014/frankfurt/frankfurt_000001_058504_leftImg8bit_foggy_beta_0.02.png",
        #     "/benchmark/持续学习数据集/public_data/foggy_cityscapes_COCO_format/val2014/munster/munster_000007_000019_leftImg8bit_foggy_beta_0.02.png",
        #     "/benchmark/持续学习数据集/public_data/foggy_cityscapes_COCO_format/val2014/munster/munster_000009_000019_leftImg8bit_foggy_beta_0.02.png",
        #     "/benchmark/持续学习数据集/public_data/foggy_cityscapes_COCO_format/val2014/munster/munster_000011_000019_leftImg8bit_foggy_beta_0.02.png",
        #     "/benchmark/持续学习数据集/public_data/foggy_cityscapes_COCO_format/val2014/munster/munster_000008_000019_leftImg8bit_foggy_beta_0.02.png",
        #     "/benchmark/持续学习数据集/public_data/foggy_cityscapes_COCO_format/val2014/munster/munster_000016_000019_leftImg8bit_foggy_beta_0.02.png",
        #     "/benchmark/持续学习数据集/public_data/foggy_cityscapes_COCO_format/val2014/lindau/lindau_000005_000019_leftImg8bit_foggy_beta_0.02.png",
        #     "/benchmark/持续学习数据集/public_data/foggy_cityscapes_COCO_format/val2014/lindau/lindau_000026_000019_leftImg8bit_foggy_beta_0.02.png",
        #     "/benchmark/持续学习数据集/public_data/foggy_cityscapes_COCO_format/val2014/lindau/lindau_000028_000019_leftImg8bit_foggy_beta_0.02.png",
        #     "/benchmark/持续学习数据集/public_data/foggy_cityscapes_COCO_format/val2014/lindau/lindau_000038_000019_leftImg8bit_foggy_beta_0.02.png",
        #     "/benchmark/持续学习数据集/public_data/foggy_cityscapes_COCO_format/val2014/lindau/lindau_000041_000019_leftImg8bit_foggy_beta_0.02.png",
        #     "/benchmark/持续学习数据集/public_data/foggy_cityscapes_COCO_format/val2014/lindau/lindau_000057_000019_leftImg8bit_foggy_beta_0.02.png",
        # ],

        "kitti": [
            "/benchmark/持续学习数据集/public_data/cityscapes_COCO_format/val2014/lindau/lindau_000001_000019_leftImg8bit.png",
            "/benchmark/持续学习数据集/public_data/cityscapes_COCO_format/val2014/lindau/lindau_000026_000019_leftImg8bit.png",
            "/benchmark/持续学习数据集/public_data/cityscapes_COCO_format/val2014/lindau/lindau_000028_000019_leftImg8bit.png",
            "/benchmark/持续学习数据集/public_data/cityscapes_COCO_format/val2014/lindau/lindau_000041_000019_leftImg8bit.png",
            "/benchmark/持续学习数据集/public_data/cityscapes_COCO_format/val2014/lindau/lindau_000057_000019_leftImg8bit.png",
            "/benchmark/持续学习数据集/public_data/cityscapes_COCO_format/val2014/munster/munster_000008_000019_leftImg8bit.png",
            "/benchmark/持续学习数据集/public_data/cityscapes_COCO_format/val2014/munster/munster_000012_000019_leftImg8bit.png",
            "/benchmark/持续学习数据集/public_data/cityscapes_COCO_format/val2014/munster/munster_000016_000019_leftImg8bit.png",
            "/benchmark/持续学习数据集/public_data/cityscapes_COCO_format/val2014/munster/munster_000017_000019_leftImg8bit.png",
            "/benchmark/持续学习数据集/public_data/cityscapes_COCO_format/val2014/munster/munster_000036_000019_leftImg8bit.png",
            "/benchmark/持续学习数据集/public_data/cityscapes_COCO_format/val2014/munster/munster_000007_000019_leftImg8bit.png",
            "/benchmark/持续学习数据集/public_data/cityscapes_COCO_format/val2014/munster/munster_000009_000019_leftImg8bit.png",
            "/benchmark/持续学习数据集/public_data/cityscapes_COCO_format/val2014/munster/munster_000027_000019_leftImg8bit.png",
            "/benchmark/持续学习数据集/public_data/cityscapes_COCO_format/val2014/frankfurt/frankfurt_000000_000576_leftImg8bit.png",
            "/benchmark/持续学习数据集/public_data/cityscapes_COCO_format/val2014/frankfurt/frankfurt_000000_003025_leftImg8bit.png",
            "/benchmark/持续学习数据集/public_data/cityscapes_COCO_format/val2014/frankfurt/frankfurt_000000_011810_leftImg8bit.png",
            "/benchmark/持续学习数据集/public_data/cityscapes_COCO_format/val2014/frankfurt/frankfurt_000000_020215_leftImg8bit.png",
            "/benchmark/持续学习数据集/public_data/cityscapes_COCO_format/val2014/frankfurt/frankfurt_000001_007285_leftImg8bit.png",
            "/benchmark/持续学习数据集/public_data/cityscapes_COCO_format/val2014/frankfurt/frankfurt_000001_015768_leftImg8bit.png",
            "/benchmark/持续学习数据集/public_data/cityscapes_COCO_format/val2014/frankfurt/frankfurt_000001_015091_leftImg8bit.png"
        ],

        # "sim10k": [
        #     "/benchmark/持续学习数据集/public_data/cityscapes_COCO_format/val2014/lindau/lindau_000001_000019_leftImg8bit.png",
        #     "/benchmark/持续学习数据集/public_data/cityscapes_COCO_format/val2014/lindau/lindau_000026_000019_leftImg8bit.png",
        #     "/benchmark/持续学习数据集/public_data/cityscapes_COCO_format/val2014/lindau/lindau_000028_000019_leftImg8bit.png",
        #     "/benchmark/持续学习数据集/public_data/cityscapes_COCO_format/val2014/lindau/lindau_000041_000019_leftImg8bit.png",
        #     "/benchmark/持续学习数据集/public_data/cityscapes_COCO_format/val2014/lindau/lindau_000057_000019_leftImg8bit.png",
        #     "/benchmark/持续学习数据集/public_data/cityscapes_COCO_format/val2014/munster/munster_000008_000019_leftImg8bit.png",
        #     "/benchmark/持续学习数据集/public_data/cityscapes_COCO_format/val2014/munster/munster_000012_000019_leftImg8bit.png",
        #     "/benchmark/持续学习数据集/public_data/cityscapes_COCO_format/val2014/munster/munster_000016_000019_leftImg8bit.png",
        #     "/benchmark/持续学习数据集/public_data/cityscapes_COCO_format/val2014/munster/munster_000017_000019_leftImg8bit.png",
        #     "/benchmark/持续学习数据集/public_data/cityscapes_COCO_format/val2014/munster/munster_000036_000019_leftImg8bit.png",
        #     "/benchmark/持续学习数据集/public_data/cityscapes_COCO_format/val2014/munster/munster_000007_000019_leftImg8bit.png",
        #     "/benchmark/持续学习数据集/public_data/cityscapes_COCO_format/val2014/munster/munster_000009_000019_leftImg8bit.png",
        #     "/benchmark/持续学习数据集/public_data/cityscapes_COCO_format/val2014/munster/munster_000027_000019_leftImg8bit.png",
        #     "/benchmark/持续学习数据集/public_data/cityscapes_COCO_format/val2014/frankfurt/frankfurt_000000_000576_leftImg8bit.png",
        #     "/benchmark/持续学习数据集/public_data/cityscapes_COCO_format/val2014/frankfurt/frankfurt_000000_003025_leftImg8bit.png",
        #     "/benchmark/持续学习数据集/public_data/cityscapes_COCO_format/val2014/frankfurt/frankfurt_000000_011810_leftImg8bit.png",
        #     "/benchmark/持续学习数据集/public_data/cityscapes_COCO_format/val2014/frankfurt/frankfurt_000000_020215_leftImg8bit.png",
        #     "/benchmark/持续学习数据集/public_data/cityscapes_COCO_format/val2014/frankfurt/frankfurt_000001_007285_leftImg8bit.png",
        #     "/benchmark/持续学习数据集/public_data/cityscapes_COCO_format/val2014/frankfurt/frankfurt_000001_015768_leftImg8bit.png",
        #     "/benchmark/持续学习数据集/public_data/cityscapes_COCO_format/val2014/frankfurt/frankfurt_000001_015091_leftImg8bit.png",
        #     "/benchmark/持续学习数据集/public_data/cityscapes_COCO_format/val2014/frankfurt/frankfurt_000001_080391_leftImg8bit.png",
        #     "/benchmark/持续学习数据集/public_data/cityscapes_COCO_format/val2014/frankfurt/frankfurt_000001_080091_leftImg8bit.png",
        #     "/benchmark/持续学习数据集/public_data/cityscapes_COCO_format/val2014/frankfurt/frankfurt_000001_079206_leftImg8bit.png",
        #     "/benchmark/持续学习数据集/public_data/cityscapes_COCO_format/val2014/frankfurt/frankfurt_000001_077233_leftImg8bit.png",
        #     "/benchmark/持续学习数据集/public_data/cityscapes_COCO_format/val2014/frankfurt/frankfurt_000001_077092_leftImg8bit.png",
        #     "/benchmark/持续学习数据集/public_data/cityscapes_COCO_format/val2014/frankfurt/frankfurt_000001_075296_leftImg8bit.png",
        #     "/benchmark/持续学习数据集/public_data/cityscapes_COCO_format/val2014/frankfurt/frankfurt_000001_073911_leftImg8bit.png",
        #     "/benchmark/持续学习数据集/public_data/cityscapes_COCO_format/val2014/frankfurt/frankfurt_000001_073464_leftImg8bit.png",
        # ],

        # "xingbang": [
        #     "/benchmark/持续学习数据集/testset/xingbang_bigsecond_merge_COCO_format/val2017/000000000016.jpg",
        #     "/benchmark/持续学习数据集/testset/xingbang_bigsecond_merge_COCO_format/val2017/000000000027.jpg",
        #     "/benchmark/持续学习数据集/testset/xingbang_bigsecond_merge_COCO_format/val2017/000000000020.jpg",
        #     "/benchmark/持续学习数据集/testset/xingbang_bigsecond_merge_COCO_format/val2017/000000000033.jpg",
        #     "/benchmark/持续学习数据集/testset/xingbang_bigsecond_merge_COCO_format/val2017/000000000049.jpg",
        #     "/benchmark/持续学习数据集/testset/xingbang_bigsecond_merge_COCO_format/val2017/000000000009.jpg",
        #     "/benchmark/持续学习数据集/testset/xingbang_bigsecond_merge_COCO_format/val2017/000000000014.jpg",
        #     "/benchmark/持续学习数据集/testset/xingbang_bigsecond_merge_COCO_format/val2017/000000000017.jpg",
        #     "/benchmark/持续学习数据集/testset/xingbang_bigsecond_merge_COCO_format/val2017/000000000019.jpg",
        #     "/benchmark/持续学习数据集/testset/xingbang_bigsecond_merge_COCO_format/val2017/000000000030.jpg",
        #     "/benchmark/持续学习数据集/testset/xingbang_bigsecond_merge_COCO_format/val2017/000000000044.jpg",
        #     "/benchmark/持续学习数据集/testset/xingbang_bigsecond_merge_COCO_format/val2017/000000000051.jpg",
        #     "/benchmark/持续学习数据集/testset/xingbang_bigsecond_merge_COCO_format/val2017/000000000084.jpg",
        #     "/benchmark/持续学习数据集/testset/xingbang_bigsecond_merge_COCO_format/val2017/000000000085.jpg",
        #     "/benchmark/持续学习数据集/testset/xingbang_bigsecond_merge_COCO_format/val2017/000000000096.jpg",
        #     "/benchmark/持续学习数据集/testset/xingbang_bigsecond_merge_COCO_format/val2017/000000000103.jpg",
        #     "/benchmark/持续学习数据集/testset/xingbang_bigsecond_merge_COCO_format/val2017/000000000108.jpg",
        #     "/benchmark/持续学习数据集/testset/xingbang_bigsecond_merge_COCO_format/val2017/000000000123.jpg",
        #     "/benchmark/持续学习数据集/testset/xingbang_bigsecond_merge_COCO_format/val2017/000000000126.jpg",
        #     "/benchmark/持续学习数据集/testset/xingbang_bigsecond_merge_COCO_format/val2017/000000000135.jpg"
        # ],

        # "zhonglian": [
        #     "/benchmark/持续学习数据集/testset/zhonglian_bigsecond_merge_COCO_format/val2017/000000000025.jpg",
        #     "/benchmark/持续学习数据集/testset/zhonglian_bigsecond_merge_COCO_format/val2017/000000000029.jpg",
        #     "/benchmark/持续学习数据集/testset/zhonglian_bigsecond_merge_COCO_format/val2017/000000000040.jpg",
        #     "/benchmark/持续学习数据集/testset/zhonglian_bigsecond_merge_COCO_format/val2017/000000000065.jpg",
        #     "/benchmark/持续学习数据集/testset/zhonglian_bigsecond_merge_COCO_format/val2017/000000000078.jpg",
        #     "/benchmark/持续学习数据集/testset/zhonglian_bigsecond_merge_COCO_format/val2017/000000000084.jpg",
        #     "/benchmark/持续学习数据集/testset/zhonglian_bigsecond_merge_COCO_format/val2017/000000000414.jpg",
        #     "/benchmark/持续学习数据集/testset/zhonglian_bigsecond_merge_COCO_format/val2017/000000000393.jpg",
        #     "/benchmark/持续学习数据集/testset/zhonglian_bigsecond_merge_COCO_format/val2017/000000000334.jpg",
        #     "/benchmark/持续学习数据集/testset/zhonglian_bigsecond_merge_COCO_format/val2017/000000000325.jpg",
        #     "/benchmark/持续学习数据集/testset/zhonglian_bigsecond_merge_COCO_format/val2017/000000000280.jpg",
        # ],
    }

    model_list = []
    for type in img_dic.keys():
        config = config_dic[type]
        checkpoint = checkpoint_dic[type]
        model = init_detector(config, checkpoint, device='cuda:0')
        model_list.append(model)

    pbar = tqdm(img_dic.keys())
    for id, type in enumerate(pbar):
        for img in img_dic[type]:
            model = model_list[id]
            img_name = os.path.basename(img)
            result = inference_detector(model, img)
            os.makedirs(os.path.join("../plot_result_0730", type), exist_ok=True)
            if type == 'xingbang' or type == 'zhonglian':
                line_thickness = 12
            else:
                line_thickness = args.thickness
            save_result(img, result, out_file=os.path.join('../plot_result_0730', type, os.path.basename(img)), thickness=line_thickness, score_thr=score_list[id])
            # model.show_result(img, result, out_file=os.path.join('../plot_result_Our', type, os.path.basename(img)), thickness=3, bbox_color='red', text_color='red', score_thr=score_list[id]) # , front_scale=0.5

            pbar.set_description(f"{type}:")
            pbar.set_postfix({'img_name': '{}'.format(img_name)})