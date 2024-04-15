#!/usr/bin/env python
# -*- coding:utf-8-*-
# @desc:  模型测试, 查看误检和漏检

import os
import os.path as osp
import copy
import json

import cv2
import numpy as np
import torch
from mmcv import Config
from mmcv.cnn import fuse_conv_bn
from mmcv.runner import (load_checkpoint)

from mmdet.core.bbox.iou_calculators import bbox_overlaps
from mmdet.datasets import build_dataset, build_dataloader
from mmdet.models import build_detector

BATCH_SIZE = 1
THRESHOLD = 0.6
CONFIG = "configs/xx/cfg_2_ssd_128x128.py"
WORK_DIR = "train_result/20221011_cfg_2_ssd_128x128-11"
EPOCH = 1340
CKPT = f"{WORK_DIR}/epoch_{EPOCH}.pth"
SAVE_DIR = f"{WORK_DIR}/test_2"
SAVE_IMG_FLAG = 1
TP_ASSIGN_IOU = 0.65

def build_model():
    cfg = Config.fromfile(CONFIG)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    _ckpt = CKPT
    if not osp.isfile(_ckpt) or not osp.exists(_ckpt):
        raise FileNotFoundError(f'{_ckpt} is not existed.')

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=BATCH_SIZE,
        workers_per_gpu=0,
        dist=False,
        shuffle=False)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    _test_cfg = cfg.get('test_cfg')
    if _test_cfg is None:
        _test_cfg = cfg.model.get('test_cfg')
    assert _test_cfg is not None
    cfg.model['test_cfg']['score_thr'] = 0.2
    cfg.model['test_cfg']['nms']['iou_threshold'] = 0.5
    model = build_detector(cfg.model)
    checkpoint = load_checkpoint(model, _ckpt, map_location='cpu')
    model = fuse_conv_bn(model)
    model.eval()
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    return data_loader, model


def dummy_img():
    INPUT_W, INPUT_H,INPUT_C = 128,128,3
    MEAN = [128., ] * INPUT_C
    STD = [128., ] * INPUT_C
    # read pic
    img0 = cv2.imread("./14.png", cv2.IMREAD_COLOR)
    # img0 = np.random.randint(0, 255, [INPUT_H, INPUT_W, 3], dtype=np.uint8)
    h, w = img0.shape[:2]
    # ========================== preprocess img =============================
    img1 = cv2.resize(img0, (INPUT_W, INPUT_H))
    ori_image = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    # ori_image = img1
    if 1 == INPUT_C:
        ori_image = cv2.cvtColor(ori_image, cv2.COLOR_RGB2GRAY)
        input_image = ori_image.astype(np.float32) - np.asarray(MEAN)
        input_image /= np.asarray(STD)
        input_image = input_image[np.newaxis, ...]
    else:
        # [h,w,c]
        input_image = ori_image.astype(np.float32) - np.asarray([[MEAN]])
        input_image /= np.asarray([[STD]])
        input_image = np.transpose(input_image, [2, 0, 1])

    img_c, img_h, img_w = input_image.shape
    img_data = input_image[np.newaxis, :, :, :]
    return img_data


@torch.no_grad()
def model_test():
    _dataloader, _model = build_model()
    _num_class = len(_model.CLASSES)
    miss_num = {i: 0 for i in range(_num_class)}
    error_num, true_num = copy.deepcopy(miss_num), copy.deepcopy(miss_num)
    total_num = copy.deepcopy(miss_num)
    ignore_num = 0
    for i, data in enumerate(_dataloader):

        # if i >=1: break

        img_metas = data['img_metas'].data[0]
        img_tensors = data['img'].data[0]
        b, c, h, w = img_tensors.shape
        assert len(img_metas) == b

        print(img_tensors.dtype)
        a = np.asarray(dummy_img(), dtype=np.float32)
        img_tensors = torch.from_numpy(a)
        print(img_tensors)

        results = _model(return_loss=False, rescale=False, img=[img_tensors], img_metas=[img_metas])
        print(results)
        exit(10)
        for jj, meta in enumerate(img_metas):

            # print(meta)
            img_filepath = meta['filename']
            fname = osp.basename(img_filepath)
            ori_shape = meta['ori_shape']  # [h,w,c]
            mean = meta['img_norm_cfg']['mean']
            std = meta['img_norm_cfg']['std']
            gt_boxes = meta['gt_bboxes'].data
            gt_boxes_ignore = meta['gt_bboxes_ignore'].data
            gt_labels = meta['gt_labels'].data
            det_result = results[jj]    #[class_0_res, class_1_res, ...]

            # print('det_result: ', det_result)
            # print('det_result size: ', det_result)
            # print('gt_boxes: ', gt_boxes.numpy())
            for k, gt_label in enumerate(gt_labels.numpy()):
                total_num[gt_label] += 1

            ignore_num += gt_boxes_ignore.shape[0]

            miss_labels, error_labels = [], []
            miss_boxes, error_boxes = [], []
            true_labels, true_boxes = [], []
            if len(det_result) == 0:
                for k, gt_label in enumerate(gt_labels.numpy()):
                    miss_num[gt_label] += 1
                    miss_labels.append(gt_label)
                    miss_boxes.append(gt_boxes[k].numpy())
                continue

            for class_idx, det_bboxes in enumerate(det_result):
                det_bboxes = np.asarray(det_bboxes)  # [n,5]
                det_bboxes_src = det_bboxes[det_bboxes[:, -1] >= THRESHOLD]
                det_bboxes = det_bboxes_src[:, :4]
                _inds = (gt_labels == class_idx)
                _gt_boxes = gt_boxes[_inds]
                _gt_labels = gt_labels[_inds]

                # print('_gt_labels: ', _gt_labels)
                # print('_gt_boxes: ', _gt_boxes)
                # print(f'cls: {class_idx}, det_bboxes: {det_bboxes}')
                # continue
                det_num = det_bboxes.shape[0]
                gt_num = _gt_boxes.shape[0]
                gt_num2 = gt_boxes_ignore.shape[0]
                if det_num == 0 and gt_num > 0:
                    miss_num[class_idx] += 1
                    miss_labels.append(_gt_labels.numpy())
                    miss_boxes.append(_gt_boxes.numpy())
                elif gt_num == 0 and gt_num2 == 0 and det_num > 0:
                    error_num[class_idx] += 1
                    error_labels.append([class_idx for _ in range(det_num)])
                    error_boxes.append(det_bboxes_src)
                else:
                    _ious = bbox_overlaps(torch.from_numpy(det_bboxes), _gt_boxes, mode='iou')  # [n,m]
                    # print('_ious: ', _ious)
                    assign_inds = []
                    _tp_labels, _fp_labels = [], []
                    _tp_boxes, _fp_boxes = [], []
                    for i, iou_d2gts in enumerate(_ious):
                        is_assign = False
                        # foreach ious of det_box and gt_boxes
                        for j, iou in enumerate(iou_d2gts):
                            if iou >= TP_ASSIGN_IOU and class_idx == _gt_labels[j]:
                                is_assign = True
                                assign_inds.append(j)
                                _tp_labels.append(class_idx)
                                _tp_boxes.append(det_bboxes_src[i])
                                break

                        if not is_assign:
                            _fp_labels.append(class_idx)
                            _fp_boxes.append(det_bboxes_src[i])

                    # print(_tp_labels, _tp_boxes)

                    if len(_tp_labels)>0:
                        true_num[class_idx] += len(_tp_labels)
                        true_labels.append(_tp_labels)
                        true_boxes.append(_tp_boxes)
                    if len(_fp_labels)>0:
                        error_num[class_idx] += len(_fp_labels)
                        error_labels.append(_fp_labels)
                        error_boxes.append(_fp_boxes)

                    all_gt_inds = [i for i in range(_gt_boxes.size(0))]
                    if len(assign_inds) < len(all_gt_inds):
                        miss_inds = list(set(all_gt_inds).difference(set(assign_inds)))
                        if len(miss_inds) > 0:
                            miss_num[class_idx] += len(miss_inds)
                            miss_labels.append(_gt_labels[miss_inds].numpy())
                            miss_boxes.append(_gt_boxes[miss_inds].numpy())

            print(f'miss_num: {miss_num}', f'error_num: {error_num}', f'true_num: {true_num}',
                  f'total_num: {total_num}')
            _error_flag = len(error_labels) > 0
            _miss_flag = len(miss_labels) > 0
            _true_flag = len(true_labels) > 0
            # if not _error_flag and not _miss_flag:
            #     continue

            if _error_flag:
                # print('error_labels: ', error_labels)
                # print('error_boxes: ', error_boxes)
                error_labels = np.asarray(error_labels).reshape(-1)
                error_boxes = np.asarray(error_boxes).reshape(-1, 5)
            if _miss_flag:
                # print('miss_labels: ', miss_labels)
                # print('miss_boxes: ', miss_boxes)
                miss_labels = np.asarray(miss_labels).reshape(-1)
                miss_boxes = np.asarray(miss_boxes).reshape(-1, 4)

            if len(true_labels) > 0:
                true_labels = np.asarray(true_labels).reshape(-1)
                true_boxes = np.asarray(true_boxes).reshape(-1, 5)

            if SAVE_IMG_FLAG:
                img = img_tensors[jj].permute(1, 2, 0)  # [c,h,w]
                img = img * std + mean
                _ch = img.size(-1)
                img = np.asarray(img.numpy(), dtype=np.uint8)
                if _ch == 1:
                    img = cv2.cvtColor(np.squeeze(img), cv2.COLOR_GRAY2BGR)
                else:
                    img = img[:, :, ::-1]
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                h, w = img.shape[:2]
                for idx, (cls, box) in enumerate(zip(gt_labels, gt_boxes)):
                    x1, y1, x2, y2 = list(map(int, box))
                    cls = int(cls)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
                    cv2.putText(img, f"cls:{cls}", (x1, max(5, y1 - 5)),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)

                for idx, (cls, box) in enumerate(zip(true_labels, true_boxes)):
                    x1, y1, x2, y2 = list(map(int, box[:4]))
                    p = box[-1]
                    cls = int(cls)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 1)
                    # print(f'============= {x1,y1, x2,y2, p, cls}')
                    cv2.putText(img, "cls:{}({:.2f})".format(cls, p), (x1, min(h - 10, y2 + 5)),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)

                for idx, (cls, box) in enumerate(zip(miss_labels, miss_boxes)):
                    x1, y1, x2, y2 = list(map(int, box))
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 3)

                for idx, (cls, box) in enumerate(zip(error_labels, error_boxes)):
                    x1, y1, x2, y2 = list(map(int, box[:4]))
                    p = box[-1]
                    cls = int(cls)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
                    cv2.putText(img, "cls:{}({:.2f})".format(cls, p), (x1, min(h - 10, y2 + 5)),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)

                img = cv2.resize(img, dsize=(0, 0), fx=2, fy=2)
                if _miss_flag:
                    _savefilepath = f'{SAVE_DIR}/miss/{fname}'
                    os.makedirs(osp.dirname(_savefilepath), exist_ok=True)
                    cv2.imwrite(_savefilepath, img)
                if _error_flag:
                    _savefilepath = f'{SAVE_DIR}/error/{fname}'
                    os.makedirs(osp.dirname(_savefilepath), exist_ok=True)
                    cv2.imwrite(_savefilepath, img)
                if _true_flag:
                    _savefilepath = f'{SAVE_DIR}/true/{fname}'
                    os.makedirs(osp.dirname(_savefilepath), exist_ok=True)
                    cv2.imwrite(_savefilepath, img)

    #####################################################################
    _report_json_path = f'{SAVE_DIR}/report.json'
    _data = {i: 0 for i in range(_num_class)}
    for i in range(_num_class):
        _data[i] = {
            "total": total_num[i],
            "ignore": ignore_num,
            "miss": miss_num[i],
            "miss_p": miss_num[i] / total_num[i],
            "error": error_num[i],
            "error_p": error_num[i] / total_num[i],
            "true": true_num[i],
            "true_p": true_num[i] / total_num[i],
        }

    from pprint import pprint
    pprint(_data)
    os.makedirs(osp.dirname(_report_json_path), exist_ok=True)
    with open(_report_json_path, 'w', encoding='utf-8') as f:
        json.dump(_data, f, indent=4)

    print('done.')


if __name__ == '__main__':
    model_test()