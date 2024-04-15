# -*- coding: utf-8 -*-
# coding=utf-8
_base_ = ['../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py']
img_scale = (640, 640)

# model settings
model = dict(
    type='YOLOX',
    input_size=img_scale,
    random_size_range=(36, 44),
    random_size_interval=10,
    backbone=dict(type='CSPDarknet', deepen_factor=1.33, widen_factor=1.25),
    neck=dict(
        type='YOLOXPAFPN',
        in_channels=[320, 640, 1280], out_channels=320, num_csp_blocks=4),
    bbox_head=dict(
        type='YOLOXHead', num_classes=1, in_channels=320, feat_channels=320),
    train_cfg=dict(assigner=dict(type='SimOTAAssigner', center_radius=2.5)),
    # In order to align the source code, the threshold of the val phase is
    # 0.01, and the threshold of the test phase is 0.001.
    test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.6)))

# dataset settings
dataset_type = 'CocoDataset'

train_pipeline = [
    dict(type='Mosaic', img_scale=img_scale, pad_val=114.0),
    dict(
        type='RandomAffine',
        scaling_ratio_range=(0.1, 2),
        border=(-img_scale[0] // 2, -img_scale[1] // 2)),
    dict(
        type='MixUp',
        img_scale=img_scale,
        ratio_range=(0.8, 1.6),
        pad_val=114.0),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Resize', img_scale=img_scale, keep_ratio=True),
    
    dict(type='Target_aug', img_scale=img_scale),
    
    dict(
        type='Pad',
        pad_to_square=True,
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
    ]

classes = ('part',)

train_dataset_1 = dict(
    type='MultiImageMixDataset',
    dataset=dict(
        classes=classes,
        type=dataset_type,
        ann_file="/nfs/testset/Indus_S_COCO_format/annotations/instances_train2017.json",
        img_prefix="/nfs/testset/Indus_S_COCO_format/train2017/",
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True)
        ],
        filter_empty_gt=False,
    ),
    pipeline=train_pipeline)

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Pad',
                pad_to_square=True,
                pad_val=dict(img=(114.0, 114.0, 114.0))),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]

val_21 = dict(
    classes=classes,
    type=dataset_type,
    ann_file="/nfs/testset/Indus_S_COCO_format/annotations/instances_val2017.json",
    img_prefix="/nfs/testset/Indus_S_COCO_format/val2017/",
    pipeline=test_pipeline)
    
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    persistent_workers=True,
    train=train_dataset_1,
    val=val_21,
    test=val_21,
    )

# optimizer
# default 8 gpu
optimizer = dict(
    type='SGD',
    lr=0.001,
    momentum=0.9,
    weight_decay=5e-4,
    nesterov=True,
    #paramwise_cfg=dict(norm_decay_mult=0., bias_decay_mult=0.)
    paramwise_cfg=dict(
        custom_keys={'norm': dict(decay_mult=0.), 'bias': dict(decay_mult=0.), 'clus_m': dict(lr_mult=0.2)})
    )
optimizer_config = dict(grad_clip=None)

max_epochs = 70
num_last_epochs = 10
resume_from = None
interval = 10
load_from = "/home/ml/code/mmdetection/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth"

# learning policy
lr_config = dict(
    _delete_=True,
    policy='YOLOX',
    warmup='exp',
    by_epoch=False,
    warmup_by_epoch=True,
    warmup_ratio=1,
    warmup_iters=3,  # 5 epoch
    num_last_epochs=num_last_epochs,
    min_lr_ratio=0.01)

runner = dict(type='EpochBasedRunner', max_epochs=max_epochs)

custom_hooks = [
    dict(
        type='YOLOXModeSwitchHook',
        num_last_epochs=num_last_epochs,
        priority=48),
    dict(
        type='SyncNormHook',
        num_last_epochs=num_last_epochs,
        interval=interval,
        priority=48),
    dict(
        type='ExpMomentumEMAHook',
        resume_from=resume_from,
        momentum=0.0001,
        priority=49)
]
checkpoint_config = dict(interval=interval)
evaluation = dict(
    save_best='auto',
    # The evaluation interval is 'interval' when running epoch is
    # less than ?max_epochs - num_last_epochs?.
    # The evaluation interval is 1 when running epoch is greater than
    # or equal to ?max_epochs - num_last_epochs?.
    interval=interval,
    dynamic_intervals=[(max_epochs - num_last_epochs, 1)],
    metric='bbox')
log_config = dict(interval=100)
auto_scale_lr = dict(base_batch_size=2)
fp16 = dict(loss_scale=512.)
find_unused_parameters = True
