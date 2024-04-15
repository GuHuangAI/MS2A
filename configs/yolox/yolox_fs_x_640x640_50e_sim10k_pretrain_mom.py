# -*- coding: utf-8 -*-
# coding=utf-8
_base_ = ['../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py']
img_scale = (640, 640)

# model settings
model = dict(
    type='YOLOX_FS',
    input_size=img_scale,
    # random_size_range=(36, 44),
    random_size_range=(15, 25),
    random_size_interval=10,
    backbone=dict(type='CSPDarknet', deepen_factor=1.33, widen_factor=1.25),
    neck=dict(
        type='YOLOXPAFPN',
        in_channels=[320, 640, 1280], out_channels=320, num_csp_blocks=4),
    bbox_head=dict(
        type='YOLOX_FS_Head_MulScale_Mom',
        window_sizes1=[[10, 10], [10, 10], [10, 10]],
        window_sizes2=[
                        [[10, 10], [5, 5], [2, 2]],
                        [[10, 10], [5, 5], [2, 2]],
                        [[10, 10], [5, 5], [2, 2]],
                        ],
        in_channel2=100,
        cluster=dict(cityscapes='/benchmark/持续学习数据集/projects/few_shot/fewshot_fea/sim10k/sim10k_cluster.pth',
        foggy_cityscapes='/benchmark/持续学习数据集/projects/few_shot/fewshot_fea/sim10k/citycar_cluster.pth',),
        att_layers=2, num_classes=1, in_channels=320, feat_channels=320,
        loss_cls=dict(
                     type='VarifocalLoss',
                     use_sigmoid=True,
                     reduction='sum',
                     loss_weight=1.0),
                 loss_fg=dict(
                     type='DiceLoss',
                     use_sigmoid=True,
                     loss_weight=2.0),
                 loss_bbox=dict(
                     type='CIoULoss',
                     eps=1e-16,
                     reduction='sum',
                     loss_weight=2.0),
                 loss_obj=dict(
                     type='VarifocalLoss',
                     use_sigmoid=True,
                     reduction='sum',
                     loss_weight=1.0),
                 loss_l1=dict(type='SmoothL1Loss', reduction='sum', loss_weight=1.0),
        ),
    train_cfg=dict(assigner=dict(type='SimOTAAssigner', center_radius=2.5)),
    # In order to align the source code, the threshold of the val phase is
    # 0.01, and the threshold of the test phase is 0.001.
    test_cfg=dict(score_thr=0.01, nms=dict(type='soft_nms', iou_threshold=0.5), max_per_img=100),
    finetune=False)

# dataset settings
data_root = '/nfs/public_data/cityscapes_COCO_format/'
dataset_type = 'CocoDataset'

train_pipeline = [
    # dict(type='Mosaic', img_scale=img_scale, pad_val=114.0),
    dict(type='Resize', img_scale=img_scale, keep_ratio=True),
    dict(
        type='Pad',
        pad_to_square=True,
        # If the image is three-channel, the pad value needs
        # to be set separately for each channel.
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(
        type='CopyPaste',
        max_num_pasted=100,
        bbox_occluded_thr=10,
        mask_occluded_thr=300),
    dict(type='Mosaic', img_scale=img_scale, pad_val=114.0),
    dict(
        type='RandomAffine',
        scaling_ratio_range=(0.5, 2),
        border=(-img_scale[0] // 2, -img_scale[1] // 2)),
    #dict(
    #    type='MixUp',
    #    img_scale=img_scale,
    #    ratio_range=(0.5, 1.5),
    #    pad_val=114.0),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', flip_ratio=0.5),
    # According to the official implementation, multi-scale
    # training is not considered here but in the
    # 'mmdet/models/detectors/yolox.py'.
    # dict(type='Resize', img_scale=img_scale, keep_ratio=True),
    # dict(
    #     type='Pad',
    #     pad_to_square=True,
    #     # If the image is three-channel, the pad value needs
    #     # to be set separately for each channel.
    #     pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'env_id', 'gt_bboxes', 'gt_labels'])
]
    
classes = ('car',)

train_dataset = dict(
    type='MixPasteConcatDataset',
    datasets=[
        dict(
        type=dataset_type,
        classes=classes,
        env_id=0,
        ann_file="/benchmark/持续学习数据集/public_data/SIM10k_COCO_format/annotations/only_car_instances_train2014.json",
        img_prefix="/benchmark/持续学习数据集/public_data/SIM10k_COCO_format/train2014/",
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True)
        ],
        filter_empty_gt=False,
        ),
        dict(
        type=dataset_type,
        classes=classes,
        env_id=1,
        ann_file="/benchmark/持续学习数据集/public_data/cityscapes_COCO_format/annotations/only_car_instances_train2014_8.json",
        img_prefix="/benchmark/持续学习数据集/public_data/cityscapes_COCO_format/train2014/",
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True)
        ],
        filter_empty_gt=False,
        )
    ],
    pipeline=train_pipeline,
    additional_affine=dict(
        type='RandomAffine',
        scaling_ratio_range=(0.2, 0.5),
        border=(-img_scale[0] // 4, -img_scale[1] // 4)),
    additional_resize=[dict(type='Resize', img_scale=img_scale, keep_ratio=True),
                       dict(
                           type='Pad',
                           pad_to_square=True,
                           pad_val=dict(img=(114.0, 114.0, 114.0))),],
    repeat_times=[1, 10])


test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale= (640, 640),
        #img_scale=[(480, 480), (640, 640), (800, 800)],
        flip=False,
        #flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Pad',
                pad_to_square=True,
                pad_val=dict(img=(114.0, 114.0, 114.0))),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'env_id'])
        ])
]

val_21 = dict(
    classes=classes,
    type=dataset_type,
    ann_file="/benchmark/持续学习数据集/public_data/SIM10k_COCO_format/annotations/only_car_instances_val2014.json",
    img_prefix='/benchmark/持续学习数据集/public_data/SIM10k_COCO_format/train2014/',
    env_id=0,
    pipeline=test_pipeline)

val_22 = dict(
    classes=classes,
    type=dataset_type,
    ann_file="/benchmark/持续学习数据集/public_data/cityscapes_COCO_format/annotations/only_car_instances_val2014.json",
    img_prefix='/benchmark/持续学习数据集/public_data/cityscapes_COCO_format/val2014/',
    env_id=1,
    pipeline=test_pipeline)
    
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    persistent_workers=True,
    train=train_dataset,
    val=[val_22, val_21],
    test=[val_22],
    )

# optimizer
# default 8 gpu
optimizer = dict(
    type='SGD',
    lr=0.002,
    momentum=0.9,
    weight_decay=5e-4,
    nesterov=True,
    #paramwise_cfg=dict(norm_decay_mult=0., bias_decay_mult=0.)
    paramwise_cfg=dict(
        custom_keys={'norm': dict(decay_mult=0.), 'bias': dict(decay_mult=0.)})
    )
optimizer_config = dict(grad_clip=None)

max_epochs = 50
num_last_epochs = 5
resume_from = None
interval = 10
load_from = "/home/ml/code/mmdetection/work_dirs/yolox_fs_x_640x640_50e_sim10k_base/best_bbox_mAP_epoch_70.pth"

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
log_config = dict(interval=10)
auto_scale_lr = dict(base_batch_size=4)
fp16 = dict(loss_scale=512.)
find_unused_parameters = True