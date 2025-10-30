_base_ = ['../../../_base_/default_runtime.py']

# runtime
train_cfg = dict(max_epochs=1, val_interval=1)

default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=1))

# optimizer
optim_wrapper = dict(
    optimizer=dict(lr=1e-04, type='Adam', weight_decay=0.005),
    paramwise_cfg=dict(
        custom_keys=dict({
            'head.net1': dict(decay_mult=0, lr_mult=0),
            'head.net2': dict(decay_mult=0, lr_mult=0),
        }
        )
    )
)

# learning policy
param_scheduler = [
    dict(begin=0, by_epoch=False, end=5, start_factor=0.01, type='LinearLR'),  # warm-up
]
# automatically scaling LR based on the actual training batch size
# auto_scale_lr = dict(base_batch_size=512)

# hooks
default_hooks = dict(checkpoint=dict(save_best='coco/AP', rule='greater'))

# codec settings
codec = dict(
    type='MSRAHeatmap', input_size=(192, 256), heatmap_size=(48, 64), sigma=2)


# model settings
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='ImageRouter',
    ),
    head=dict(
        type='TwoHead',
        loss=dict(type='KeypointMSELoss', use_target_weight=True),
        decoder=codec),
    test_cfg=dict(
        flip_test=True,
        flip_mode='heatmap',
        shift_heatmap=True,
    )
)

# base dataset settings
dataset_type = 'CocoDataset'
data_mode = 'topdown'
data_root = 'data/coco/'

# pipelines
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    # dict(type='RandomFlip', direction='horizontal'),
    # dict(type='RandomHalfBody'),
    # dict(type='RandomBBoxTransform'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]
val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs')
]

# data loaders
train_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/person_keypoints_train2017.json',
        data_prefix=dict(img='train2017/'),
        pipeline=train_pipeline,
    ),
)

# val 测试
val_dataloader = dict(
    batch_size=256,
    # batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    # sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    sampler=dict(type='DefaultSampler', shuffle=True, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/person_keypoints_val2017.json',
        bbox_file='data/coco/person_detection_results/'
        'COCO_val2017_detections_AP_H_56_person.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,
        pipeline=val_pipeline,
    ))
test_dataloader = val_dataloader

# evaluators
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/person_keypoints_val2017.json')
test_evaluator = val_evaluator

find_unused_parameters = True


# test 测试
# val_dataloader = dict(
#     batch_size=128,
#     # batch_size=1,
#     num_workers=8,
#     persistent_workers=True,
#     drop_last=False,
#     sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         data_mode=data_mode,
#         ann_file='annotations/image_info_test-dev2017.json',
#         bbox_file='data/coco/person_detection_results/'
#         'COCO_test-dev2017_detections_AP_H_609_person.json',
#         data_prefix=dict(img='test2017/'),
#         test_mode=True,
#         pipeline=val_pipeline,
#     ))
# test_dataloader = val_dataloader
#
# # evaluators
# val_evaluator = dict(
#     type='CocoMetric',
#     ann_file=data_root + 'annotations/person_keypoints_val2017.json')
# test_evaluator = val_evaluator


