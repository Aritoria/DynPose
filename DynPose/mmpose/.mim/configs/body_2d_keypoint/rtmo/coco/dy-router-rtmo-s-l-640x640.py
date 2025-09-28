_base_ = ['../../../_base_/default_runtime.py']

# runtime
train_cfg = dict(max_epochs=1, val_interval=1, dynamic_intervals=[(580, 1)])

# auto_scale_lr = dict(base_batch_size=256)

default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=5))
# optim_wrapper = dict(
#     clip_grad=dict(max_norm=0.1, norm_type=2),
#     constructor='ForceDefaultOptimWrapperConstructor',
#     optimizer=dict(lr=1e-05, type='AdamW', weight_decay=0.005),
#     paramwise_cfg=dict(
#         bias_decay_mult=0,
#         bypass_duplicate=True,
#         custom_keys=dict({
#             'head':dict(decay_mult=0, lr_mult=0),
#         }),
#         force_default_settings=True,
#         norm_decay_mult=0),
#     type='OptimWrapper')
# param_scheduler = [
#     dict(
#         begin=0,
#         by_epoch=True,
#         convert_to_iter_based=True,
#         end=5,
#         type='QuadraticWarmupLR'),
#     dict(
#         T_max=280,
#         begin=5,
#         by_epoch=True,
#         convert_to_iter_based=True,
#         end=280,
#         eta_min=0.0002,
#         type='CosineAnnealingLR'),
#     dict(begin=280, by_epoch=True, end=281, factor=2.5, type='ConstantLR'),
#     dict(
#         T_max=300,
#         begin=281,
#         by_epoch=True,
#         convert_to_iter_based=True,
#         end=580,
#         eta_min=0.0002,
#         type='CosineAnnealingLR'),
#     dict(begin=580, by_epoch=True, end=600, factor=1, type='ConstantLR'),
# ]
optim_wrapper = dict(
    optimizer=dict(lr=1e-04, type='Adam', weight_decay=0.005),
    paramwise_cfg=dict(
        custom_keys=dict({
            'head': dict(decay_mult=0, lr_mult=0),
            'neck': dict(decay_mult=0, lr_mult=0),
        }
        )
    )
)

# learning policy
param_scheduler = [
    dict(
        type='LinearLR', begin=0, end=5, start_factor=0.01,
        by_epoch=False),  # warm-up
    # dict(
    #     type='MultiStepLR',
    #     begin=0,
    #     end=5,
    #     milestones=[170, 200],
    #     gamma=0.1,
    #     by_epoch=True),
    # dict(
    #     type='QuadraticWarmupLR',
    #     by_epoch=True,
    #     begin=0,
    #     end=5,
    #     convert_to_iter_based=True)
]

# data
input_size = (640, 640)
metafile = 'configs/_base_/datasets/coco.py'
codec = dict(type='YOLOXPoseAnnotationProcessor', input_size=input_size)

train_pipeline_stage1 = [
    dict(type='LoadImage', backend_args=None),
    dict(
        type='Mosaic',
        img_scale=(640, 640),
        pad_val=114.0,
        pre_transform=[dict(type='LoadImage', backend_args=None)]),
    dict(
        type='BottomupRandomAffine',
        input_size=(640, 640),
        shift_factor=0.1,
        rotate_factor=10,
        scale_factor=(0.75, 1.0),
        pad_val=114,
        distribution='uniform',
        transform_mode='perspective',
        bbox_keep_corner=False,
        clip_border=True,
    ),
    dict(
        type='YOLOXMixUp',
        img_scale=(640, 640),
        ratio_range=(0.8, 1.6),
        pad_val=114.0,
        pre_transform=[dict(type='LoadImage', backend_args=None)]),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip'),
    dict(type='FilterAnnotations', by_kpt=True, by_box=True, keep_empty=False),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs'),
]
train_pipeline_stage2 = [
    dict(type='LoadImage'),
    dict(
        type='BottomupRandomAffine',
        input_size=(640, 640),
        shift_prob=0,
        rotate_prob=0,
        scale_prob=0,
        scale_type='long',
        pad_val=(114, 114, 114),
        bbox_keep_corner=False,
        clip_border=True,
    ),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip'),
    dict(type='BottomupGetHeatmapMask', get_invalid=True),
    dict(type='FilterAnnotations', by_kpt=True, by_box=True, keep_empty=False),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs'),
]

data_mode = 'bottomup'
data_root = 'data/'

val_pipeline = [
    dict(type='LoadImage'),
    dict(
        type='BottomupResize', input_size=input_size, pad_val=(114, 114, 114)),
    dict(
        type='PackPoseInputs',
        meta_keys=('id', 'img_id', 'img_path', 'ori_shape', 'img_shape',
                   'input_size', 'input_center', 'input_scale'))
]

valsim_pipeline = [
    dict(type='LoadImage'),
    dict(type='BottomupResize', input_size=input_size, pad_val=(114, 114, 114)),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs'),
]


# train datasets
dataset_coco = dict(
    type='CocoDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='coco/annotations/person_keypoints_train2017.json',
    data_prefix=dict(img='coco/train2017/'),
    pipeline=train_pipeline_stage1,
)

# 修改pip之后的
dataset_coco_new_pip = dict(
    type='CocoDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='coco/annotations/person_keypoints_train2017.json',
    data_prefix=dict(img='coco/train2017/'),
    pipeline=valsim_pipeline,
)

val_coco = dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        data_mode=data_mode,
        ann_file='coco/annotations/person_keypoints_val2017.json',
        data_prefix=dict(img='coco/val2017/'),
        pipeline=val_pipeline,
    )


train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    # dataset=dataset_coco
    dataset=dataset_coco_new_pip
)



val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        data_mode=data_mode,
        ann_file='coco/annotations/person_keypoints_val2017.json',
        data_prefix=dict(img='coco/val2017/'),
        test_mode=True,
        pipeline=val_pipeline,
    ))
test_dataloader = val_dataloader

# evaluators
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'coco/annotations/person_keypoints_val2017.json',
    score_mode='bbox',
    nms_mode='none',
)
test_evaluator = val_evaluator

# hooks
custom_hooks = [
    dict(
        type='YOLOXPoseModeSwitchHook',
        num_last_epochs=20,
        new_train_pipeline=train_pipeline_stage2,
        priority=48),
    dict(
        type='RTMOModeSwitchHook',
        epoch_attributes={
            280: {
                'proxy_target_cc': True,
                'loss_mle.loss_weight': 5.0,
                'loss_oks.loss_weight': 10.0
            },
        },
        priority=48),
    dict(type='SyncNormHook', priority=48),
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        strict_load=False,
        priority=49),
]

# model
widen_factor = 0.5
deepen_factor = 0.33

model = dict(
    type='BottomupPoseEstimator',
    init_cfg=dict(
        type='Kaiming',
        layer='Conv2d',
        a=2.23606797749979,
        distribution='uniform',
        mode='fan_in',
        nonlinearity='leaky_relu'),
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        pad_size_divisor=32,
        mean=[0, 0, 0],
        std=[1, 1, 1],
        batch_augments=[
            dict(
                type='BatchSyncRandomResize',
                random_size_range=(480, 800),
                size_divisor=32,
                interval=1),
        ]),
    backbone=dict(
        type='ImageRouterRTMO',
    ),
    head=dict(
        type='TwoHeadRTMO',
        num_keypoints=17,
        featmap_strides=(16, 32),
        head_module_cfg=dict(
            num_classes=1,
            in_channels=256,
            cls_feat_channels=256,
            channels_per_group=36,
            pose_vec_channels=256,
            widen_factor=widen_factor,
            stacked_convs=2,
            norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg=dict(type='Swish')),
        assigner=dict(
            type='SimOTAAssigner',
            dynamic_k_indicator='oks',
            oks_calculator=dict(type='PoseOKS', metainfo=metafile),
            use_keypoints_for_center=True),
        prior_generator=dict(
            type='MlvlPointGenerator',
            centralize_points=True,
            strides=[16, 32]),
        dcc_cfg=dict(
            in_channels=256,
            feat_channels=128,
            num_bins=(192, 256),
            spe_channels=128,
            gau_cfg=dict(
                s=128,
                expansion_factor=2,
                dropout_rate=0.0,
                drop_path=0.0,
                act_fn='SiLU',
                pos_enc='add')),
        overlaps_power=0.5,
        loss_cls=dict(
            type='VariFocalLoss',
            reduction='sum',
            use_target_weight=True,
            loss_weight=1.0),
        loss_bbox=dict(
            type='IoULoss',
            mode='square',
            eps=1e-16,
            reduction='sum',
            loss_weight=5.0),
        loss_oks=dict(
            type='OKSLoss',
            reduction='none',
            metainfo=metafile,
            loss_weight=30.0),
        loss_vis=dict(
            type='BCELoss',
            use_target_weight=True,
            reduction='mean',
            loss_weight=1.0),
        loss_mle=dict(
            type='MLECCLoss',
            use_target_weight=True,
            loss_weight=1.0,
        ),
        loss_bbox_aux=dict(type='L1Loss', reduction='sum', loss_weight=1.0),
    ),
    test_cfg=dict(
        input_size=input_size,
        score_thr=0.1,
        nms_thr=0.65,
    ))

find_unused_parameters = True
