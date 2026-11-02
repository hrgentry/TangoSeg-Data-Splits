_test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(reduce_zero_label=False, type='LoadAnnotations'),
    dict(thr=0, type='BinarizeSeg'),
    dict(keep_ratio=False, scale=(
        512,
        512,
    ), type='Resize'),
    dict(type='SyncOriShape'),
    dict(type='PackSegInputs'),
]
_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(reduce_zero_label=False, type='LoadAnnotations'),
    dict(thr=0, type='BinarizeSeg'),
    dict(keep_ratio=False, scale=(
        512,
        512,
    ), type='Resize'),
    dict(type='SyncOriShape'),
    dict(direction='horizontal', prob=0.5, type='RandomFlip'),
    dict(type='PackSegInputs'),
]
custom_imports = dict(
    allow_failed_imports=False,
    imports=[
        'mmpretrain.models',
        'projects.deepcrack',
        'projects.shuttlenet',
        'projects.dinov3_sealing.crack_modules',
        'projects.mixercseg_baselines.adapters',
        'projects.mixercseg_baselines.transforms',
        'projects.tangoseg',
        'projects.mixercseg',
    ])
data_preprocessor = dict(
    bgr_to_rgb=True,
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    pad_val=0,
    seg_pad_val=255,
    size=(
        512,
        512,
    ),
    std=[
        58.395,
        57.12,
        57.375,
    ],
    type='SegDataPreProcessor')
data_root = 'data/DeepCrack/dataset'
dataset_name = 'DeepCrack'
default_hooks = dict(
    checkpoint=dict(
        by_epoch=True,
        interval=1,
        max_keep_ckpts=3,
        rule='greater',
        save_best='fgIoU',
        type='CheckpointHook'),
    logger=dict(interval=10, log_metric_by_epoch=True, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='SegVisualizationHook'))
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='gloo'),
    mp_cfg=dict(mp_start_method='spawn', opencv_num_threads=0))
launcher = 'none'
load_from = 'pretrain/deeplabv3plus_r50-d8_512x512_160k_ade20k_20200615_124504-6135c7e0.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True)
max_epochs = 5
model = dict(
    auxiliary_head=dict(
        align_corners=False,
        channels=256,
        concat_input=False,
        dropout_ratio=0.1,
        in_channels=1024,
        in_index=2,
        loss_decode=[
            dict(ignore_index=255, loss_weight=0.4, type='IgnoreSafeCELoss'),
            dict(
                ignore_index=255,
                loss_weight=2.0,
                type='DiceLoss',
                use_sigmoid=False),
        ],
        norm_cfg=dict(requires_grad=True, type='BN'),
        num_classes=2,
        num_convs=1,
        type='FCNHead'),
    backbone=dict(
        contract_dilation=True,
        depth=50,
        dilations=(
            1,
            1,
            2,
            4,
        ),
        norm_cfg=dict(requires_grad=True, type='BN'),
        norm_eval=False,
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        strides=(
            1,
            2,
            1,
            1,
        ),
        style='pytorch',
        type='ResNetV1c'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_val=0,
        seg_pad_val=255,
        size=(
            512,
            512,
        ),
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='SegDataPreProcessor'),
    decode_head=dict(
        align_corners=False,
        c1_channels=48,
        c1_in_channels=256,
        channels=512,
        dilations=(
            1,
            12,
            24,
            36,
        ),
        dropout_ratio=0.1,
        in_channels=2048,
        in_index=3,
        loss_decode=[
            dict(ignore_index=255, loss_weight=1.0, type='IgnoreSafeCELoss'),
            dict(
                ignore_index=255,
                loss_weight=5.0,
                type='DiceLoss',
                use_sigmoid=False),
        ],
        norm_cfg=dict(requires_grad=True, type='BN'),
        num_classes=2,
        type='DepthwiseSeparableASPPHead'),
    pretrained=None,
    test_cfg=dict(mode='whole'),
    train_cfg=dict(),
    type='EncoderDecoder')
model_name = 'deeplabv3plus_r50'
norm_cfg = dict(requires_grad=True, type='BN')
num_workers = 4
optim_wrapper = dict(
    clip_grad=None,
    optimizer=dict(lr=0.0005, type='AdamW', weight_decay=0.01),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0, by_epoch=True, end=5, eta_min=1e-06, power=1.0,
        type='PolyLR'),
]
randomness = dict(seed=42)
resume = False
test_batch_size = 8
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=8,
    dataset=dict(
        ann_file='splits/test.txt',
        data_root='data/DeepCrack/dataset',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(reduce_zero_label=False, type='LoadAnnotations'),
            dict(thr=0, type='BinarizeSeg'),
            dict(keep_ratio=False, scale=(
                512,
                512,
            ), type='Resize'),
            dict(type='SyncOriShape'),
            dict(type='PackSegInputs'),
        ],
        test_mode=True,
        type='DeepCrackPairDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    iou_metrics=[
        'mIoU',
        'mDice',
        'mFscore',
    ], type='BinaryFgIoUMetric')
train_batch_size = 8
train_cfg = dict(max_epochs=5, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_size=8,
    dataset=dict(
        ann_file='splits/train.txt',
        data_root='data/DeepCrack/dataset',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(reduce_zero_label=False, type='LoadAnnotations'),
            dict(thr=0, type='BinarizeSeg'),
            dict(keep_ratio=False, scale=(
                512,
                512,
            ), type='Resize'),
            dict(type='SyncOriShape'),
            dict(direction='horizontal', prob=0.5, type='RandomFlip'),
            dict(type='PackSegInputs'),
        ],
        type='DeepCrackPairDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
val_batch_size = 8
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=8,
    dataset=dict(
        ann_file='splits/val.txt',
        data_root='data/DeepCrack/dataset',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(reduce_zero_label=False, type='LoadAnnotations'),
            dict(thr=0, type='BinarizeSeg'),
            dict(keep_ratio=False, scale=(
                512,
                512,
            ), type='Resize'),
            dict(type='SyncOriShape'),
            dict(type='PackSegInputs'),
        ],
        test_mode=True,
        type='DeepCrackPairDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    iou_metrics=[
        'mIoU',
        'mDice',
        'mFscore',
    ], type='BinaryFgIoUMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
    ])
work_dir = 'work_dirs/matrix_final/deepcrack/deeplabv3plus_r50'
