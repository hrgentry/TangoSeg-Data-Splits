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
        'projects.deepcrack',
        'projects.dinov3_sealing.crack_modules',
        'projects.mixercseg_baselines.adapters',
        'projects.mixercseg_baselines.transforms',
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
data_root = 'data/Crack500/dataset'
dataset_name = 'Crack500'
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
log_level = 'INFO'
log_processor = dict(by_epoch=True)
max_epochs = 5
model = dict(
    base_channel=32,
    bce_loss_weight=1.0,
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            127.5,
            127.5,
            127.5,
        ],
        pad_val=0,
        seg_pad_val=255,
        size=(
            512,
            512,
        ),
        std=[
            127.5,
            127.5,
            127.5,
        ],
        type='SegDataPreProcessor'),
    dice_loss_weight=5.0,
    fuse_channels=64,
    num_res=(
        4,
        4,
        4,
    ),
    test_cfg=dict(mode='whole'),
    threshold=0.5,
    train_cfg=dict(),
    type='RestorMixerSegmentor')
model_name = 'restormixer'
num_workers = 4
optim_wrapper = dict(
    accumulative_counts=4,
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
    batch_size=2,
    dataset=dict(
        ann_file='splits/test.txt',
        data_root='data/Crack500/dataset',
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
    num_workers=2,
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
    batch_size=2,
    dataset=dict(
        ann_file='splits/train.txt',
        data_root='data/Crack500/dataset',
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
    drop_last=True,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
val_batch_size = 8
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=2,
    dataset=dict(
        ann_file='splits/val.txt',
        data_root='data/Crack500/dataset',
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
    num_workers=2,
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
work_dir = 'work_dirs/matrix_final/crack500/restormixer'
