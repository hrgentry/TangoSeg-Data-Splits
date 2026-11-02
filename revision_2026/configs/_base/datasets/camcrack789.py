dataset_name = 'CamCrack789'
data_root = 'data/CamCrack789/dataset'

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=(512, 512))

_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='BinarizeSeg', thr=0),
    dict(type='Resize', scale=(512, 512), keep_ratio=False),
    dict(type='SyncOriShape'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackSegInputs'),
]

_test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='BinarizeSeg', thr=0),
    dict(type='Resize', scale=(512, 512), keep_ratio=False),
    dict(type='SyncOriShape'),
    dict(type='PackSegInputs'),
]

train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    # 553 % 8 == 1 → last micro-batch size 1 breaks BN (N=1,H=W=1).
    drop_last=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='DeepCrackPairDataset',
        data_root=data_root,
        ann_file='splits/train.txt',
        pipeline=_train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='DeepCrackPairDataset',
        data_root=data_root,
        ann_file='splits/val.txt',
        pipeline=_test_pipeline,
        test_mode=True))

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='DeepCrackPairDataset',
        data_root=data_root,
        ann_file='splits/test.txt',
        pipeline=_test_pipeline,
        test_mode=True))
