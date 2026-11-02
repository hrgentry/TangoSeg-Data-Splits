dataset_name = 'Crack500'
data_root = 'data/Crack500/dataset'

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
    # 2355 train images is ODD, so any even micro-batch (e.g. the batch_size=2 +
    # accumulative_counts=4 form the heavy models use) ends the epoch on a
    # 1-sample batch. BN over a 1x1 spatial map then raises "Expected more than
    # 1 value per channel when training" -- this killed every
    # crack500/convnext_xlarge run for ~10h on 2026-08-29. The camcrack789 base
    # has carried this flag from the start; crack500's never got it.
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
