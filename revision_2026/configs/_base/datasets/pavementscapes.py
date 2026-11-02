# Pavementscapes (Tong et al., arXiv 2208.00775) as a binary crack benchmark, 2026-10-14.
#
# Source: survey-vehicle top-down frames, 2048x1024 grayscale, official split 2500/500/1000.
# Masks store class*30; classes 1-3 (longitudinal / lateral / alligator crack, thin
# crack-following polygons) -> crack, classes 4-6 (area distress: pothole, repair, rut;
# identified by shape, see benchmarks/pavementscapes/README.md) -> background.
# Binary masks: crack_bin/<split>/*.png {0,255}, built from mask/mask/ (originals untouched).
#
# Protocol = HardCrack's, since both are full-resolution survey frames:
#   train    = 512x512 tiles of the train frames (tiles_train/, 5x3 grid, stride 384 in x,
#              rows 0/384/512 in y), tiles with >=50 crack px + 10% crack-free tiles
#              (seed 0) -> splits/train_tiles.txt;
#   val/test = untouched 2048x1024 frames, 512/384 sliding window (set in each leaf).
dataset_name = 'PavementScapes'
data_root = 'data/PavementScapes'  # junction -> D:/Data/PavementScapes_Segmentation

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
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackSegInputs'),
]

_test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='BinarizeSeg', thr=0),
    dict(type='PackSegInputs'),
]

train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='DeepCrackPairDataset',
        data_root=data_root,
        ann_file='splits/train_tiles.txt',
        pipeline=_train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
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
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='DeepCrackPairDataset',
        data_root=data_root,
        ann_file='splits/test.txt',
        pipeline=_test_pipeline,
        test_mode=True))
