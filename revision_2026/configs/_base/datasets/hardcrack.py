# HardCrack v1.0 -- the paper's only in-domain benchmark (公开集实验计划 v0.3 D7).
#
# Split-file roles (verified 2026-09-11):
#   train_tiles.txt: official input, 20,996 tile image/mask pairs.
#   train.txt: inventory of 3,412 train originals, not this loader's input.
#   unique train_manifest.jsonl image_id: 3,163 contributing parents;
#       249 inventory originals contribute no official tiles.
#   An ID-only parent list is provenance, not a two-column dataset input.
# Re-sampling full images changes the protocol, even with the same parents.
# Keep ann_file=train_tiles.txt; record actual split/manifest/data version hashes.
# See paper/wavecrack_mamba/data/HardCrack_DATASHEET.md, section 2.4.1.
#
# Protocol (hardcrack_tiling_split_方案_v0.2 §5 / 组织方案 v0.1 §5):
#   train    = the official 512x512 tile set (tiles_train_v1, 20,996 tiles
#              selected from train images only, multi-scale, stride 384);
#   val/test = the UNTOUCHED full-resolution source images (srcA 1200x900,
#              srcB 3854x2065, srcC 4096x2048) scored by sliding-window inference
#              with a 512 window and 384 stride -- the same grid the tiles were cut on.
#
# Why no Resize here, unlike the public-set bases: their Resize((512,512)) is the
# patch protocol of ~512px datasets; on a 4096x2048 srcC image it is an 8x
# down-sample that erases exactly the thin cracks HardCrack exists to measure.
# Tiles are already 512x512, so Resize/SyncOriShape would be identities on the
# train side and are omitted too.
#
# Masks are {0,255}; BinarizeSeg(thr=0) -> {0,1} (canonical repo convention).
#
# Two things every HardCrack leaf config must carry (they cannot live here):
#   model = dict(test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(384, 384)))
#       -- mmengine forbids a second base overriding `model`;
#   val/test batch_size = 1 -- SegDataPreProcessor stacks a test batch only when
#       every image has the same size, and HardCrack mixes three sizes.
dataset_name = 'HardCrack'
data_root = 'data/HardCrack'  # junction -> D:/Data/hardcrack/HardCrack_v1.0 (release copy, neutral ids)

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

# Native resolution: no Resize, no SyncOriShape; the model's slide test_cfg does
# the rest and predictions come back at ori_shape for the metric.
_test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='BinarizeSeg', thr=0),
    dict(type='PackSegInputs'),
]

train_dataloader = dict(
    batch_size=1,  # placeholder; every leaf sets its own (8, or 2 x accumulate 4)
    num_workers=4,
    persistent_workers=True,
    # Keep full batches under the established protocol; leaf batch sizes vary.
    # The current manifest count is documented above.
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
