# Pavementscapes leaf = configs/hardcrack/convnext_xlarge_512x512_b8_5ep.py with the dataset base swapped (2026-10-14).
# Deltas vs the public-set leaf: dataset base -> hardcrack.py; val/test batch_size -> 1
# (full-resolution images of three sizes cannot be stacked); model.test_cfg -> sliding
# window (HardCrack protocol; set in the leaf because mmengine forbids a second base
# overriding `model`). Everything else (schedule, per-model batch/optimizer) is the
# same-track b8_5ep protocol unchanged.
_base_ = [
    '../../_base/runtime.py',
    '../../_base/schedule_b8_5ep.py',
    '../../_base/datasets/pavementscapes.py',
    '../../_base/models/convnext_xlarge.py',
]

# Effective batch 8 as 2 x accumulate 4: the recipe of every public-matrix ConvNeXt-XL run
# (configs/<public>/convnext_xlarge_512x512_b8_5ep.py). At micro-batch 8 the card reserves
# ~32 GB and WDDM spills to system RAM (iteration time 1.05 -> 2.1 s from epoch 4, 2026-10-15).
train_dataloader = dict(batch_size=2, num_workers=4)
optim_wrapper = dict(accumulative_counts=4)
val_dataloader = dict(batch_size=1, num_workers=4)
test_dataloader = dict(batch_size=1, num_workers=4)

work_dir = './work_dirs/mixercseg_baselines/pavementscapes/convnext_xlarge_512x512_b8_5ep'

# HardCrack protocol: full-resolution val/test via sliding window, 512 window / 384
# stride = the grid the training tiles were cut on (tiling_split v0.2 §5).
model = dict(test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(384, 384)))
