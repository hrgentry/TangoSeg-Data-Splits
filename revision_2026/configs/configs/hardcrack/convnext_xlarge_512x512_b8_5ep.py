# HardCrack leaf derived from configs/crack500/convnext_xlarge_512x512_b8_5ep.py (2026-09-05).
# Deltas vs the public-set leaf: dataset base -> hardcrack.py; val/test batch_size -> 1
# (full-resolution images of three sizes cannot be stacked); model.test_cfg -> sliding
# window (HardCrack protocol; set in the leaf because mmengine forbids a second base
# overriding `model`). Everything else (schedule, per-model batch/optimizer) is the
# same-track b8_5ep protocol unchanged.
_base_ = [
    '../../_base/runtime.py',
    '../../_base/schedule_b8_5ep.py',
    '../../_base/datasets/hardcrack.py',
    '../../_base/models/convnext_xlarge.py',
]

train_dataloader = dict(batch_size=8, num_workers=4)
val_dataloader = dict(batch_size=1, num_workers=4)
test_dataloader = dict(batch_size=1, num_workers=4)

work_dir = './work_dirs/mixercseg_baselines/hardcrack/convnext_xlarge_512x512_b8_5ep'

# HardCrack protocol: full-resolution val/test via sliding window, 512 window / 384
# stride = the grid the training tiles were cut on (tiling_split v0.2 §5).
model = dict(test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(384, 384)))
