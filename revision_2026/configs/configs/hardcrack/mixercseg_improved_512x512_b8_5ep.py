# HardCrack leaf derived from configs/crack500/mixercseg_improved_512x512_b8_5ep.py (2026-09-05).
# Deltas vs the public-set leaf: dataset base -> hardcrack.py; val/test batch_size -> 1
# (full-resolution images of three sizes cannot be stacked); model.test_cfg -> sliding
# window (HardCrack protocol; set in the leaf because mmengine forbids a second base
# overriding `model`). Everything else (schedule, per-model batch/optimizer) is the
# same-track b8_5ep protocol unchanged.
_base_ = [
    '../../_base/runtime.py',
    '../../_base/schedule_b8_5ep.py',
    '../../_base/datasets/hardcrack.py',
    '../../_base/models/mixercseg_improved.py',
]

train_dataloader = dict(batch_size=8, num_workers=4)
val_dataloader = dict(batch_size=1, num_workers=4)
test_dataloader = dict(batch_size=1, num_workers=4)

optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    dtype='bfloat16',
    optimizer=dict(type='AdamW', lr=5e-4, weight_decay=0.01),
    clip_grad=dict(max_norm=1.0),
    paramwise_cfg=dict(
        custom_keys={
            'A_logs': dict(decay_mult=0.0),
            'Ds': dict(decay_mult=0.0),
            'dt_projs_bias': dict(decay_mult=0.0),
            'norm': dict(decay_mult=0.0),
        }))
param_scheduler = [
    # end=2 (not 1): a 1-epoch LinearLR window has a single interpolation
    # step, so warmup never reaches end_factor and leaves lr at
    # start_factor*base_lr=5e-7; the incremental PolyLR then inherits that
    # as its starting point and ramps *up* to eta_min. Verified against the
    # scheduler chain directly.
    dict(type='LinearLR', start_factor=1e-3, begin=0, end=2, by_epoch=True),
    dict(type='PolyLR', eta_min=1e-6, power=1.0, begin=2, end=5, by_epoch=True),
]

work_dir = './work_dirs/mixercseg_baselines/hardcrack/mixercseg_improved_512x512_b8_5ep'

# HardCrack protocol: full-resolution val/test via sliding window, 512 window / 384
# stride = the grid the training tiles were cut on (tiling_split v0.2 §5).
model = dict(test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(384, 384)))
