_base_ = [
    '../../_base/runtime.py',
    '../../_base/schedule_b8_5ep.py',
    '../../_base/datasets/deepcrack.py',
    '../../_base/models/mixercseg_improved.py',
]

train_dataloader = dict(batch_size=8, num_workers=4)
val_dataloader = dict(batch_size=8, num_workers=4)
test_dataloader = dict(batch_size=8, num_workers=4)

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

work_dir = './work_dirs/mixercseg_baselines/deepcrack/mixercseg_improved_512x512_b8_5ep'
