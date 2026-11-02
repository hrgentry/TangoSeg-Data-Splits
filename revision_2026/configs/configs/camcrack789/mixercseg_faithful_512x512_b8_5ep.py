_base_ = [
    '../../_base/runtime.py',
    '../../_base/schedule_b8_5ep.py',
    '../../_base/datasets/camcrack789.py',
    '../../_base/models/mixercseg_faithful.py',
]

train_dataloader = dict(batch_size=8, num_workers=4, drop_last=True)
val_dataloader = dict(batch_size=8, num_workers=4)
test_dataloader = dict(batch_size=8, num_workers=4)

optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    dtype='bfloat16',
    optimizer=dict(type='AdamW', lr=5e-4, weight_decay=0.01))

work_dir = './work_dirs/mixercseg_baselines/camcrack789/mixercseg_faithful_512x512_b8_5ep'
