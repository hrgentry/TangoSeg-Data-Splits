_base_ = [
    '../../_base/runtime.py',
    '../../_base/schedule_b8_5ep.py',
    '../../_base/datasets/crackmap.py',
    '../../_base/models/deepcrack_fast_fuse.py',
]

train_dataloader = dict(batch_size=8, num_workers=4)
val_dataloader = dict(batch_size=8, num_workers=4)
test_dataloader = dict(batch_size=8, num_workers=4)

work_dir = './work_dirs/mixercseg_baselines/crackmap/deepcrack_fast_fuse_512x512_b8_5ep'
