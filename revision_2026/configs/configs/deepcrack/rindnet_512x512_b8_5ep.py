_base_ = [
    '../../_base/runtime.py',
    '../../_base/schedule_b8_5ep.py',
    '../../_base/datasets/deepcrack.py',
    '../../_base/models/rindnet.py',
]

train_dataloader = dict(batch_size=8, num_workers=4)
val_dataloader = dict(batch_size=8, num_workers=4)
test_dataloader = dict(batch_size=8, num_workers=4)

work_dir = './work_dirs/mixercseg_baselines/deepcrack/rindnet_512x512_b8_5ep'
