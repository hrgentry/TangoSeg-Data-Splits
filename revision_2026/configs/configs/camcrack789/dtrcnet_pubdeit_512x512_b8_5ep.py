# DTrC-Net, public-DeiT variant (see _base/models/dtrcnet_pubdeit.py). Same
# data, schedule and batch as dtrcnet_512x512_b8_5ep.py; only the model base
# and the work_dir differ, so the old run stays intact for provenance.
_base_ = [
    '../../_base/runtime.py',
    '../../_base/schedule_b8_5ep.py',
    '../../_base/datasets/camcrack789.py',
    '../../_base/models/dtrcnet_pubdeit.py',
]

train_dataloader = dict(batch_size=8, num_workers=4)
val_dataloader = dict(batch_size=8, num_workers=4)
test_dataloader = dict(batch_size=8, num_workers=4)

work_dir = './work_dirs/mixercseg_baselines/camcrack789/dtrcnet_pubdeit_512x512_b8_5ep'
