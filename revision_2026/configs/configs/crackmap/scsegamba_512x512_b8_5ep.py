_base_ = [
    '../../_base/runtime.py',
    '../../_base/schedule_b8_5ep.py',
    '../../_base/datasets/crackmap.py',
    '../../_base/models/scsegamba.py',
]

# A leaf `custom_imports` REPLACES the base's rather than merging with it, so listing only
# SCSegamba dropped projects.deepcrack -- which registers DeepCrackPairDataset, the dataset
# class every public-set base uses -- and the run died at build time. Same fix as the
# deepcrack config; mambavision/restormixer already had the full list.
custom_imports = dict(
    imports=[
        'projects.deepcrack',
        'projects.dinov3_sealing.crack_modules',
        'projects.mixercseg_baselines.adapters',
        'projects.mixercseg_baselines.transforms',
        'projects.SCSegamba.scsegamba_mmseg',
    ],
    allow_failed_imports=False)

train_dataloader = dict(batch_size=8, num_workers=4)
val_dataloader = dict(batch_size=8, num_workers=4)
test_dataloader = dict(batch_size=8, num_workers=4)

work_dir = './work_dirs/mixercseg_baselines/crackmap/scsegamba_512x512_b8_5ep'
