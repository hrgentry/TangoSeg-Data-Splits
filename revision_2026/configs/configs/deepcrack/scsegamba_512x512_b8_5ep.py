_base_ = [
    '../../_base/runtime.py',
    '../../_base/schedule_b8_5ep.py',
    '../../_base/datasets/deepcrack.py',
    '../../_base/models/scsegamba.py',
]

# Leaf override — must include runtime imports + SCSegamba (cannot live in
# model base alongside runtime.py custom_imports).
# A leaf `custom_imports` REPLACES the base's, it does not merge with it, so listing only
# SCSegamba here dropped projects.deepcrack and the run died at build time with
# "DeepCrackPairDataset is not in the mmseg::dataset registry" (seen 2026-08-11 on .80).
# Mirrors the list in mambavision/restormixer, which had it right.
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

work_dir = './work_dirs/mixercseg_baselines/deepcrack/scsegamba_512x512_b8_5ep'
