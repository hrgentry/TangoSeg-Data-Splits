_base_ = [
    '../../_base/runtime.py',
    '../../_base/schedule_b8_5ep.py',
    '../../_base/datasets/deepcrack.py',
    '../../_base/models/mambavision.py',
]

# WSL: adapters + dataset registration (DeepCrackPairDataset lives in projects.deepcrack).
custom_imports = dict(
    imports=[
        'projects.deepcrack',
        'projects.dinov3_sealing.crack_modules',
        'projects.mixercseg_baselines.adapters',
        'projects.mixercseg_baselines.transforms',
    ],
    allow_failed_imports=False)

train_dataloader = dict(batch_size=2, num_workers=2)
val_dataloader = dict(batch_size=2, num_workers=2)
test_dataloader = dict(batch_size=2, num_workers=2)
# Keep effective batch=8 under 24GB VRAM (MambaVision is heavy).
optim_wrapper = dict(accumulative_counts=4)

work_dir = './work_dirs/mixercseg_baselines/deepcrack/mambavision_512x512_b8_5ep'
