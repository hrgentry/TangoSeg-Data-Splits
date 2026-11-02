_base_ = [
    '../../_base/runtime.py',
    '../../_base/schedule_b8_5ep.py',
    '../../_base/datasets/deepcrack.py',
    '../../_base/models/convnext_xlarge.py',
]

train_dataloader = dict(batch_size=2, num_workers=2)
val_dataloader = dict(batch_size=2, num_workers=2)
test_dataloader = dict(batch_size=2, num_workers=2)

work_dir = './work_dirs/mixercseg_baselines/deepcrack/convnext_xlarge_512x512_b8_5ep'

# Keep effective batch=8 while fitting a 24GB card: ConvNeXt-XL peaked at
# 24076 MiB at batch 8 on .80 and the driver spilled to system RAM. Patched on
# both hosts so the protocol is identical wherever the model runs.
optim_wrapper = dict(accumulative_counts=4)
