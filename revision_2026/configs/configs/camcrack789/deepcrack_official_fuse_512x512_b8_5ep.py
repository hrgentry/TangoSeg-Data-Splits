_base_ = [
    '../../_base/runtime.py',
    '../../_base/schedule_b8_5ep.py',
    '../../_base/datasets/camcrack789.py',
    '../../_base/models/deepcrack_official_fuse.py',
]

# 官方 Fuse 在批量 8 下训练会 CUDA 失败（见模型基座注释）。micro-batch 2 x
# 累积 4 保持等效批量 8 与相同的优化步数，与另外三个重模型同一处置。
train_dataloader = dict(batch_size=2, num_workers=2)
val_dataloader = dict(batch_size=2, num_workers=2)
test_dataloader = dict(batch_size=2, num_workers=2)

optim_wrapper = dict(accumulative_counts=4)

work_dir = './work_dirs/mixercseg_baselines/camcrack789/deepcrack_official_fuse_512x512_b8_5ep'
