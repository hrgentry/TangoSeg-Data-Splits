# CACAIE 一审 B7 (R2-6 后半): 原始分辨率训练与推理。
# 与 ../../crackmap_parent/scsegamba_512x512_b8_5ep.py 的唯一差别是不做 Resize: 256x256 的图原先被放大到 512x512
# 数据预处理不再补到固定的 512x512, 只补到 32 的倍数 (256x256 本身就是 32 的倍数, 不产生补边)。
# 轮数、调度、种子等 50 轮协议的覆盖由 scripts/rev1_local_chain.ps1 的 b7 阶段统一给出。
_base_ = ['../../crackmap_parent/scsegamba_512x512_b8_5ep.py']

_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='BinarizeSeg', thr=0),
    dict(type='SyncOriShape'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackSegInputs'),
]
_test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='BinarizeSeg', thr=0),
    dict(type='SyncOriShape'),
    dict(type='PackSegInputs'),
]
train_dataloader = dict(dataset=dict(pipeline=_train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=_test_pipeline))
test_dataloader = dict(dataset=dict(pipeline=_test_pipeline))
data_preprocessor = dict(size=None, size_divisor=32)
model = dict(data_preprocessor=dict(size=None, size_divisor=32))
