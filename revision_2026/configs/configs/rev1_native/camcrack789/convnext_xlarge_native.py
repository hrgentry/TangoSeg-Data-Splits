# CACAIE 一审 B7 (R2-6 后半): 原始分辨率训练与推理。
# 与 ../../camcrack789/convnext_xlarge_512x512_b8_5ep.py 的唯一差别是不做 Resize: 640x480 的图原先被各向异性地拉成 512x512 (纵横比 4:3 -> 1:1)
# 数据预处理把 640x480 补边到 640x640 (底部 160 行, 标签 255 即 ignore, 不计损失也不计指标):
# SCSegamba 由 token 数开平方推边长、MixerCSeg faithful 在 H!=W 时 .view 失败, 两者都假定
# 方形输入 (2026-10-30 冒烟发现)。补边不缩放、不变形, 仍是保持纵横比的原始分辨率处理;
# 为六个模型可比, 统一用补边而不是只给这两个模型改上游代码。
# 轮数、调度、种子等 50 轮协议的覆盖由 scripts/rev1_local_chain.ps1 的 b7 阶段统一给出。
_base_ = ['../../camcrack789/convnext_xlarge_512x512_b8_5ep.py']

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
# SegDataPreProcessor 只在训练时按 size 补边; 验证与测试要另给 test_cfg, 否则仍是 480x640
# (方形输入假设的两个模型在验证时失败, 2026-10-31 冒烟发现)。推理后 mmseg 按记录的补边裁回原尺寸。
data_preprocessor = dict(size=(640, 640), size_divisor=None, test_cfg=dict(size=(640, 640)))
model = dict(data_preprocessor=dict(size=(640, 640), size_divisor=None, test_cfg=dict(size=(640, 640))))
