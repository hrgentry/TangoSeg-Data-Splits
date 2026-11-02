# DeepCrack (官方 Fuse) 在 CrackMap 上, 改用**作者发表的训练策略**训练 500 轮。
#
# 目的: 矩阵那一格在统一协议下只有 ODS 17.4, 与该架构在另外三集的 71-73 相差悬殊,
# 需要排除"这是我们复现出错"的解释。移植保真度已单独验证 (灌入 qinnzou/DeepCrack
# 官方权重, 六路输出最大差 0.0, 74 个张量 30.91M 名称形状全同); 本配置进一步验证
# 换成作者自己的训练策略后该架构在同一份数据上能到什么水平。
#
# 与 qinnzou/DeepCrack 的 codes/config.py + train.py + trainer.py 对齐:
#   epoch 500                    -> train_cfg.max_epochs
#   use_adam=True, lr=1e-3       -> Adam(lr=1e-3)
#   weight_decay=0.0000          -> weight_decay=0
#   无学习率调度                  -> param_scheduler = []  (官方 lr_decay 在配置里
#                                   定义但训练代码从未使用, 全程恒定学习率)
#   train_batch_size 8, drop_last=True
#   pos_pixel_weight 1           -> 模型基座的 pos_weight=1.0
#   augCompose([[RandomColorJitter, 0.5], [RandomBlur, 0.2]]), 无翻转
#     -> PhotoMetricDistortion (brightness 32 / contrast 0.5 / saturation 0.5 /
#        hue 18 与官方的 32 / 0.5 / 0.5 / 0.1x179≈17.9 一一对应)
#        + DeepCrackRandomBlur(prob=0.2, kernel_size=5)
#        并移除数据基座里的 RandomFlip
#
# 有意保持不变的两项 (使结果与矩阵那一格可比):
#   - 512x512 输入路径与 BGR/255 预处理, 与矩阵完全相同;
#   - 测试索引与评测器完全相同。
#   本次只替换训练策略, 不替换数据呈现方式或评分方式。
#
# 被迫的一项偏离: 官方批量 8 在 512x512 下训练一步会以 CUDA
# "invalid configuration argument" 失败 (实测 batch 4 峰值 19.6 GiB,
# batch 8 外推约 39 GiB > 32 GiB 卡)。改用 micro-batch 2 x 累积 4, 等效批量与
# 优化步数均与官方一致。
_base_ = [
    '../../_base/runtime.py',
    '../../_base/datasets/crackmap.py',
    '../../_base/models/deepcrack_official_fuse.py',
]

max_epochs = 500

_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='BinarizeSeg', thr=0),
    dict(type='Resize', scale=(512, 512), keep_ratio=False),
    dict(type='SyncOriShape'),
    dict(type='PhotoMetricDistortion'),
    dict(type='DeepCrackRandomBlur', prob=0.2, kernel_size=5),
    dict(type='PackSegInputs'),
]

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    drop_last=True,
    dataset=dict(pipeline=_train_pipeline))
val_dataloader = dict(batch_size=2, num_workers=2)
test_dataloader = dict(batch_size=2, num_workers=2)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='Adam', lr=1e-3, weight_decay=0.0),
    accumulative_counts=4,
    clip_grad=None)

# 官方全程恒定学习率, 因此不挂任何调度器。
param_scheduler = []

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=25)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=True, interval=max_epochs,
                    max_keep_ckpts=1, save_best=None))

randomness = dict(seed=42)

work_dir = './work_dirs/_official_recipe/crackmap_deepcrack_500ep'
