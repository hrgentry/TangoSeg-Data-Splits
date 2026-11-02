# CrackMap 源照片互斥划分上的 50 epoch 协议(从零臂)。
# 与 ./tangoseg_512x512_b8_5ep.py 只差预算:该叶已把 ann_file 指到 splits_parent/,
# 模型/优化器/增广一路继承 ../crackmap/,此处不重复。
_base_ = ['./tangoseg_512x512_b8_5ep.py']

max_epochs = 50

# PolyLR 必须铺满 50 epoch;继承 5ep 的调度会让 lr 在第 5 轮落到 eta_min,
# 剩余 45 轮以 ~0 学习率空转。
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=1e-6,
        power=1.0,
        begin=0,
        end=max_epochs,
        by_epoch=True),
]

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=5)

work_dir = './work_dirs/mixercseg_baselines/crackmap_parent/tangoseg_512x512_b8_50ep'
