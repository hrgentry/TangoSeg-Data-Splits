# A1 参照臂（从零、完整 TANGO）在 Crack500 母图互斥划分上的 50 epoch 协议。
#
# 为什么加这一组：DeepCrack 上 A7-A1 只有 +0.47 ODS（crack_v5 上是 +7.74），
# 但 DeepCrack 仅 300 张训练图、两臂末端 fgIoU 都到 ~73，怀疑是"太小太容易、
# 从零臂也已收敛"的产物。Crack500 母图互斥划分有 2357 个训练裁块、难度居中，
# 用来判定预训练效应到底是 crack_v5 特有还是普遍消失。
#
# 为什么用母图互斥而非发行划分：本文自己论证发行划分的 test 母图全部出现在
# train，虽实测只差 0.08 分，但正文明确写了"仍应为正确性重建"，匹配对照理应
# 跑在干净划分上。两臂同在此划分内配对，划分选择不影响对照本身。
#
# 除 max_epochs 与调度跨度外，逐项继承 5ep 配置（bf16/lr5e-4/clip1.0/b8），
# 与 DeepCrack 五臂完全同式样。
_base_ = ['./tangoseg_512x512_b8_5ep.py']

max_epochs = 50

# PolyLR 必须铺满 50 epoch；继承 5ep 的调度会让 lr 在第 5 轮就落到 eta_min，
# 剩余 45 轮以 ~0 学习率空转（该坑已在 DeepCrack 50ep 配置注释中记录）。
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

work_dir = './work_dirs/tango_arms_c500/a1_base'
