# TangoSeg + ADE20K 预训练 MSCAN-S trunk,CrackMap 源照片互斥划分。
# 只改 trunk 初始化,其余一路继承 ./tangoseg_512x512_b8_50ep.py。
#
# 为什么是 ADE20K 分割权重而非既有 A7 臂的 IN1k 分类权重:A7 装的是
# mscan_s ImageNet-1k **分类** checkpoint;本臂装 SegNeXt 的 ADE20K **分割**
# checkpoint(mIoU 44.16),423 个 MSCAN 张量按 shape 全部对上。两臂回答的是
# 不同问题 —— A7 问"分类预训练值多少",本臂问"同任务分割预训练值多少"。
# prefix='backbone.' 用来从整段 segmentor checkpoint 里挑出 trunk 那部分并剥掉前缀。
_base_ = ['./tangoseg_512x512_b8_50ep.py']

checkpoint_file = 'pretrain/segnext_mscan-s_ade20k.pth'

model = dict(
    backbone=dict(
        trunk=dict(
            init_cfg=dict(
                type='Pretrained',
                checkpoint=checkpoint_file,
                prefix='backbone.'))))

work_dir = './work_dirs/mixercseg_baselines/crackmap_parent/tangoseg_ade20k_512x512_b8_50ep'
