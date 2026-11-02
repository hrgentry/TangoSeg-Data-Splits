# A7 臂（ImageNet-1k 预训练 MSCAN-S 主干）在 Crack500 母图互斥划分上的
# 50 epoch 协议。与同目录 tangoseg_512x512_b8_50ep.py 的**唯一差别**是主干
# init_cfg，权重与 DeepCrack 侧 A7 用的是同一个 checkpoint。
_base_ = ['./tangoseg_512x512_b8_50ep.py']

checkpoint_file = ('https://download.openmmlab.com/mmsegmentation/v0.5/'
                   'pretrain/segnext/mscan_s_20230227-f33ccdf2.pth')

model = dict(
    backbone=dict(
        trunk=dict(
            init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file))))

work_dir = './work_dirs/tango_arms_c500/a7_in1k'
