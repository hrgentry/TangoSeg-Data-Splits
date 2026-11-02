# Arm A7 (ImageNet-1k pretrained MSCAN-S trunk) on the public DeepCrack
# 50-epoch protocol. Replaces the crack_v5 A7 arm in the TRB paper: the paper
# uses public datasets only, so the pretraining contrast must be measured on
# the same DeepCrack b8/50ep/3-seed protocol as the budget arm (whose runs are
# the from-scratch reference). Single change vs tangoseg_512x512_b8_50ep:
# trunk init_cfg. Same checkpoint as projects/tangoseg A7 (downloads once).
_base_ = './tangoseg_512x512_b8_50ep.py'

checkpoint_file = ('https://download.openmmlab.com/mmsegmentation/v0.5/'
                   'pretrain/segnext/mscan_s_20230227-f33ccdf2.pth')

model = dict(
    backbone=dict(
        trunk=dict(
            init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file))))

work_dir = './work_dirs/tango_arms/a7_in1k'
