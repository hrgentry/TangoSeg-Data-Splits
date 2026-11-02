# TangoSeg with an ADE20K-pretrained MSCAN-S trunk at the 50-epoch budget on DeepCrack.
# Only the trunk initialization differs from ./tangoseg_512x512_b8_50ep.py.
#
# Full rationale (why the SegNeXt ADE20K segmentation checkpoint and not the A7
# ImageNet-1k classification one, which 423 tensors transfer, why prefix='backbone.')
# lives in configs/hardcrack/tangoseg_ade20k_512x512_b8_5ep.py -- kept in one place
# so the four public-set leaves cannot drift apart from it.
_base_ = ['./tangoseg_512x512_b8_50ep.py']

checkpoint_file = 'pretrain/segnext_mscan-s_ade20k.pth'

model = dict(
    backbone=dict(
        trunk=dict(
            init_cfg=dict(
                type='Pretrained',
                checkpoint=checkpoint_file,
                prefix='backbone.'))))

work_dir = './work_dirs/mixercseg_long/deepcrack/tangoseg_ade20k_512x512_b8_50ep'
