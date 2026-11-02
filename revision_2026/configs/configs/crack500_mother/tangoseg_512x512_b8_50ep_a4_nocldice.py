# Arm A4 (no centerline-Dice loss) on parent-disjoint Crack500, 50-epoch protocol.
# 理由见同目录 a2_noprop 抬头; 与 deepcrack/ 下同名配置同式样 (loss_decode 去掉
# SoftClDiceLoss, 其余两项权重不变)。
_base_ = './tangoseg_512x512_b8_50ep.py'

model = dict(
    decode_head=dict(loss_decode=[
        dict(type='CrossEntropyLoss', use_sigmoid=True, class_weight=[5.0],
             loss_weight=1.0),
        dict(type='TangoBinaryDiceLoss', smooth=1.0, loss_weight=1.0),
    ]))

work_dir = './work_dirs/tango_arms_c500/a4_nocldice'
