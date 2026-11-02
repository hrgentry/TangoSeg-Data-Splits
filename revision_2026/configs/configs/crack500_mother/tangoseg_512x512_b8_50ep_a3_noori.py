# Arm A3 (no orientation supervision, loss weight 0) on parent-disjoint Crack500,
# 50-epoch protocol. 理由见同目录 a2_noprop 抬头; 与 deepcrack/ 下同名配置同式样。
_base_ = './tangoseg_512x512_b8_50ep.py'

model = dict(
    decode_head=dict(loss_ori=dict(type='OrientationLoss', loss_weight=0.0)))

work_dir = './work_dirs/tango_arms_c500/a3_noori'
