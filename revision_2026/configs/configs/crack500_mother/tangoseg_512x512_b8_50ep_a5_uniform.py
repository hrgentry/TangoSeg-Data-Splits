# Arm A5 (uniform steering, w_k = 1/K) on parent-disjoint Crack500, 50-epoch protocol.
# 理由见同目录 a2_noprop 抬头; 与 deepcrack/ 下同名配置同式样 (方向头与损失保留, 只关掉
# 场对传播的转向作用)。
_base_ = './tangoseg_512x512_b8_50ep.py'

model = dict(decode_head=dict(propagation=dict(steering='uniform')))

work_dir = './work_dirs/tango_arms_c500/a5_uniform'
