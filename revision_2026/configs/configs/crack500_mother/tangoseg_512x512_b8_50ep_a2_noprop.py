# Arm A2 (no propagation, T=0) on parent-disjoint Crack500, 50-epoch protocol.
#
# 为什么补这一组 (2026-09-11 对抗性审稿 T0-4): 四个裂缝先验开关此前只在 DeepCrack 上
# 消融, 而 DeepCrack 是四集里 50 轮模型间极差最小 (4.22)、初始化效应也消失 (+0.12) 的
# 那个集; 稿件却把"先验 < 初始化"写进 priority order, 靠的是 DeepCrack 的先验数字和
# Crack500 的初始化数字并排 —— 跨数据集拼接。让三个发展因素落在同一个数据集上, 只能
# 在 Crack500 母图互斥划分上把 A2-A5 补齐 (A1 与 A7 已有, work_dirs/tango_arms_c500)。
#
# 与 deepcrack/ 下同名配置完全同式样: 只改 propagation.iterations=0, 方向头/损失保留,
# 参数张量逐个保留 (T=0 时容量配平, 见 propagation.py)。参照臂为同划分的 a1_base 三种子。
_base_ = './tangoseg_512x512_b8_50ep.py'

model = dict(decode_head=dict(propagation=dict(iterations=0)))

work_dir = './work_dirs/tango_arms_c500/a2_noprop'
