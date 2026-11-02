# TANGO 5 轮臂, 但优化器包装换成 12 个基线共用的那一套 (对照臂, 2026-09-11)。
#
# 为什么要这一臂: 对抗性审稿指出, 矩阵里只有 TANGO / MixerCSeg modified 两个配置
# 带 bf16 AMP + 梯度裁剪 max_norm=1.0 + norm/bias 免衰减 (paramwise_cfg), 其余
# 12 个基线是 fp32 OptimWrapper、clip_grad=None、无 paramwise —— 见
# _base/schedule_b8_5ep.py。稿件 §3.2 却说 optimizer 已标准化。梯度裁剪恰是防止
# 短预算下塌陷的标准手段, 而 §4.5 的稳定性叙事 (TANGO 三种子极差 2.18 vs
# faithful 29.04) 因此无法与"只有 TANGO 开了裁剪"区分。
#
# 本臂**只**换 optim_wrapper 为 schedule 基座里的原样 (fp32、无裁剪、无 paramwise),
# 数据 / 模型 / 调度 / 批量 / 轮数 / 种子全部继承 tangoseg_512x512_b8_5ep.py。
# 实测 TANGO 矩阵运行的裁剪前 grad_norm 在前 10% 训练里中位 2.4-3.1、最大 4.2-5.2,
# 即裁剪把早期步长压了 2-5 倍; 这一臂就是量这件事对最终分数值多少。
#
# 为什么是 fp32 而不是只关裁剪: 审稿人要的是"与其它 12 行同一个包装", 少一个开关
# 不够。父配置注释说 fp16 下递归传播不稳, 但那是 fp16, fp32 不在此列。
_base_ = ['./tangoseg_512x512_b8_5ep.py']

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=5e-4, weight_decay=0.01),
    clip_grad=None)

work_dir = './work_dirs/mixercseg_baselines/deepcrack/tangoseg_512x512_b8_5ep_basewrap'
