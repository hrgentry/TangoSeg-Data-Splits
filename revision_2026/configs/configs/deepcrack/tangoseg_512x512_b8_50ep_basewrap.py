# TANGO 50 轮臂, 优化器包装换成 12 个基线共用的那一套。理由见同目录
# tangoseg_512x512_b8_5ep_basewrap.py 抬头。继承 tangoseg_512x512_b8_50ep.py
# (PolyLR 0->50、val_interval=5), 只覆盖 optim_wrapper, 与既有 TANGO 预算臂
# (work_dirs/tango_budget/tango_50ep_seed4x) 逐键只差包装这一项。
_base_ = ['./tangoseg_512x512_b8_50ep.py']

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=5e-4, weight_decay=0.01),
    clip_grad=None)

work_dir = './work_dirs/mixercseg_baselines/deepcrack/tangoseg_512x512_b8_50ep_basewrap'
