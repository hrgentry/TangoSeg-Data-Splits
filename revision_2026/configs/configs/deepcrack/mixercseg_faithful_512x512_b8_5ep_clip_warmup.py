# MixerCSeg-faithful, 5 epochs, with the SAME scheduler shape as the 50-epoch arm.
#
# 为什么需要这个配置：预算对照必须只差预算。原有的
# `mixercseg_faithful_512x512_b8_5ep_clip.py` 用的是 PolyLR-only（无 warmup），
# 而 `..._50ep_clip.py` 用 LinearLR(0→2)+PolyLR(2→50)。拿那两个相减，差的是
# 「预算 + 有没有 warmup」两件事，不能归因给预算。
#
# 本配置继承 50ep 臂（拿到它的 warmup），只把预算改回 5 epoch，于是
# 5ep 与 50ep 之间**只剩 max_epochs 一个变量**。
#
# 注意 LinearLR 的 end=2 而非 1：by_epoch 下 1-epoch 窗口只有一个插值步，
# warmup 永远到不了 end_factor，lr 卡在 start_factor*base_lr=5e-7，随后增量式
# PolyLR 以该值为起点向 eta_min 爬升——2026-07-27 之前 improved 的全部低分
# （5ep 4.16、50ep 4.33/46.82）都源于此，修好后同配方是 62.58/68.85。
_base_ = ['./mixercseg_faithful_512x512_b8_50ep_clip.py']

max_epochs = 5

param_scheduler = [
    dict(type='LinearLR', start_factor=1e-3, begin=0, end=2, by_epoch=True),
    dict(
        type='PolyLR',
        eta_min=1e-6,
        power=1.0,
        begin=2,
        end=max_epochs,
        by_epoch=True),
]

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)

work_dir = './work_dirs/mixercseg_budget/faithful_5ep'
