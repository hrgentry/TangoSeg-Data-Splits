# TANGO at the 50-epoch budget on DeepCrack.
# The 5-epoch fair track under-trains the slow-converging Mamba baselines, so a
# claim of "TANGO beats MixerCSeg" needs both sides at a budget where each has
# converged. TANGO is ~100x cheaper per iteration than MixerCSeg here, so its
# 50-epoch arm is affordable; everything except the epoch budget is unchanged.
_base_ = ['./tangoseg_512x512_b8_5ep.py']

max_epochs = 50

# PolyLR must span all 50 epochs; inheriting the 5-epoch schedule would drive lr
# to eta_min by epoch 5 and leave the remaining 45 epochs training at ~0.
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=1e-6,
        power=1.0,
        begin=0,
        end=max_epochs,
        by_epoch=True),
]

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=5)

work_dir = './work_dirs/mixercseg_long/deepcrack/tangoseg_512x512_b8_50ep'
