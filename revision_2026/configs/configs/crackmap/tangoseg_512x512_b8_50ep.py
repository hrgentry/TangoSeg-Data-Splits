# TANGO at the 50-epoch budget on CrackMap (smallest of the four sets).
# Companion to the DeepCrack 50-epoch arm; see that file for the rationale.
_base_ = ['./tangoseg_512x512_b8_5ep.py']

max_epochs = 50

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

work_dir = './work_dirs/mixercseg_long/crackmap/tangoseg_512x512_b8_50ep'
