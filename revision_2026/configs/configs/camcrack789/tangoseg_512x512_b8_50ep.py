# TANGO at the 50-epoch budget on CamCrack789, bf16 AMP.
# Companion to the DeepCrack 50-epoch arm; see that file for the rationale, and
# configs/crack500/tangoseg_512x512_b8_50ep.py for why this sits next to the
# _fp32 leaf.
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

work_dir = './work_dirs/mixercseg_long/camcrack789/tangoseg_512x512_b8_50ep'
