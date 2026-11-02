# TANGO 50-epoch arm in fp32 on CamCrack789, for torch 2.0 hosts.
# See configs/deepcrack/tangoseg_512x512_b8_50ep_fp32.py for why AMP is dropped.
_base_ = ['./tangoseg_512x512_b8_5ep.py']

max_epochs = 50

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=5e-4, weight_decay=0.01),
    clip_grad=dict(max_norm=1.0),
    paramwise_cfg=dict(norm_decay_mult=0.0, bias_decay_mult=0.0))

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

work_dir = './work_dirs/mixercseg_long/camcrack789/tangoseg_512x512_b8_50ep_fp32'
