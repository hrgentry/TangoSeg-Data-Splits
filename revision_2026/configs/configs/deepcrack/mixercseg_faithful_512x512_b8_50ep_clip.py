# MixerCSeg-faithful + gradient clipping, 50 epochs.
# Combines the two suspected handicaps of the faithful arm (missing clip_grad
# and a 5-epoch budget) so the comparison against TANGO rests on a converged,
# non-collapsed MixerCSeg rather than a mis-configured one.
_base_ = ['./mixercseg_faithful_512x512_b8_5ep.py']

max_epochs = 50

optim_wrapper = dict(clip_grad=dict(max_norm=1.0))

param_scheduler = [
    # end=2 (not 1): a 1-epoch LinearLR window has a single interpolation
    # step, so warmup never reaches end_factor and leaves lr at
    # start_factor*base_lr=5e-7; the incremental PolyLR then inherits that
    # as its starting point and ramps *up* to eta_min. Verified against the
    # scheduler chain directly.
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
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=5)

work_dir = './work_dirs/mixercseg_local_oflex/deepcrack/mixercseg_faithful_512x512_b8_50ep_clip'
