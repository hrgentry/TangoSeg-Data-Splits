# MixerCSeg paper Sec. 4.2: 50 epochs, batch size 1, AdamW lr=5e-4.
max_epochs = 50
train_batch_size = 1
val_batch_size = 1
test_batch_size = 1
num_workers = 4

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=5e-4, weight_decay=0.01),
    clip_grad=None)

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
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=1)
