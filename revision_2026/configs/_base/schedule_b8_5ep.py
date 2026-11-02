# MixerCSeg public-set fair track: 5 epochs, batch size 8, AdamW lr=5e-4.
# Dataset bases consume train_batch_size / val_batch_size / test_batch_size.
max_epochs = 5
train_batch_size = 8
val_batch_size = 8
test_batch_size = 8
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
