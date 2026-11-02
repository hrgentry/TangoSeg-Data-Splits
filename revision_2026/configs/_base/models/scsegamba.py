# SCSegamba import is declared in each leaf config (overrides runtime
# custom_imports). Defining it here would collide with runtime.py's
# custom_imports among _base_ entries (mmengine KeyError).

model = dict(
    type='SCSegambaSegmentor',
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        mean=[127.5, 127.5, 127.5],
        std=[127.5, 127.5, 127.5],
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255,
        size=(512, 512)),
    bce_loss_ratio=1.0 / 6.0,
    dice_loss_ratio=5.0 / 6.0,
    threshold=0.5,
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

model_name = 'scsegamba'
