# DeepCrack fast-fuse variant (architecture-native multi-side BCE).
model = dict(
    type='DeepCrackFastFuseSegmentor',
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        mean=[0.0, 0.0, 0.0],
        std=[255.0, 255.0, 255.0],
        bgr_to_rgb=False,
        pad_val=0,
        seg_pad_val=255,
        size=(512, 512),
        test_cfg=dict(size_divisor=32)),
    pos_weight=1.0,
    loss_weight=1.0,
    divide_loss_by_batch=True,
    threshold=0.5,
    test_cfg=dict(mode='whole'))

model_name = 'deepcrack_fast_fuse'
