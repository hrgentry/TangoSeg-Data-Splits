# Crack-ShuttleNetV2: adaptive memory + boundary aux.
# CE:Dice set to 1:5 to match MixerCSeg fair protocol.
model = dict(
    type='CrackShuttleNetV2Segmentor',
    in_channels=3,
    num_classes=2,
    num_levels=5,
    num_rounds=4,
    min_channels=16,
    class_weight=[0.5, 5.0],
    loss_ce_weight=1.0,
    loss_dice_weight=5.0,
    loss_boundary_weight=0.5,
    boundary_pos_weight=4.0,
    boundary_kernel_size=3,
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255,
        size=(512, 512),
        test_cfg=dict(size_divisor=32)),
    test_cfg=dict(mode='whole'))

model_name = 'shuttlenet_v2'
