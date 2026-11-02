# TangoSeg (TangoTrunk + TangoHead) for MixerCSeg public-set fair track.
# Model definition mirrors projects/tangoseg/tango_crack_v5_512.py; schedule /
# datasets come from mixercseg_baselines b8/5ep bases.

norm_cfg = dict(type='BN', requires_grad=True)

model = dict(
    type='EncoderDecoder',
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255,
        size=(512, 512)),
    pretrained=None,
    backbone=dict(
        type='TangoTrunk',
        detail_channels=32,
        trunk=dict(
            type='MSCAN',
            embed_dims=[64, 128, 320, 512],
            depths=[2, 2, 4, 2],
            mlp_ratios=[8, 8, 4, 4],
            drop_path_rate=0.1,
            norm_cfg=norm_cfg)),
    decode_head=dict(
        type='TangoHead',
        in_channels=(32, 64, 128, 320, 512),
        in_index=(0, 1, 2, 3, 4),
        channels=128,
        num_classes=2,
        out_channels=1,
        threshold=0.5,
        propagation=dict(
            channels=64, num_orientations=4, radius=5, iterations=3),
        refine_channels=64,
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                use_sigmoid=True,
                class_weight=[5.0],
                loss_weight=1.0),
            dict(type='TangoBinaryDiceLoss', smooth=1.0, loss_weight=1.0),
            dict(type='SoftClDiceLoss', iterations=5, loss_weight=0.5),
        ],
        loss_coarse=[
            dict(
                type='CrossEntropyLoss',
                use_sigmoid=True,
                class_weight=[5.0],
                loss_weight=0.4),
            dict(type='TangoBinaryDiceLoss', smooth=1.0, loss_weight=0.4),
        ],
        loss_ori=dict(type='OrientationLoss', loss_weight=0.5)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

model_name = 'tangoseg'
