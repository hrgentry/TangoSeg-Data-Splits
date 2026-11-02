norm_cfg = dict(type='BN', requires_grad=True)

# Backbone-only ImageNet-21K init; decode/aux heads train from scratch.
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
        type='mmpretrain.ConvNeXt',
        arch='xlarge',
        out_indices=[0, 1, 2, 3],
        drop_path_rate=0.4,
        layer_scale_init_value=1.0,
        gap_before_final_norm=False,
        init_cfg=dict(
            type='Pretrained',
            checkpoint=(
                'pretrain/convnext-xlarge_3rdparty_in21k_'
                '20220301-08aa5ddc.pth'),
            prefix='backbone.')),
    decode_head=dict(
        type='UPerHead',
        in_channels=[256, 512, 1024, 2048],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[
            dict(
                type='IgnoreSafeCELoss',
                loss_weight=1.0,
                ignore_index=255),
            dict(
                type='DiceLoss',
                use_sigmoid=False,
                loss_weight=5.0,
                ignore_index=255),
        ]),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=1024,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[
            dict(
                type='IgnoreSafeCELoss',
                loss_weight=0.4,
                ignore_index=255),
            dict(
                type='DiceLoss',
                use_sigmoid=False,
                loss_weight=2.0,
                ignore_index=255),
        ]),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

model_name = 'convnext_xlarge'
