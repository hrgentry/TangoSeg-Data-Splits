# RINDNet edge variant (ResNet-50 backbone) — ImageNet mean/std.
model = dict(
    type='RINDNetSegmentor',
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255,
        size=(512, 512)),
    resnet_pretrained=True,
    resnet_pretrained_path=None,
    bce_loss_weight=1.0,
    dice_loss_weight=5.0,
    threshold=0.5,
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

model_name = 'rindnet'
