model = dict(
    type='DTrCNetSegmentor',
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        mean=[127.5, 127.5, 127.5],
        std=[127.5, 127.5, 127.5],
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255,
        size=(512, 512)),
    resnet_pretrained=True,
    deit_pretrained=True,
    bce_loss_weight=1.0,
    dice_loss_weight=5.0,
    threshold=0.5,
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

model_name = 'dtrcnet'
