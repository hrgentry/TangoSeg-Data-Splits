# MambaVision-Tiny (~13.57M). Requires mamba_ssm (WSL recommended).
# ImageNet mean/std to match MambaVision pretrained backbone.
model = dict(
    type='MambaVisionSegmentor',
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255,
        size=(512, 512)),
    dim=80,
    in_dim=32,
    depths=(1, 3, 8, 4),
    num_heads=(2, 4, 8, 16),
    window_size=(8, 8, 64, 32),
    mlp_ratio=4.0,
    drop_path_rate=0.3,
    pretrained=None,
    fuse_channels=128,
    bce_loss_weight=1.0,
    dice_loss_weight=5.0,
    threshold=0.5,
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

model_name = 'mambavision'
