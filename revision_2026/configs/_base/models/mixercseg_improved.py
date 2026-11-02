# MixerCSeg IMPROVED arm for public-set fair track (b8/5ep).
# Model / loss / optimizer semantics mirror
# projects/mixercseg/mixercseg_crack_v5_512_improved.py.
# selective_scan_backend='oflex': the CUDA ext IS installed and healthy; the
# earlier 'torch' pin came from a mis-diagnosis (importing the .pyd without
# importing torch first always fails with "DLL load failed" on Windows).
# torch fallback made improved ~2x slower than faithful (embed_dims x2 doubles
# the python scan loop); on oflex improved is the faster arm.

model = dict(
    type='EncoderDecoder',
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        mean=[127.5, 127.5, 127.5],
        std=[127.5, 127.5, 127.5],
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255,
        size=(512, 512)),
    pretrained=None,
    backbone=dict(
        type='MixerCSegBackbone',
        embed_dims=(32, 64, 128, 256),
        depths=(1, 1, 1, 1),
        state_dims=(8, 8, 16, 16),
        mlp_ratio=2.0,
        selective_scan_backend='oflex',
        conv_norm='GN',
        stem_min_channels=16,
        attention_stages=(2, 3),
        dynamic_routing=False,
        use_hog=False),
    decode_head=dict(
        type='MixerCSegHead',
        in_channels=(32, 64, 128, 256),
        in_index=(0, 1, 2, 3),
        channels=64,
        num_classes=2,
        out_channels=1,
        threshold=0.5,
        output_scale=4,
        fuse_low_res=True,
        simple_pred=True,
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                use_sigmoid=True,
                class_weight=[5.0],
                loss_weight=1.0),
            dict(type='BinarySigmoidDiceLoss', smooth=1.0, loss_weight=1.0),
        ]),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

model_name = 'mixercseg_improved'
