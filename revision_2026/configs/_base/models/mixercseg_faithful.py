# MixerCSeg FAITHFUL arm for public-set fair track (b8/5ep).
# Model / loss / optimizer semantics mirror
# projects/mixercseg/mixercseg_crack_v5_512_faithful.py.
# selective_scan_backend='oflex': the CUDA ext IS installed and healthy; the
# earlier 'torch' pin came from a mis-diagnosis (importing the .pyd without
# importing torch first always fails with "DLL load failed" on Windows).
# torch fallback is ~300x slower (72 s/iter vs 0.2 s/iter at b8/512^2).

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
        embed_dims=(16, 32, 64, 128),
        depths=(1, 1, 1, 1),
        state_dims=(8, 8, 16, 16),
        mlp_ratio=2.0,
        selective_scan_backend='oflex'),
    decode_head=dict(
        type='MixerCSegHead',
        in_channels=(16, 32, 64, 128),
        in_index=(0, 1, 2, 3),
        channels=8,
        num_classes=2,
        out_channels=1,
        threshold=0.5,
        output_scale=4,
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=0.87),
            dict(type='BinarySigmoidDiceLoss', smooth=1.0, loss_weight=0.13),
        ]),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

model_name = 'mixercseg_faithful'
