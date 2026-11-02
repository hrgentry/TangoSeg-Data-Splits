# DeepCrack with the ORIGINAL side-fusion order of Zou et al. (TIP 2019):
# concatenate, upsample the concatenated features, then convolve. The matrix's
# `deepcrack_fast_fuse` row instead convolves at the native side resolution and
# upsamples the single-channel result -- same topology and parameter count, but
# not the published operation order. This config exists to measure what that
# difference is worth under the common protocol.
#
# Verified against https://github.com/qinnzou/DeepCrack (2026-09-05): identical
# parameter names and shapes (74 tensors, 30.91M); loading the official weights
# into this wrapper and running both forwards gives a maximum absolute logit
# difference of 0.0 on all six outputs.
model = dict(
    type='DeepCrackSegmentor',
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

model_name = 'deepcrack_official_fuse'
