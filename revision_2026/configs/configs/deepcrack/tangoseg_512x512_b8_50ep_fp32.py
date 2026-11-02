# TANGO 50-epoch arm in fp32, for hosts on torch 2.0 (e.g. <lan-host>).
# torch 2.0 has no bfloat16 kernel for upsample_bilinear2d, which TangoHead hits
# when it resizes to the detail branch, so the shared bf16 AMP wrapper crashes
# there. fp32 is used rather than fp16 because the recurrent propagation is
# unstable in fp16 (see tangoseg_512x512_b8_5ep.py). TANGO is cheap enough that
# dropping AMP is affordable; note the precision difference when comparing
# against the bf16 arms.
_base_ = ['./tangoseg_512x512_b8_50ep.py']

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=5e-4, weight_decay=0.01),
    clip_grad=dict(max_norm=1.0),
    paramwise_cfg=dict(norm_decay_mult=0.0, bias_decay_mult=0.0))

work_dir = './work_dirs/mixercseg_long/deepcrack/tangoseg_512x512_b8_50ep_fp32'
