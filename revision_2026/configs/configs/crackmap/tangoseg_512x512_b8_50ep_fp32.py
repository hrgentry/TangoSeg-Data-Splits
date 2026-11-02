# TANGO 50-epoch arm in fp32 on CrackMap; see the DeepCrack fp32 config for why
# torch 2.0 hosts cannot use the shared bf16 wrapper.
_base_ = ['./tangoseg_512x512_b8_50ep.py']

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=5e-4, weight_decay=0.01),
    clip_grad=dict(max_norm=1.0),
    paramwise_cfg=dict(norm_decay_mult=0.0, bias_decay_mult=0.0))

work_dir = './work_dirs/mixercseg_long/crackmap/tangoseg_512x512_b8_50ep_fp32'
