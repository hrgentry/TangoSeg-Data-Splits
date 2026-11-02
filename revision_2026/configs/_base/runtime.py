custom_imports = dict(
    imports=[
        'mmpretrain.models',
        'projects.deepcrack',
        'projects.shuttlenet',
        'projects.dinov3_sealing.crack_modules',
        'projects.mixercseg_baselines.adapters',
        'projects.mixercseg_baselines.transforms',
        'projects.tangoseg',
        'projects.mixercseg',
    ],
    allow_failed_imports=False)

default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='spawn', opencv_num_threads=0),
    dist_cfg=dict(backend='gloo'))
launcher = 'none'

vis_backends = [dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(by_epoch=True)
log_level = 'INFO'
# Do not set load_from here — model bases that need ADE20K/ImageNet
# checkpoints define their own load_from; duplicate keys across bases raise.
resume = False
randomness = dict(seed=42)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=10, log_metric_by_epoch=True),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=True,
        interval=1,
        max_keep_ckpts=3,
        save_best='fgIoU',
        rule='greater'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))

val_evaluator = dict(
    type='BinaryFgIoUMetric',
    iou_metrics=['mIoU', 'mDice', 'mFscore'])
test_evaluator = val_evaluator
