# faithful + clip_grad on CamCrack789, where the unclipped arm collapsed hardest
# (fgIoU 4.94 -> 4.74 -> 2.10 -> 0.18 -> 0.18). Single variable vs the committed config.
_base_ = ['./mixercseg_faithful_512x512_b8_5ep.py']

optim_wrapper = dict(clip_grad=dict(max_norm=1.0))

work_dir = './work_dirs/mixercseg_local_oflex/camcrack789/mixercseg_faithful_512x512_b8_5ep_clip'
