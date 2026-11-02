# faithful + clip_grad on CrackMap, where the unclipped arm ended at 3.88 fgIoU.
_base_ = ['./mixercseg_faithful_512x512_b8_5ep.py']

optim_wrapper = dict(clip_grad=dict(max_norm=1.0))

work_dir = './work_dirs/mixercseg_local_oflex/crackmap/mixercseg_faithful_512x512_b8_5ep_clip'
