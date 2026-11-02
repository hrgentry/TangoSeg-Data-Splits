# MixerCSeg-faithful + gradient clipping, 5 epochs.
# The committed faithful config is the ONLY one of the 16 models in this matrix
# without clip_grad, and it collapsed to near-zero foreground on 3 of 4 datasets
# (DeepCrack 6.84, CamCrack789 0.18, CrackMap 3.88) while every clipped model
# trained normally. This arm isolates that single variable at the same budget.
_base_ = ['./mixercseg_faithful_512x512_b8_5ep.py']

optim_wrapper = dict(clip_grad=dict(max_norm=1.0))

work_dir = './work_dirs/mixercseg_local_oflex/deepcrack/mixercseg_faithful_512x512_b8_5ep_clip'
