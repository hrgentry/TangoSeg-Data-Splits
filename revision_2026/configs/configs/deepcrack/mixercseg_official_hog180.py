# Architecture-only config for loading the *released* MixerCSeg weights.
# The published checkpoints were trained with hog_bins=180, while our faithful
# port defaults to 36, so loading them against the training config fails with a
# shape mismatch on every hog_feat conv. Nothing else changes; this config is
# for evaluation of official weights, never for training.
_base_ = ['./mixercseg_faithful_512x512_b8_5ep.py']

model = dict(backbone=dict(hog_bins=180))
