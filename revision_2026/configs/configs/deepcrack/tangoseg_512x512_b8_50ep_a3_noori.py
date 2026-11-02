# Arm A3 (no orientation supervision) on the public DeepCrack 50-epoch
# protocol. Public-set replacement for the crack_v5 A3 arm. The field is
# free-learned, steering still uses it; isolates whether structure-tensor
# supervision earns its place.
_base_ = './tangoseg_512x512_b8_50ep.py'

model = dict(
    decode_head=dict(loss_ori=dict(type='OrientationLoss', loss_weight=0.0)))

work_dir = './work_dirs/tango_arms/a3_noori'
