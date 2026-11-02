# Arm A5 (uniform steering, w_k = 1/K) on the public DeepCrack 50-epoch
# protocol. Public-set replacement for the crack_v5 A5 arm. Orientation head
# and loss are KEPT so this isolates only the use of the field for steering,
# not its existence.
_base_ = './tangoseg_512x512_b8_50ep.py'

model = dict(decode_head=dict(propagation=dict(steering='uniform')))

work_dir = './work_dirs/tango_arms/a5_uniform'
