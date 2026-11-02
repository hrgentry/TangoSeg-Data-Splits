# Arm A2 (no propagation, T=0) on the public DeepCrack 50-epoch protocol.
# Public-set replacement for the crack_v5 A2 arm (TRB paper uses public data
# only). Orientation head/loss stay so the only change is that evidence never
# flows; reference is the from-scratch tango_budget 50ep runs.
_base_ = './tangoseg_512x512_b8_50ep.py'

model = dict(decode_head=dict(propagation=dict(iterations=0)))

work_dir = './work_dirs/tango_arms/a2_noprop'
