# Arm A4 (no SoftClDice) on the public DeepCrack 50-epoch protocol.
# Public-set replacement for the crack_v5 A4 arm; isolates the topology loss.
# Lists replace wholesale under _base_ merge, so the loss list is restated.
_base_ = './tangoseg_512x512_b8_50ep.py'

model = dict(
    decode_head=dict(loss_decode=[
        dict(type='CrossEntropyLoss', use_sigmoid=True, class_weight=[5.0],
             loss_weight=1.0),
        dict(type='TangoBinaryDiceLoss', smooth=1.0, loss_weight=1.0),
    ]))

work_dir = './work_dirs/tango_arms/a4_nocldice'
