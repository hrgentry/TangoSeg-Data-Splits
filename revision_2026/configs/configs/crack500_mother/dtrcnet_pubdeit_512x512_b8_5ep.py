# crack500 母图互斥划分臂, public-DeiT 变体。**唯一变量**同 dtrcnet_512x512_b8_5ep.py:
# ann_file 由 splits/ 改为 splits_mother/。模型/优化器/调度/增广/batch/epoch 全部
# 继承 ../crack500/dtrcnet_pubdeit_512x512_b8_5ep.py, 不在此处重复。
_base_ = ['../crack500/dtrcnet_pubdeit_512x512_b8_5ep.py']

train_dataloader = dict(dataset=dict(ann_file='splits_mother/train.txt'))
val_dataloader = dict(dataset=dict(ann_file='splits_mother/val.txt'))
test_dataloader = dict(dataset=dict(ann_file='splits_mother/test.txt'))

work_dir = './work_dirs/mixercseg_baselines/crack500_mother/dtrcnet_pubdeit_512x512_b8_5ep'
