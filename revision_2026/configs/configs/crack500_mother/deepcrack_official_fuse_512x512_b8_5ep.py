# crack500 母图互斥划分臂。**唯一变量**：ann_file 由 splits/ 改为
# splits_mother/。其余全部继承 ../crack500/deepcrack_official_fuse_512x512_b8_5ep.py。
_base_ = ['../crack500/deepcrack_official_fuse_512x512_b8_5ep.py']

train_dataloader = dict(dataset=dict(ann_file='splits_mother/train.txt'))
val_dataloader = dict(dataset=dict(ann_file='splits_mother/val.txt'))
test_dataloader = dict(dataset=dict(ann_file='splits_mother/test.txt'))

work_dir = './work_dirs/mixercseg_baselines/crack500_mother/deepcrack_official_fuse_512x512_b8_5ep'
