# crack500 母图互斥划分臂。**唯一变量**：ann_file 由 splits/ 改为
# splits_mother/（裁块级随机分配 → 母图级分配，比例仍为 7:1:2、裁块集合完全
# 相同）。模型/优化器/调度/增广/batch/epoch 全部继承 ../crack500/mixercseg_improved_512x512_b8_5ep.py，
# 不在此处重复，以免两臂意外分叉。
# 划分由 tools/make_crack500_mother_split.py 生成，母图三两互斥已复核。
_base_ = ['../crack500/mixercseg_improved_512x512_b8_5ep.py']

train_dataloader = dict(dataset=dict(ann_file='splits_mother/train.txt'))
val_dataloader = dict(dataset=dict(ann_file='splits_mother/val.txt'))
test_dataloader = dict(dataset=dict(ann_file='splits_mother/test.txt'))

work_dir = './work_dirs/mixercseg_baselines/crack500_mother/mixercseg_improved_512x512_b8_5ep'
