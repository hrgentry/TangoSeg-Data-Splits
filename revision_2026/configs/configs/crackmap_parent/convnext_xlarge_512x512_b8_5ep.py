# CrackMap 源照片互斥划分臂 (2026-09-11, 对抗性审稿 T0-1)。**唯一变量**: ann_file 由
# splits/ 改为 splits_parent/ (图像级随机分配 -> GoPro 源文件级分配, 比例仍按 7:1:2、
# 图像集合完全相同)。模型/优化器/调度/增广/batch/epoch 全部继承
# ../crackmap/convnext_xlarge_512x512_b8_5ep.py, 不在此处重复, 以免两臂意外分叉。
# 划分由 tools/make_crackmap_parent_split.py 生成, 源文件三两互斥已断言。
_base_ = ['../crackmap/convnext_xlarge_512x512_b8_5ep.py']

train_dataloader = dict(dataset=dict(ann_file='splits_parent/train.txt'))
val_dataloader = dict(dataset=dict(ann_file='splits_parent/val.txt'))
test_dataloader = dict(dataset=dict(ann_file='splits_parent/test.txt'))

work_dir = './work_dirs/mixercseg_baselines/crackmap_parent/convnext_xlarge_512x512_b8_5ep'
