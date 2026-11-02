# CACAIE 一审 B5（R2-7）：TANGO + ImageNet-1k 分类预训练 MSCAN-S trunk，camcrack789。
# 与同目录 tangoseg_ade20k_512x512_b8_50ep.py 成对：两者都只改 trunk 初始化，其余一路
# 继承 ./tangoseg_512x512_b8_50ep.py，并在 .228 上用同一个队列、同一组种子与那组从零
# 参照对比。原 A7 臂（tangoseg_512x512_b8_50ep_a7_in1k.py，仅 DeepCrack 与母图互斥
# Crack500）用的是本机另训的从零参照，两种预训练的效应因此不能直接比大小；本臂补齐
# 四个数据集、共用参照后才可以。权重与 A7 是同一个 checkpoint。
_base_ = ['./tangoseg_512x512_b8_50ep.py']

checkpoint_file = ('https://download.openmmlab.com/mmsegmentation/v0.5/'
                   'pretrain/segnext/mscan_s_20230227-f33ccdf2.pth')

model = dict(
    backbone=dict(
        trunk=dict(
            init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file))))

work_dir = './work_dirs/mixercseg_baselines/camcrack789/tangoseg_in1k_512x512_b8_50ep'
