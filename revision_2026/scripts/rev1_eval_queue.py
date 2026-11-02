#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CACAIE 一审：找出所有该评测的 checkpoint，按（实验族, 数据集, 划分）分组调用
eval_rev1_bins.py。已评过且 checkpoint 指纹未变的跳过，所以可以在训练进行中反复运行，
每次只补新出现的产物。

实验族（输出 work_dirs/rev1/bins/<族>/<数据集>__<划分>/<label>.npz）：
  m50_s42            种子 42 的 50 轮矩阵（work_dirs/matrix_50ep），正文现行数字的来源
  m50_s43, m50_s44   B1 新种子
  m5_s42             种子 42 的 5 轮短臂（work_dirs/matrix_final）
  m5_s42mb           RINDNet 种子 42 短臂按微批 4×2 重跑（只有 rindnet）
  m5_s43, m5_s44     B3 新种子短臂
  traj_s43, traj_s44 B3 轨迹：代表模型每 5 轮一个快照，label = <模型>_e<轮>
  tango              TANGO 匹配对照全部种子（42–44 旧 + 45/46 新），label = <臂>_seed<S>；
                     DeepCrack 的参照臂 a1_base 来自 work_dirs/tango_budget/tango_50ep_seed*
                     （原三种子就是这样配对的），另含 5 轮预算臂 tango_5ep_seed*
划分：50 轮矩阵评测试集与验证集（验证集用于 held-out 阈值，R2-1）；DeepCrack 的验证
文件与测试文件相同，只评测试集，held-out 改用测试集随机对半交叉拟合。短臂、轨迹、TANGO
对照只评测试集。

--env win 只跑 Windows 能跑的模型；--env wsl 只跑需要官方 mamba_ssm 的三个模型
（RestorMixer、MambaVision、SCSegamba），须在本机 WSL 的 ~/mmseg_env 里调用本脚本。
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

R = Path(__file__).resolve().parents[2]
CFG = 'projects/mixercseg_baselines/configs'
OUT = R / 'work_dirs/rev1/bins'

MODELS = ['convnext_xlarge', 'hrnet_w48', 'rindnet', 'deeplabv3plus_r50', 'shuttlenet_v2',
          'tangoseg', 'unet', 'carnet', 'deepcrack_fast_fuse', 'segformer_mit_b5',
          'dtrcnet_pubdeit', 'mixercseg_improved', 'mixercseg_faithful',
          'restormixer', 'mambavision', 'scsegamba']
WSL_MODELS = {'restormixer', 'mambavision', 'scsegamba'}
REP = ['convnext_xlarge', 'segformer_mit_b5', 'unet', 'tangoseg', 'mixercseg_faithful', 'scsegamba']
TRAJ_DS = ['deepcrack', 'crack500_mother', 'camcrack789', 'crackmap_parent']

DS = {  # 数据集 -> (data_root, 测试划分, 验证划分或 None)
    'deepcrack': ('data/DeepCrack/dataset', 'splits/test.txt', None),
    'crack500': ('data/Crack500/dataset', 'splits/test.txt', 'splits/val.txt'),
    'crack500_mother': ('data/Crack500/dataset', 'splits_mother/test.txt', 'splits_mother/val.txt'),
    'camcrack789': ('data/CamCrack789/dataset', 'splits/test.txt', 'splits/val.txt'),
    'crackmap': ('data/CrackMap/dataset', 'splits/test.txt', 'splits/val.txt'),
    'crackmap_parent': ('data/CrackMap/dataset', 'splits_parent/test.txt', 'splits_parent/val.txt'),
}
TANGO_ARMS = ['a2_noprop', 'a3_noori', 'a4_nocldice', 'a5_uniform', 'a7_in1k']


def jobs():
    """产出 (族, 数据集, 是否评验证集, label, 配置, checkpoint, 模型名)。"""
    def mat(family, root, final, val):
        for ds in DS:
            for m in MODELS:
                yield (family, ds, val, m, f'{CFG}/{ds}/{m}_512x512_b8_5ep.py',
                       R / root / ds / m / final, m)

    yield from mat('m50_s42', 'work_dirs/matrix_50ep', 'epoch_50.pth', True)
    # R2-1 最后一问: 在未触碰的数据上检验“取最后一轮”规则。种子 42 的每个格子都留着
    # 按验证集 fgIoU@0.5 选出的 best_fgIoU_epoch_*.pth; 评测它在验证集与测试集上的直方图,
    # 与 m50_s42 的末轮对照。DeepCrack 的验证文件就是测试文件, 照评, 分析时单列。
    for ds in DS:
        for m in MODELS:
            bests = sorted((R / 'work_dirs/matrix_50ep' / ds / m).glob('best_fgIoU_epoch_*.pth'))
            if bests:
                yield ('best50_s42', ds, True, m, f'{CFG}/{ds}/{m}_512x512_b8_5ep.py', bests[-1], m)
    yield from mat('m5_s42', 'work_dirs/matrix_final', 'epoch_5.pth', False)
    for s in (43, 44):
        yield from mat(f'm50_s{s}', f'work_dirs/rev1/b1_50ep_s{s}', 'epoch_50.pth', True)
        yield from mat(f'm5_s{s}', f'work_dirs/rev1/b3_5ep_s{s}', 'epoch_5.pth', False)
        for ds in TRAJ_DS:
            for m in REP:
                for e in range(5, 50, 5):
                    yield (f'traj_s{s}', ds, False, f'{m}_e{e:02d}',
                           f'{CFG}/{ds}/{m}_512x512_b8_5ep.py',
                           R / f'work_dirs/rev1/b1_50ep_s{s}' / ds / m / f'epoch_{e}.pth', m)
    for ds in DS:
        yield ('m5_s42mb', ds, False, 'rindnet', f'{CFG}/{ds}/rindnet_512x512_b8_5ep.py',
               R / 'work_dirs/rev1/b3_5ep_s42' / ds / 'rindnet' / 'epoch_5.pth', 'rindnet')

    for ds, old in (('deepcrack', 'work_dirs/tango_arms'), ('crack500_mother', 'work_dirs/tango_arms_c500')):
        for s in (42, 43, 44, 45, 46):
            new = s >= 45
            base = R / ('work_dirs/rev1/b4_tango/' + ds if new else old)
            if ds == 'deepcrack' and not new:
                ref = R / f'work_dirs/tango_budget/tango_50ep_seed{s}/epoch_50.pth'
            else:
                ref = base / f'a1_base_seed{s}' / 'epoch_50.pth'
            # 验证集也评: A8 的几何指标与独立阈值都要用验证集选 t* (DeepCrack 无独立验证集, 自动跳过)
            yield ('tango', ds, True, f'a1_base_seed{s}', f'{CFG}/{ds}/tangoseg_512x512_b8_50ep.py',
                   ref, 'tangoseg')
            for arm in TANGO_ARMS:
                yield ('tango', ds, True, f'{arm}_seed{s}',
                       f'{CFG}/{ds}/tangoseg_512x512_b8_50ep_{arm}.py',
                       base / f'{arm}_seed{s}' / 'epoch_50.pth', 'tangoseg')
    for s in (42, 43, 44):
        yield ('tango', 'deepcrack', False, f'tango_5ep_seed{s}',
               f'{CFG}/deepcrack/tangoseg_512x512_b8_5ep.py',
               R / f'work_dirs/tango_budget/tango_5ep_seed{s}/epoch_5.pth', 'tangoseg')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--env', required=True, choices=['win', 'wsl'])
    ap.add_argument('--families', nargs='*', default=None)
    ap.add_argument('--datasets', nargs='*', default=None)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    groups = {}
    for fam, ds, val, label, cfg, ckpt, model in jobs():
        if a.families and fam not in a.families:
            continue
        if a.datasets and ds not in a.datasets:
            continue
        if (model in WSL_MODELS) != (a.env == 'wsl'):
            continue
        if not ckpt.exists():
            continue
        root, test, vsplit = DS[ds]
        for split in [test] + ([vsplit] if val and vsplit else []):
            od = OUT / fam / f'{ds}__{Path(split).stem}'
            dst = od / f'{label}.npz'
            # 只按“产物比 checkpoint 新”粗筛；指纹核对在 eval_rev1_bins.py 里做
            if dst.exists() and dst.stat().st_mtime > ckpt.stat().st_mtime:
                continue
            groups.setdefault((fam, ds, root, split, str(od)), []).append((label, cfg, str(ckpt)))

    n = sum(len(v) for v in groups.values())
    print(f'{n} model-evaluations in {len(groups)} groups', flush=True)
    env = dict(os.environ, PYTHONPATH=str(R), PYTHONIOENCODING='utf-8',
               MIXERCSEG_SELECTIVE_SCAN_BACKEND='oflex')
    rc_all = 0
    for (fam, ds, root, split, od), ms in sorted(groups.items()):
        cmd = [sys.executable, str(R / 'projects/mixercseg_baselines/eval_rev1_bins.py'),
               '--data-root', root, '--split', split, '--out-dir', od]
        for label, cfg, ckpt in ms:
            cmd += ['--model', label, cfg, ckpt]
        print(f'== {fam} {ds} {split}: {len(ms)} models', flush=True)
        if a.dry_run:
            continue
        rc = subprocess.call(cmd, cwd=str(R), env=env)
        if rc:
            print(f'   rc={rc}', flush=True)
            rc_all = 1
    return rc_all


if __name__ == '__main__':
    sys.exit(main())
