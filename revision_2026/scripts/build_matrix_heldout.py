#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Table 4 的新数据源：独立标定阈值下的矩阵（CACAIE 一审 A1/A2, R2-1）。

R2-1 要求主结果用不在测试集上拟合的阈值：在不相交的数据上选 t*、冻结、报测试集 F1 与 IoU，
同时给出 AP 与 oracle pixel-ODS。作者决定（2026-10-30）：Table 4 主格改为独立阈值 F1，
oracle pixel-ODS 与 AP 移到补充材料；等 B1 三种子齐了再一次性替换稿件数字。

每个 (种子, 数据集, 模型)：
  heldout_F1 / heldout_IoU  验证集上选阈值后在测试集上的 F1 与前景 IoU（主数字）；
                            DeepCrack 的验证文件就是测试文件，改用测试集按图随机对半交叉拟合
                            （每半用另一半选的阈值），重复 200 次取平均；
  ODS                       oracle pixel-ODS（在测试集上选阈值，上界，即原 Table 4 的数）；
  AP                        与阈值无关的排序指标；
  boot_*                    测试集抽样区间：按源照片（切块数据）或按图重抽样 10,000 次；
                            nested = 每次重抽样都重新选阈值，fixed = 阈值固定在全集 t*。
跨种子：每个数值报各种子的值、均值与样本标准差；种子只有一个时 sd 为 null。
相邻名次的嵌套区间按第一个种子（42）计算，只反映测试集抽样，不含训练随机性——稿件须照此措辞。

数据来自 eval_rev1_bins.py 的直方图（work_dirs/rev1/bins/m50_s<seed>/...），统计由
rev1_bins_stats.py 计算。输出 JSON 给论文仓 data/matrix_heldout.json。
"""
import argparse
import importlib.util
import json
import statistics
import sys
from pathlib import Path

R = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location('_stats', R / 'projects/mixercseg_baselines/rev1_bins_stats.py')
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)

BINS = R / 'work_dirs/rev1/bins'
# (数据集, 测试划分目录, 验证划分目录或 None, 分组规则)
DATASETS = [
    ('deepcrack', 'deepcrack__test', None, 'none'),
    ('crack500', 'crack500__test', 'crack500__val', 'crack500'),
    ('crack500_mother', 'crack500_mother__test', 'crack500_mother__val', 'crack500'),
    ('camcrack789', 'camcrack789__test', 'camcrack789__val', 'none'),
    ('crackmap_parent', 'crackmap_parent__test', 'crackmap_parent__val', 'crackmap'),
    # 原 3.1 节 CrackMap 对照的另一臂（图像级随机划分）; 不进 Table 4, 只给划分对照用
    ('crackmap', 'crackmap__test', 'crackmap__val', 'crackmap'),
]
MEAN_DS = ['deepcrack', 'crack500_mother', 'camcrack789', 'crackmap_parent']   # Table 4 的 Mean 列
FIELDS = ['heldout_F1', 'heldout_IoU', 'heldout_thr', 'ODS', 'ODS_thr', 'AP', 'IoU_at_ODS']


def one(seed, ds, test, cal, groups, boot, halves):
    a = argparse.Namespace(test=BINS / f'm50_s{seed}' / test,
                           cal=(BINS / f'm50_s{seed}' / cal) if cal else None,
                           groups=groups, boot=boot, halves=halves if cal is None else 0)
    t = S.cmd_table(a)
    out = {}
    for k, v in t['models'].items():
        if cal is None:   # DeepCrack: 交叉拟合
            v = dict(v, heldout_F1=v['crossfit_F1_mean'], heldout_IoU=v['crossfit_IoU_mean'],
                     heldout_thr=None)
        out[k] = {f: v.get(f) for f in FIELDS}
        for b in ('boot_nested_ODS_ci', 'boot_nested_halfwidth', 'boot_fixed_thr_F1_ci',
                  'boot_fixed_thr_halfwidth'):
            if b in v:
                out[k][b] = v[b]
    return out, (t.get('adjacent_pairs_nested'), t.get('all_pairs_nested'))


def agg(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return dict(per_seed=[round(x, 3) for x in vals], mean=round(statistics.mean(vals), 3),
                sd=round(statistics.stdev(vals), 3) if len(vals) > 1 else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    ap.add_argument('--boot', type=int, default=10000)
    ap.add_argument('--halves', type=int, default=200)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    seeds = [s for s in a.seeds
             if all((BINS / f'm50_s{s}' / t).is_dir() for _, t, _, _ in DATASETS)]
    per = {}       # per[seed][ds][model] = dict
    adj, allp = {}, {}
    for s in seeds:
        per[s] = {}
        for ds, test, cal, groups in DATASETS:
            per[s][ds], pairs = one(s, ds, test, cal, groups, a.boot if s == seeds[0] else 0, a.halves)
            if s == seeds[0]:
                adj[ds], allp[ds] = pairs
    models = sorted(set.intersection(*[set(per[s][ds]) for s in seeds for ds, *_ in DATASETS]))
    res = {}
    for ds, *_ in DATASETS:
        res[ds] = {}
        for m in models:
            res[ds][m] = {f: agg([per[s][ds][m][f] for s in seeds]) for f in FIELDS}
            for b in ('boot_nested_ODS_ci', 'boot_nested_halfwidth', 'boot_fixed_thr_F1_ci',
                      'boot_fixed_thr_halfwidth'):
                if b in per[seeds[0]][ds][m]:
                    res[ds][m][b + f'_seed{seeds[0]}'] = per[seeds[0]][ds][m][b]
    mean4 = {m: agg([statistics.mean(per[s][ds][m]['heldout_F1'] for ds in MEAN_DS) for s in seeds])
             for m in models}
    out = dict(note=__doc__.strip().splitlines()[0], seeds=seeds, boot=a.boot, halves=a.halves,
               mean_datasets=MEAN_DS, datasets=res, mean_heldout_F1=mean4,
               adjacent_pairs_nested_seed=seeds[0], adjacent_pairs_nested=adj,
               all_pairs_nested=allp)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False) + '\n', encoding='utf-8',
                           newline='\n')
    print('seeds %s, %d models -> %s' % (seeds, len(models), a.out))


if __name__ == '__main__':
    sys.exit(main())
