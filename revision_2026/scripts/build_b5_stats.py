#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""B5 预训练来源对照的统计（CACAIE 一审 R2-7）。纯 CPU，读 .228 拉回的逐图直方图。

R2-7：ImageNet-1k 臂与 ADE20K 臂原来各配一组分别训练的从零参照，效应大小不能直接比。
B5 在 .228 上把 IN-1k 臂按 ADE20K 研究的同一队列、同一机器、同一组种子 (42–44) 重跑，
三个臂共用同一组从零参照：
  scratch = tangoseg_s<seed>，ade20k = tangoseg_ade20k_s<seed>，in1k = tangoseg_in1k_s<seed>。
主指标是独立阈值 F1（与 Table 4/6 同口径）：Crack500（母图互斥）、CamCrack789、CrackMap
（源照片划分）在验证集选阈值、冻结；DeepCrack 无独立验证集，用测试集按图对半交叉拟合
200 次的均值。oracle pixel-ODS 一并给出。
三个对比 (ade20k−scratch, in1k−scratch, in1k−ade20k) × 4 个数据集 = 每个指标 12 个对比为一族，
同种子配对：均值、双侧 95% 配对 t 区间 (n=3)、p，族内 Holm 校正。
"""
import argparse
import importlib.util
import json
import math
import statistics
from pathlib import Path

from scipy import stats

R = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('st', R / 'projects/mixercseg_baselines/rev1_bins_stats.py')
ST = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ST)

DSS = {'deepcrack': 'none', 'crack500_mother': 'crack500', 'camcrack789': 'none', 'crackmap_parent': 'crackmap'}
ARMS = {'scratch': 'tangoseg', 'ade20k': 'tangoseg_ade20k', 'in1k': 'tangoseg_in1k'}
CONTRASTS = [('ade20k', 'scratch'), ('in1k', 'scratch'), ('in1k', 'ade20k')]
SEEDS = [42, 43, 44]


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [0.0] * len(ps), 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - k) * ps[i]))
        adj[i] = run
    return adj


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bins', default=str(R / 'work_dirs/rev1/bins_228/b5'))
    ap.add_argument('--halves', type=int, default=200)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    B = Path(a.bins)
    scores = {}
    for ds, groups in DSS.items():
        cal = B / f'{ds}__val'
        has_cal = cal.is_dir() and any(cal.glob('*.npz'))
        if ds != 'deepcrack' and not has_cal:
            raise SystemExit(f'{ds}: 缺验证集直方图 {cal}')
        t = ST.cmd_table(argparse.Namespace(test=str(B / f'{ds}__test'), cal=str(cal) if ds != 'deepcrack' else None,
                                            groups=groups, halves=a.halves if ds == 'deepcrack' else 0, boot=0))
        for k, v in t['models'].items():
            heldout = v['crossfit_F1_mean'] if ds == 'deepcrack' else v['heldout_F1']
            scores[(ds, k)] = dict(heldout_F1=heldout, ODS=v['ODS'], AP=v['AP'])
    out = dict(note=__doc__.strip().splitlines()[0], seeds=SEEDS, sign='first arm minus second, same seed',
               scores={f'{ds}/{k}': v for (ds, k), v in sorted(scores.items())}, families={})
    for metric in ('heldout_F1', 'ODS'):
        rows = []
        for ds in DSS:
            for x, y in CONTRASTS:
                xv = [scores[(ds, f'{ARMS[x]}_s{s}')][metric] for s in SEEDS]
                yv = [scores[(ds, f'{ARMS[y]}_s{s}')][metric] for s in SEEDS]
                d = [p - q for p, q in zip(xv, yv)]
                m, sd = statistics.mean(d), statistics.stdev(d)
                se = sd / math.sqrt(len(d))
                q = stats.t.ppf(0.975, len(d) - 1)
                p = float(2 * stats.t.sf(abs(m / se), len(d) - 1)) if se else 0.0
                rows.append(dict(dataset=ds, contrast=f'{x} - {y}', per_seed_change=d, first=xv, second=yv,
                                 mean=m, ci95=[m - q * se, m + q * se], p=p))
        for r, h in zip(rows, holm([r['p'] for r in rows])):
            r['p_holm'], r['established'] = h, h < 0.05
        out['families'][metric] = rows
    Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False) + '\n', encoding='utf-8', newline='\n')
    for metric, rows in out['families'].items():
        print('==', metric)
        for r in rows:
            print(f"  {r['dataset']:16s} {r['contrast']:16s} {r['mean']:+6.2f} [{r['ci95'][0]:+.2f},{r['ci95'][1]:+.2f}]"
                  f" p={r['p']:.3f} holm={r['p_holm']:.3f}{' *' if r['established'] else ''}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
