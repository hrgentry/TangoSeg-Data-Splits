#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""TANGO 匹配对照的几何指标配对推断（CACAIE 一审 A8; R1-m2, R2-6）。

输入 work_dirs/rev1/geometry_tango/<数据集>/<臂>_seed<S>.json（eval_rev1_geometry.py, 独立阈值）。
对每个先验开关与 IN-1k 初始化, 同种子减参照臂 (a1_base), 报配对均值、双侧 95% 配对 t 区间、p,
并在每个指标的全部对比 (两个数据集 × 5 个对照 = 10 个) 内做 Holm 校正——与 Table 6 的
build_matched_inference.py 同一方法, 只是指标换成几何量:
  broken   骨架 ≥20 px 的真值裂缝被断成 ≥2 段的比例 (百分点)
  thin     宽度 <3 px 的真值骨架被召回的比例 (百分点)
  clDice   中心线 Dice (百分点)
  spur     每图远离任何真值裂缝的误检连通块数
种子取两个数据集上都齐的那些 (B4 之后是 42–46)。
"""
import argparse
import json
import math
import statistics
import sys
from pathlib import Path

from scipy import stats

R = Path(__file__).resolve().parents[2]
G = R / 'work_dirs/rev1/geometry_tango'
ARMS = ['a2_noprop', 'a3_noori', 'a4_nocldice', 'a5_uniform', 'a7_in1k']
DSS = ['deepcrack', 'crack500_mother']
MET = {'broken': lambda s: 100 * s['broken_frac'],
       'thin': lambda s: 100 * s['recall_by_width']['thin'],
       'clDice': lambda s: 100 * s['clDice_mean'],
       'spur': lambda s: s['spurious_count_mean']}


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [0.0] * len(ps), 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - k) * ps[i]))
        adj[i] = run
    return adj


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    S = {}
    for ds in DSS:
        for p in (G / ds).glob('*.json'):
            S[(ds, p.stem)] = json.loads(p.read_text(encoding='utf-8'))['summary']
    seeds = [s for s in range(40, 60)
             if all((ds, f'{arm}_seed{s}') in S for ds in DSS for arm in ARMS + ['a1_base'])]
    out = dict(note=__doc__.strip().splitlines()[0], seeds=seeds, sign='arm minus reference', families={})
    for mk, fn in MET.items():
        rows = []
        for ds in DSS:
            ref = [fn(S[(ds, f'a1_base_seed{s}')]) for s in seeds]
            for arm in ARMS:
                arm_v = [fn(S[(ds, f'{arm}_seed{s}')]) for s in seeds]
                d = [x - y for x, y in zip(arm_v, ref)]
                n = len(d)
                m = statistics.mean(d)
                se = statistics.stdev(d) / math.sqrt(n)
                t = m / se if se else 0.0
                q = stats.t.ppf(0.975, n - 1)
                rows.append(dict(dataset=ds, arm=arm, n=n, per_seed_change=d, reference_per_seed=ref,
                                 arm_per_seed=arm_v, mean=m, ci95=[m - q * se, m + q * se],
                                 p=float(2 * stats.t.sf(abs(t), n - 1))))
        for r, h in zip(rows, holm([r['p'] for r in rows])):
            r['p_holm'] = h
            r['established'] = h < 0.05
        out['families'][mk] = rows
    Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False) + '\n', encoding='utf-8', newline='\n')
    print('seeds', seeds, '->', a.out)
    for mk, rows in out['families'].items():
        print(mk, [(r['dataset'][:4], r['arm'], round(r['mean'], 2)) for r in rows if r['established']])


if __name__ == '__main__':
    sys.exit(main())
