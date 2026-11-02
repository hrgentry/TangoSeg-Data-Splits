#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""B7 原始分辨率对照的统计（CACAIE 一审 R2-6 后半）。纯 CPU，读 eval_rev1_native.py 的结果。

两臂都在原始分辨率真值上评测（独立阈值，eval_rev1_native.py）：
  resize512  矩阵的 512x512 训练权重（种子 42 = work_dirs/matrix_50ep，43/44 = B1 的 b1_50ep_s<seed>）；
  native     B7 在原始分辨率上训练的权重（.228，同种子）。
对每个数据集 × 模型，同种子配对 native − resize512：均值、双侧 95% 配对 t 区间 (n=3)、p；
每个指标的 12 个对比（2 个数据集 × 6 个模型）为一族做 Holm 校正——与 Table 6 同一方法。
指标：heldout_F1、clDice、broken（骨架 ≥20 px 的真值裂缝被断开的比例）、thin（宽度 <3 px 的
真值骨架召回）、width_bias（预测宽度偏差中位数，px）。前四个乘 100 报百分点。
另报每个种子上两臂 6 模型排名的 Spearman 相关与换位个数。
"""
import argparse
import json
import math
import statistics
from pathlib import Path

from scipy import stats

R = Path(__file__).resolve().parents[2]
DSS = ['camcrack789', 'crackmap_parent']
MODELS = ['tangoseg', 'unet', 'segformer_mit_b5', 'mixercseg_faithful', 'scsegamba', 'convnext_xlarge']
SEEDS = [42, 43, 44]
MET = {'heldout_F1': lambda r: r['heldout_F1'],
       'clDice': lambda r: 100 * r['geometry']['clDice_mean'],
       'broken': lambda r: 100 * r['geometry']['broken_frac'],
       'thin': lambda r: 100 * (r['geometry']['recall_by_width']['thin'] or 0.0),
       'width_bias': lambda r: r['geometry']['width_bias_px_median']}


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [0.0] * len(ps), 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - k) * ps[i]))
        adj[i] = run
    return adj


def load(seed, ds, mode, m):
    d = R / 'work_dirs/rev1' / ('native_eval' if seed == 42 else f'native_eval_s{seed}') / ds / mode / f'{m}.json'
    return json.loads(d.read_text(encoding='utf-8'))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    res = {(s, ds, mo, m): load(s, ds, mo, m) for s in SEEDS for ds in DSS for mo in ('resize512', 'native') for m in MODELS}
    out = dict(note=__doc__.strip().splitlines()[0], seeds=SEEDS, sign='native minus resize512, same seed',
               families={}, ranks={})
    for mk, fn in MET.items():
        rows = []
        for ds in DSS:
            for m in MODELS:
                r5 = [fn(res[(s, ds, 'resize512', m)]) for s in SEEDS]
                nv = [fn(res[(s, ds, 'native', m)]) for s in SEEDS]
                d = [x - y for x, y in zip(nv, r5)]
                mean, se = statistics.mean(d), statistics.stdev(d) / math.sqrt(len(d))
                q = stats.t.ppf(0.975, len(d) - 1)
                p = float(2 * stats.t.sf(abs(mean / se), len(d) - 1)) if se else 0.0
                rows.append(dict(dataset=ds, model=m, resize512=r5, native=nv, per_seed_change=d,
                                 mean=mean, ci95=[mean - q * se, mean + q * se], p=p))
        for r, h in zip(rows, holm([r['p'] for r in rows])):
            r['p_holm'], r['established'] = h, h < 0.05
        out['families'][mk] = rows
    for ds in DSS:
        for s in SEEDS:
            f = lambda mo, m: res[(s, ds, mo, m)]['heldout_F1']
            a5, nv = [f('resize512', m) for m in MODELS], [f('native', m) for m in MODELS]
            oa = sorted(MODELS, key=lambda m: -f('resize512', m))
            ob = sorted(MODELS, key=lambda m: -f('native', m))
            out['ranks'][f'{ds}/s{s}'] = dict(spearman=float(stats.spearmanr(a5, nv)[0]),
                                              moved=sum(x != y for x, y in zip(oa, ob)),
                                              resize512_order=oa, native_order=ob)
    Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False) + '\n', encoding='utf-8', newline='\n')
    for mk, rows in out['families'].items():
        print('==', mk)
        for r in rows:
            print(f"  {r['dataset'][:8]:8s} {r['model']:20s} {r['mean']:+7.2f} [{r['ci95'][0]:+.2f},{r['ci95'][1]:+.2f}]"
                  f" per-seed {['%+.2f' % x for x in r['per_seed_change']]} holm={r['p_holm']:.3f}{' *' if r['established'] else ''}")
    for k, v in out['ranks'].items():
        print(f"  rank {k:22s} spearman {v['spearman']:+.2f} moved {v['moved']}/6")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
