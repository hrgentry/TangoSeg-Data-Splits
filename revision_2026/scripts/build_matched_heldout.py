#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Table 6 的新数据源：TANGO 匹配对照，五种子，独立阈值（CACAIE 一审 B4 + A8；R1-m1、R1-m2、R2-5、R2-6）。

每个对照 = 同种子的“改一处的臂 − 完整从零参照 (a1_base)”，种子 42–46，两个数据集。指标族：
  heldout_F1  独立阈值 F1（与 Table 4 同口径）：Crack500（母图互斥）在验证集选阈值，DeepCrack 无
              独立验证集，用测试集按图对半交叉拟合 200 次的均值。来自 work_dirs/rev1/bins/tango。
  clDice / broken / thin / spur
              几何指标，同样在独立阈值下（build_tango_geometry.py 的结果，eval_rev1_geometry.py）：
              clDice 中心线 Dice；broken 骨架 ≥20 px 的真值裂缝被断成 ≥2 段的比例；thin 宽度 <3 px
              的真值骨架召回；spur 每图远离真值裂缝的误检连通块数。
每个指标族 10 个对比（5 个改动 × 2 个数据集），报配对均值、双侧 95% 配对 t 区间 (n=5)、p，族内 Holm。
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
B = R / 'work_dirs/rev1/bins/tango'
ARMS = ['a2_noprop', 'a3_noori', 'a4_nocldice', 'a5_uniform', 'a7_in1k']
DSS = ['deepcrack', 'crack500_mother']
SEEDS = [42, 43, 44, 45, 46]


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [0.0] * len(ps), 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - k) * ps[i]))
        adj[i] = run
    return adj


def family(get):
    rows = []
    for ds in DSS:
        ref = [get(ds, f'a1_base_seed{s}') for s in SEEDS]
        for arm in ARMS:
            av = [get(ds, f'{arm}_seed{s}') for s in SEEDS]
            d = [x - y for x, y in zip(av, ref)]
            m, se = statistics.mean(d), statistics.stdev(d) / math.sqrt(len(d))
            q = stats.t.ppf(0.975, len(d) - 1)
            p = float(2 * stats.t.sf(abs(m / se), len(d) - 1)) if se else 0.0
            # 键名与 build_tango_geometry.py 的几何族一致, 便于同一张补充表排版
            rows.append(dict(dataset=ds, arm=arm, n=len(d), reference_per_seed=ref, arm_per_seed=av,
                             per_seed_change=d, mean=m, sd=statistics.stdev(d),
                             ci95=[m - q * se, m + q * se], p=p))
    for r, h in zip(rows, holm([r['p'] for r in rows])):
        r['p_holm'], r['established'] = h, h < 0.05
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--halves', type=int, default=200)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    F = {}
    t = ST.cmd_table(argparse.Namespace(test=str(B / 'deepcrack__test'), cal=None, groups='none', halves=a.halves, boot=0))
    for k, v in t['models'].items():
        F[('deepcrack', k)] = v['crossfit_F1_mean']
    t = ST.cmd_table(argparse.Namespace(test=str(B / 'crack500_mother__test'), cal=str(B / 'crack500_mother__val'),
                                        groups='crack500', halves=0, boot=0))
    for k, v in t['models'].items():
        F[('crack500_mother', k)] = v['heldout_F1']
    geo = json.loads((R / 'work_dirs/rev1/stats/tango_geometry_inference.json').read_text(encoding='utf-8'))
    assert geo['seeds'] == SEEDS, geo['seeds']
    out = dict(note=__doc__.strip().splitlines()[0], seeds=SEEDS, sign='arm minus complete from-scratch reference',
               families={'heldout_F1': family(lambda ds, lab: F[(ds, lab)])})
    for mk in ('clDice', 'broken', 'thin', 'spur'):
        out['families'][mk] = geo['families'][mk]
    Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False) + '\n', encoding='utf-8', newline='\n')
    for mk, rows in out['families'].items():
        print('==', mk)
        for r in rows:
            print(f"  {r['dataset'][:8]:8s} {r['arm']:12s} {r['mean']:+7.2f} [{r['ci95'][0]:+.2f},{r['ci95'][1]:+.2f}] "
                  f"holm={r['p_holm']:.3f}{' *' if r['established'] else ''}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
