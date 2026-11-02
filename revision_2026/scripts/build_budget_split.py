#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""B3：把训练预算拆成“训练时长”与“调度缩放”两部分（CACAIE 一审 R1-M3(a)、R2-3）。纯 CPU。

原 5 轮对 50 轮的对比同时改了两件事：更新次数少了十倍，学习率调度也被压缩到 5 轮（PolyLR
在第 5 轮就衰减到底）。B3 把 6 个代表模型按 50 轮调度重训（B1 的种子 43/44 那批运行），每 5 轮
存一个 checkpoint，于是同一模型、同一种子有三个点：
  truncated   50 轮调度的第 5 轮（与 5 轮臂更新次数相同，但学习率还在高位）；
  compressed  5 轮臂的终点（同样的更新次数，调度压缩到 5 轮）；
  full        50 轮调度的终点。
  调度缩放效应 = compressed − truncated（更新次数相同，只差调度）
  时长效应     = full − truncated（调度相同，只差更新次数）
  合计         = full − compressed（即原矩阵 5 对 50 比较的量）
主指标与 4.2 节其余部分相同：每个运行在自己的最优阈值上评（pixel-ODS），这样预算对比
不混进阈值标定的差别。另外：
  spread      CrackMap（源照片划分）上 16 个模型的短预算离散度，种子 43/44 各自复现；
  traj        代表模型每 5 轮的 pixel-ODS 轨迹（达到终点 1 分以内的最早轮次）；
  rindnet_mb  RINDNet 5 轮臂按 50 轮臂的微批（4 × 累积 2）重跑的种子 42，与原 5 轮臂比较。
"""
import argparse
import importlib.util
import json
import statistics
from pathlib import Path

R = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('st', R / 'projects/mixercseg_baselines/rev1_bins_stats.py')
ST = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ST)
B = R / 'work_dirs/rev1/bins'
REP = ['convnext_xlarge', 'segformer_mit_b5', 'unet', 'tangoseg', 'mixercseg_faithful', 'scsegamba']
DSS = ['deepcrack', 'crack500_mother', 'camcrack789', 'crackmap_parent']
SEEDS = [43, 44]


def ods(path):
    z = ST.load(path)
    return ST.ods(z['pos'].sum(0), z['neg'].sum(0))[0]


def summ(v):
    return dict(median=statistics.median(v), mean=statistics.mean(v), n=len(v),
                n_positive=sum(x > 0 for x in v), min=min(v), max=max(v))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    cells = []
    traj = {}
    for s in SEEDS:
        for ds in DSS:
            for m in REP:
                tr = ods(B / f'traj_s{s}' / f'{ds}__test' / f'{m}_e05.npz')
                co = ods(B / f'm5_s{s}' / f'{ds}__test' / f'{m}.npz')
                fu = ods(B / f'm50_s{s}' / f'{ds}__test' / f'{m}.npz')
                cells.append(dict(seed=s, dataset=ds, model=m, truncated=tr, compressed=co, full=fu,
                                  schedule=co - tr, duration=fu - tr, total=fu - co))
                curve = [ods(B / f'traj_s{s}' / f'{ds}__test' / f'{m}_e{e:02d}.npz') for e in range(5, 50, 5)] + [fu]
                within = next(e for e, v in zip(list(range(5, 50, 5)) + [50], curve) if v >= fu - 1.0)
                traj[f'{m}/{ds}/s{s}'] = dict(epochs=list(range(5, 50, 5)) + [50], ODS=curve, first_within_1pt=within)
    out = dict(note=__doc__.strip().splitlines()[0], seeds=SEEDS, models=REP, datasets=DSS, cells=cells,
               schedule=summ([c['schedule'] for c in cells]), duration=summ([c['duration'] for c in cells]),
               total=summ([c['total'] for c in cells]),
               by_dataset={ds: dict(schedule=summ([c['schedule'] for c in cells if c['dataset'] == ds]),
                                    duration=summ([c['duration'] for c in cells if c['dataset'] == ds]),
                                    total=summ([c['total'] for c in cells if c['dataset'] == ds]))
                           for ds in DSS},
               n_schedule_larger=sum(abs(c['schedule']) > abs(c['duration']) for c in cells),
               traj=traj, first_within_1pt=summ([v['first_within_1pt'] for v in traj.values()]))
    # 短预算离散度复现（CrackMap 源照片划分，16 个模型）
    sp = {}
    for s in [42] + SEEDS:
        fam5 = 'm5_s42' if s == 42 else f'm5_s{s}'
        d5, d50 = B / fam5 / 'crackmap_parent__test', B / f'm50_s{s}' / 'crackmap_parent__test'
        v5 = {p.stem: ods(p) for p in d5.glob('*.npz')}
        v50 = {p.stem: ods(p) for p in d50.glob('*.npz')}
        ms = sorted(set(v5) & set(v50))
        sp[s] = dict(n_models=len(ms), spread_5=max(v5[m] for m in ms) - min(v5[m] for m in ms),
                     spread_50=max(v50[m] for m in ms) - min(v50[m] for m in ms),
                     lowest_5=min(ms, key=v5.get), values_5={m: v5[m] for m in ms}, values_50={m: v50[m] for m in ms})
    out['spread_crackmap'] = sp
    # RINDNet 微批
    rm = {}
    for p in sorted((B / 'm5_s42mb').glob('*/rindnet.npz')):
        ds = p.parent.name.replace('__test', '')
        rm[ds] = dict(microbatch_4x2=ods(p), original=ods(B / 'm5_s42' / p.parent.name / 'rindnet.npz'))
        rm[ds]['diff'] = rm[ds]['microbatch_4x2'] - rm[ds]['original']
    out['rindnet_mb'] = rm
    Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False) + '\n', encoding='utf-8', newline='\n')
    r3 = lambda d: {k: round(v, 2) if isinstance(v, float) else v for k, v in d.items()}
    print('schedule', r3(out['schedule'])); print('duration', r3(out['duration'])); print('total', r3(out['total']))
    print('|schedule|>|duration| in', out['n_schedule_larger'], '/', len(cells))
    for ds in DSS:
        print(' ', ds, 'sched med', round(out['by_dataset'][ds]['schedule']['median'], 2), 'dur med', round(out['by_dataset'][ds]['duration']['median'], 2))
    print('first within 1pt', r3(out['first_within_1pt']))
    for s, v in sp.items():
        print('spread s%d: n=%d 5ep %.2f 50ep %.2f lowest %s' % (s, v['n_models'], v['spread_5'], v['spread_50'], v['lowest_5']))
    print('rindnet mb', {k: round(v['diff'], 2) for k, v in rm.items()})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
