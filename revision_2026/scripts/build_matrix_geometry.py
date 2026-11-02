#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""十六模型矩阵的几何指标汇总（CACAIE 一审 A8; R1-m2, R2-6）。纯 CPU。

输入 work_dirs/rev1/geometry/<数据集>/<模型>.json（eval_rev1_geometry.py, 种子 42 的矩阵权重,
独立阈值）。每个数据集报：
  - 几何量与 F1 排名的一致程度：Spearman ρ（F1 取同一阈值下的独立阈值 F1, 来自
    data/matrix_heldout.json 的种子 42）对 clDice、断开比例（越小越好, 取负）、细裂缝召回；
  - 按宽度分层的召回（<3 px 细、3–7 px 中、>7 px 宽）跨模型中位数；
  - 预测宽度偏差（每图 预测面积/预测骨架长 − 真值面积/真值骨架长 的中位数）跨模型范围。
"""
import argparse
import json
import statistics
from pathlib import Path

from scipy import stats

R = Path(__file__).resolve().parents[2]
G = R / 'work_dirs/rev1/geometry'
DSS = ['deepcrack', 'crack500_mother', 'camcrack789', 'crackmap_parent']


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    h = json.loads((R / 'paper/tangoSeg/data/matrix_heldout.json').read_text(encoding='utf-8'))['datasets']
    out = dict(note=__doc__.strip().splitlines()[0], seed=42, datasets={})
    for ds in DSS:
        S = {p.stem: json.loads(p.read_text(encoding='utf-8'))['summary'] for p in (G / ds).glob('*.json')}
        ms = sorted(S)
        f1 = [h[ds][m]['heldout_F1']['per_seed'][0] for m in ms]
        rho = dict(clDice=float(stats.spearmanr(f1, [S[m]['clDice_mean'] for m in ms])[0]),
                   broken=float(stats.spearmanr(f1, [-S[m]['broken_frac'] for m in ms])[0]),
                   thin=float(stats.spearmanr(f1, [S[m]['recall_by_width']['thin'] or 0 for m in ms])[0]))
        rec = {w: statistics.median(S[m]['recall_by_width'][w] for m in ms if S[m]['recall_by_width'][w] is not None)
               for w in ('thin', 'medium', 'wide')}
        wb = [S[m]['width_bias_px_median'] for m in ms]
        br = [S[m]['broken_frac'] for m in ms]
        out['datasets'][ds] = dict(n_models=len(ms), rho_with_F1=rho, recall_by_width_median=rec,
                                   width_bias_min=min(wb), width_bias_max=max(wb), broken_min=min(br), broken_max=max(br),
                                   n_gt_long=S[ms[0]]['n_gt_long'],
                                   models={m: dict(F1=f, clDice=S[m]['clDice_mean'], broken=S[m]['broken_frac'],
                                                   thin=S[m]['recall_by_width']['thin'], width_bias=S[m]['width_bias_px_median'],
                                                   spurious=S[m]['spurious_count_mean'])
                                           for m, f in zip(ms, f1)})
    allr = [abs(v) for d in out['datasets'].values() for v in d['rho_with_F1'].values()]
    out['abs_rho_range'] = [min(allr), max(allr)]
    allw = [d['width_bias_min'] for d in out['datasets'].values()] + [d['width_bias_max'] for d in out['datasets'].values()]
    out['width_bias_range'] = [min(allw), max(allw)]
    Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False) + '\n', encoding='utf-8', newline='\n')
    for ds, d in out['datasets'].items():
        print(ds, {k: round(v, 2) for k, v in d['rho_with_F1'].items()}, {k: round(v, 3) for k, v in d['recall_by_width_median'].items()},
              'wb', round(d['width_bias_min'], 2), round(d['width_bias_max'], 2), 'broken', round(d['broken_min'], 3), round(d['broken_max'], 3))
    print('|rho| range', [round(x, 2) for x in out['abs_rho_range']], 'width bias range', [round(x, 2) for x in out['width_bias_range']])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
