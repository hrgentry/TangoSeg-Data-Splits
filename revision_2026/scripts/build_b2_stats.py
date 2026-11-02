#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""B2 划分隔离的统计（CACAIE 一审 R1-M4(b)、R2-2）。纯 CPU，读 .228 拉回的逐图直方图。

符号与正文 3.1 一致：去掉共享源照片的一臂减共享的一臂，负值 = 共享源照片时分数更高。

CrackMap 5 折交叉验证（group 臂源照片不跨折，random 臂按图随机分折）：
  两臂各自 5 折的测试图合并后是同一组 120 张图，按图名对齐后逐模型配对。
  - pooled pixel-ODS：120 张图合并计数上选阈值（与正文现行口径相同）；
  - cross-fit F1：每折用其余 4 折合并曲线选的阈值（独立阈值）；
  - 配对嵌套 bootstrap：按源照片（GOPRxxxx）有放回重抽样，两臂在同一组重抽样上各自重选
    阈值，取差值的 2.5/97.5 百分位。
  另报 16 个模型差值的中位数、负值个数和 Wilcoxon 符号秩检验（模型作单位）。

Crack500 固定测试集（3 个分组种子 g0–g2 × 6 个代表模型，clean 减 leak）：
  每个 g 的配对嵌套区间来自 rev1_bins_stats.py paired（按母图重抽样）；这里汇总每个模型
  3 个 g 的均值，以及 18 个差值的中位数与负值个数。
"""
import argparse
import importlib.util
import json
import statistics
from pathlib import Path

import numpy as np
from scipy import stats

R = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('st', R / 'projects/mixercseg_baselines/rev1_bins_stats.py')
ST = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ST)


def concat_arm(fold_dirs, label):
    """5 折的逐图计数按图名排序拼成 120 行。"""
    names, pos, neg = [], [], []
    for d in fold_dirs:
        z = ST.load(Path(d) / f'{label}.npz')
        names += z['names']
        pos.append(z['pos'])
        neg.append(z['neg'])
    pos, neg = np.concatenate(pos), np.concatenate(neg)
    order = np.argsort(names, kind='stable')
    return [names[i] for i in order], pos[order], neg[order]


def crackmap_cv(bins, n_boot):
    cv = ST.cmd_cv(argparse.Namespace(arm=[
        f'{a}=' + ','.join(str(bins / 'b2cv' / a / f'fold{k}') for k in range(5)) for a in ('group', 'random')]))
    rng = np.random.default_rng(ST.SEED)
    rows = {}
    for m, d in cv['diff']['models'].items():
        ng_, pg, ngg = concat_arm([bins / 'b2cv/group' / f'fold{k}' for k in range(5)], m)
        nr_, pr, nrr = concat_arm([bins / 'b2cv/random' / f'fold{k}' for k in range(5)], m)
        assert ng_ == nr_, m
        gidx, ng = ST.group_index(ng_, 'crackmap')
        W = ST.boot_weights(ng, n_boot, rng) if not rows else W  # 同一组重抽样用于所有模型
        og, _ = ST.ods_rows(W @ ST.group_sums(pg, gidx, ng), W @ ST.group_sums(ngg, gidx, ng))
        orr, _ = ST.ods_rows(W @ ST.group_sums(pr, gidx, ng), W @ ST.group_sums(nrr, gidx, ng))
        rows[m] = dict(group=cv['arms']['group']['models'][m], random=cv['arms']['random']['models'][m],
                       diff_pooled_ODS=d['pooled_ODS'], diff_crossfit_F1=d['crossfit_F1'],
                       nested_ci=ST.ci(og - orr))
    v = [r['diff_pooled_ODS'] for r in rows.values()]
    x = [r['diff_crossfit_F1'] for r in rows.values()]
    return dict(order='group - random (source-disjoint folds minus random folds)',
                num_images=120, num_source_photos=ng, boot=n_boot, models=rows,
                pooled_ODS=dict(median=statistics.median(v), mean=statistics.mean(v),
                                n_negative=sum(t < 0 for t in v), n=len(v),
                                wilcoxon_p=float(stats.wilcoxon(v).pvalue),
                                n_ci_below_zero=sum(r['nested_ci'][1] < 0 for r in rows.values()),
                                n_ci_above_zero=sum(r['nested_ci'][0] > 0 for r in rows.values())),
                crossfit_F1=dict(median=statistics.median(x), mean=statistics.mean(x),
                                 n_negative=sum(t < 0 for t in x), n=len(x),
                                 wilcoxon_p=float(stats.wilcoxon(x).pvalue)))


def crack500_iso(bins, n_boot):
    per = {}
    for g in range(3):
        p = ST.cmd_paired(argparse.Namespace(a=str(bins / f'b2iso/g{g}/clean'), b=str(bins / f'b2iso/g{g}/leak'),
                                             groups='crack500', boot=n_boot))
        for m, r in p['models'].items():
            per.setdefault(m, []).append(dict(g=g, clean=r['ODS_a'], leak=r['ODS_b'], diff=r['diff'],
                                              nested_ci=r['nested_ci']))
        nimg, npar = p['num_images'], p['num_groups']
    allv = [r['diff'] for L in per.values() for r in L]
    return dict(order='clean - leak (training crops from unrelated parents minus crops from test parents)',
                num_test_images=nimg, num_test_parents=npar, boot=n_boot,
                models={m: dict(per_group=L, mean=statistics.mean(r['diff'] for r in L)) for m, L in per.items()},
                all_cells=dict(median=statistics.median(allv), mean=statistics.mean(allv),
                               n_negative=sum(t < 0 for t in allv), n=len(allv),
                               n_ci_below_zero=sum(r['nested_ci'][1] < 0 for L in per.values() for r in L),
                               n_ci_above_zero=sum(r['nested_ci'][0] > 0 for L in per.values() for r in L)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bins', default=str(R / 'work_dirs/rev1/bins_228'))
    ap.add_argument('--boot', type=int, default=10000)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    bins = Path(a.bins)
    out = dict(note=__doc__.strip().splitlines()[0], crackmap_cv=crackmap_cv(bins, a.boot),
               crack500_iso=crack500_iso(bins, a.boot))
    Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False) + '\n', encoding='utf-8', newline='\n')
    c, i = out['crackmap_cv'], out['crack500_iso']
    print('CrackMap CV pooled-ODS', {k: round(v, 3) if isinstance(v, float) else v for k, v in c['pooled_ODS'].items()})
    print('CrackMap CV cross-fit ', {k: round(v, 3) if isinstance(v, float) else v for k, v in c['crossfit_F1'].items()})
    for m, r in sorted(c['models'].items(), key=lambda t: t[1]['diff_pooled_ODS']):
        print(f"  {m:20s} {r['diff_pooled_ODS']:+.2f} {r['nested_ci']}")
    print('Crack500 iso', {k: round(v, 3) if isinstance(v, float) else v for k, v in i['all_cells'].items()})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
