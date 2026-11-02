#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Crack500 两种划分在共同测试裁块上的配对比较（CACAIE 一审 A3, R2-2 的最低要求）。

R2-2: 分发划分与母图互斥重划分的测试集只共享 674 个裁块中的 141 个，所以 3.1 节的逐模型
差值同时混着“训练集见没见过测试母图”和“测试集换了一批图”。审稿人的最低要求是在共同图像上
报配对结果。这 141 个裁块的母图在分发划分里全部也在训练集中（分发划分的 356 个测试母图
都出现在训练集），在重划分里都不在训练集中——所以在这批固定的图上比较两次训练，差值只来自
训练集是否见过同一张母图。

每个 (种子, 模型)：两次训练各自用自己的验证集选阈值（独立阈值, 与 A1 的主指标同口径），
冻结后在 141 个共同裁块上算 F1；差值 = 重划分训练 − 分发划分训练（负值 = 见过母图的那次
训练在这批图上得分更高, 与稿件 3.1 节的符号约定一致）。另给按母图分组的配对 bootstrap
区间（只反映测试抽样）。CrackMap 两种划分只共享 3 张测试图, 太少, 只报张数不做统计。
"""
import argparse
import importlib.util
import json
import statistics
import sys
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location('_stats', R / 'projects/mixercseg_baselines/rev1_bins_stats.py')
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)
B = R / 'work_dirs/rev1/bins'


def heldout_bin(fam, val, m):
    v = S.load(B / fam / val / f'{m}.npz')
    return S.ods(v['pos'].sum(0), v['neg'].sum(0))[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    ap.add_argument('--boot', type=int, default=10000)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    seeds = [s for s in a.seeds if (B / f'm50_s{s}' / 'crack500__test').is_dir()
             and (B / f'm50_s{s}' / 'crack500_mother__test').is_dir()]
    res = dict(note=__doc__.strip().splitlines()[0], sign='rebuilt minus distributed', seeds=seeds,
               per_seed={})
    for s in seeds:
        fam = f'm50_s{s}'
        dist_models = {p.stem for p in (B / fam / 'crack500__test').glob('*.npz')}
        reb_models = {p.stem for p in (B / fam / 'crack500_mother__test').glob('*.npz')}
        models = sorted(dist_models & reb_models)
        names_d = S.load(B / fam / 'crack500__test' / f'{models[0]}.npz')['names']
        names_r = S.load(B / fam / 'crack500_mother__test' / f'{models[0]}.npz')['names']
        common = sorted(set(names_d) & set(names_r))
        idx_d = [names_d.index(n) for n in common]
        idx_r = [names_r.index(n) for n in common]
        gidx, ng = S.group_index(common, 'crack500')
        W = S.boot_weights(ng, a.boot, np.random.default_rng(S.SEED)) if a.boot else None
        rows = {}
        for m in models:
            td = S.load(B / fam / 'crack500__test' / f'{m}.npz')
            tr = S.load(B / fam / 'crack500_mother__test' / f'{m}.npz')
            pd_, nd_ = td['pos'][idx_d], td['neg'][idx_d]
            pr_, nr_ = tr['pos'][idx_r], tr['neg'][idx_r]
            bd = heldout_bin(fam, 'crack500__val', m)
            br = heldout_bin(fam, 'crack500_mother__val', m)
            fd = S.at_bin(pd_.sum(0), nd_.sum(0), bd)['F1']
            fr = S.at_bin(pr_.sum(0), nr_.sum(0), br)['F1']
            row = dict(F1_distributed=fd, F1_rebuilt=fr, diff=fr - fd)
            if W is not None:
                def f1w(P, N, b):
                    PS, NS = W @ S.group_sums(P, gidx, ng), W @ S.group_sums(N, gidx, ng)
                    tp = PS[:, b:].sum(1); fp = NS[:, b:].sum(1); fn = PS.sum(1) - tp
                    return 200 * tp / np.maximum(2 * tp + fp + fn, 1)
                row['diff_ci'] = S.ci(f1w(pr_, nr_, br) - f1w(pd_, nd_, bd))
            rows[m] = row
        d = [r['diff'] for r in rows.values()]
        res['per_seed'][str(s)] = dict(
            common_crops=len(common), common_parents=ng, models=rows,
            median_diff=statistics.median(d), mean_diff=statistics.mean(d),
            n_rebuilt_higher=sum(x > 0 for x in d), n=len(d), min_diff=min(d), max_diff=max(d),
            n_ci_excludes_zero=sum(1 for r in rows.values() if 'diff_ci' in r
                                   and (r['diff_ci'][0] > 0 or r['diff_ci'][1] < 0)))
    # CrackMap: 只数共同测试图
    cm = {}
    for s in seeds:
        fam = f'm50_s{s}'
        try:
            nr = S.load(next((B / fam / 'crackmap__test').glob('*.npz')))['names']
            np_ = S.load(next((B / fam / 'crackmap_parent__test').glob('*.npz')))['names']
            cm[str(s)] = len(set(nr) & set(np_))
        except StopIteration:
            pass
    res['crackmap_common_test_images'] = cm
    Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False) + '\n', encoding='utf-8', newline='\n')
    for s, v in res['per_seed'].items():
        print('seed %s: %d common crops / %d parents; median %+.3f mean %+.3f range %+.2f..%+.2f; '
              'rebuilt higher %d/%d; CI excludes 0: %d' % (s, v['common_crops'], v['common_parents'],
              v['median_diff'], v['mean_diff'], v['min_diff'], v['max_diff'], v['n_rebuilt_higher'],
              v['n'], v['n_ci_excludes_zero']))
    print('crackmap common test images', cm)


if __name__ == '__main__':
    sys.exit(main())
