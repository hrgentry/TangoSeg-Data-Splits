#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""稿件 3.1 节划分对照的全部数字（CACAIE 一审 R1-M4(b)、R2-2）。纯 CPU。

符号一律是“去掉共享源照片的一臂 − 共享的一臂”，负值 = 共享源照片时分数更高。主指标是独立
阈值 F1（与 Table 4 同口径），三种子取均值后再跨模型汇总。

  original    原 3.1 节的两组近匹配对比（测试集随划分一起换）：
                Crack500  母图互斥重划分 − 分发划分，各自测试集；
                CrackMap  源照片划分 − 图像级随机划分，各自测试集。
              来自 data/matrix_heldout.json（三种子）。另报两列的名次变动与极差。
  common      A3：Crack500 两种划分共有的 141 个测试裁块上的配对差（data/partition_common.json）。
  fixed_c500  B2：Crack500 固定测试集，clean − leak，3 个分组种子 × 6 模型。独立阈值无验证集可用
              （B2 的 val 是从训练母图抽的，只作日志），改用测试集按母图对半交叉拟合 200 次；
              配对区间是按母图的嵌套 bootstrap（rev1_bins_stats.py paired）。
  cv_crackmap B2：CrackMap 5 折，源照片不跨折 − 随机分折，同一组 120 张测试图
              （build_b2_stats.py 的结果；独立阈值 = 每折用其余 4 折选的阈值）。
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
P = R / 'paper/tangoSeg/data'


def summ(d):
    v = list(d.values())
    return dict(median=statistics.median(v), mean=statistics.mean(v), n=len(v),
                n_negative=sum(x < 0 for x in v), min=min(v), max=max(v))


def ranks_moved(a, b):
    ra = sorted(a, key=lambda m: -a[m])
    rb = sorted(b, key=lambda m: -b[m])
    return sum(ra.index(m) != rb.index(m) for m in a)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bins228', default=str(R / 'work_dirs/rev1/bins_228'))
    ap.add_argument('--halves', type=int, default=200)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    h = json.loads((P / 'matrix_heldout.json').read_text(encoding='utf-8'))
    D = h['datasets']
    F = lambda ds: {m: D[ds][m]['heldout_F1']['mean'] for m in D[ds]}
    out = dict(note=__doc__.strip().splitlines()[0], sign='source-disjoint arm minus shared arm')
    orig = {}
    for name, new, old in (('crack500', 'crack500_mother', 'crack500'), ('crackmap', 'crackmap_parent', 'crackmap')):
        fn, fo = F(new), F(old)
        d = {m: fn[m] - fo[m] for m in fn}
        noR = {m: v for m, v in d.items() if m != 'rindnet'}
        orig[name] = dict(models=d, all=summ(d), without_rindnet=summ(noR),
                          ranks_moved=ranks_moved(fn, fo), spread_new=max(fn.values()) - min(fn.values()),
                          spread_old=max(fo.values()) - min(fo.values()),
                          spread_new_without_rindnet=max(v for m, v in fn.items() if m != 'rindnet') - min(v for m, v in fn.items() if m != 'rindnet'),
                          spread_old_without_rindnet=max(v for m, v in fo.items() if m != 'rindnet') - min(v for m, v in fo.items() if m != 'rindnet'))
    out['original'] = orig
    pc = json.loads((P / 'partition_common.json').read_text(encoding='utf-8'))
    seeds = list(pc['per_seed'])
    cm = {m: statistics.mean(pc['per_seed'][s]['models'][m]['diff'] for s in seeds) for m in pc['per_seed'][seeds[0]]['models']}
    out['common'] = dict(seeds=seeds, common_crops=pc['per_seed'][seeds[0]]['common_crops'],
                         common_parents=pc['per_seed'][seeds[0]]['common_parents'], models=cm, all=summ(cm),
                         without_rindnet=summ({m: v for m, v in cm.items() if m != 'rindnet'}),
                         per_seed_ci_excludes_zero={s: sum((lambda c: c[0] > 0 or c[1] < 0)(v['diff_ci'])
                                                           for v in pc['per_seed'][s]['models'].values()) for s in seeds})
    # B2 Crack500 固定测试集
    b2 = json.loads((R / 'work_dirs/rev1/stats/b2_partition_isolation.json').read_text(encoding='utf-8'))
    B = Path(a.bins228)
    cells = []
    for g in range(3):
        tc = ST.cmd_table(argparse.Namespace(test=str(B / f'b2iso/g{g}/clean'), cal=None, groups='crack500',
                                             halves=a.halves, boot=0))
        tl = ST.cmd_table(argparse.Namespace(test=str(B / f'b2iso/g{g}/leak'), cal=None, groups='crack500',
                                             halves=a.halves, boot=0))
        for m in tc['models']:
            pr = [r for r in b2['crack500_iso']['models'][m]['per_group'] if r['g'] == g][0]
            cells.append(dict(g=g, model=m, heldout_clean=tc['models'][m]['crossfit_F1_mean'],
                              heldout_leak=tl['models'][m]['crossfit_F1_mean'],
                              diff=tc['models'][m]['crossfit_F1_mean'] - tl['models'][m]['crossfit_F1_mean'],
                              diff_ods=pr['diff'], nested_ci_ods=pr['nested_ci']))
        nimg, npar = tc['num_images'], tc['num_groups']
    pm = {m: statistics.mean(c['diff'] for c in cells if c['model'] == m) for m in {c['model'] for c in cells}}
    out['fixed_c500'] = dict(test_crops=nimg, test_parents=npar, groups=3, cells=cells,
                             all_cells=summ({f"{c['model']}/g{c['g']}": c['diff'] for c in cells}),
                             per_model_mean=pm, per_model=summ(pm),
                             n_ci_below_zero=sum(c['nested_ci_ods'][1] < 0 for c in cells),
                             n_ci_above_zero=sum(c['nested_ci_ods'][0] > 0 for c in cells))
    out['cv_crackmap'] = b2['crackmap_cv']
    Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False) + '\n', encoding='utf-8', newline='\n')
    for k in ('crack500', 'crackmap'):
        o = orig[k]
        print(k, 'orig all', {x: round(y, 3) for x, y in o['all'].items()}, 'noR', {x: round(y, 3) for x, y in o['without_rindnet'].items()},
              'ranks moved', o['ranks_moved'], 'spread new/old', round(o['spread_new'], 2), round(o['spread_old'], 2),
              'noR', round(o['spread_new_without_rindnet'], 2), round(o['spread_old_without_rindnet'], 2))
    print('common', {x: round(y, 3) for x, y in out['common']['all'].items()}, 'noR', {x: round(y, 3) for x, y in out['common']['without_rindnet'].items()}, out['common']['per_seed_ci_excludes_zero'])
    f = out['fixed_c500']
    print('fixed c500 cells', {x: round(y, 3) for x, y in f['all_cells'].items()}, 'per-model', {x: round(y, 3) for x, y in f['per_model'].items()}, 'ci below/above', f['n_ci_below_zero'], f['n_ci_above_zero'])
    print('  per model', {m: round(v, 2) for m, v in sorted(pm.items(), key=lambda t: t[1])})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
