#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Describe 5-to-50-epoch gains by the matrix's initialization labels.

The stratification compares heterogeneous model groups, not randomized or
within-architecture initialization interventions. Pairwise counts reuse the
same models and are descriptive, not independent matched experiments.

Portable use with the published archive:
  python build_budget_by_init.py --budget-matrix benchmarks/budget_matrix_5_vs_50.json --params benchmarks/params_flops_512_merged.json --out budget_by_init.json

Without --budget-matrix, read the original per-budget evaluation files from
--bench-dir. Both paths must produce the same numerical summaries.
"""
import argparse
import json
from statistics import median
import sys
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / 'benchmarks'
OUT = BENCH / 'budget_by_init.json'
# 均值口径与 Table 4 一致：用分布式划分的 Crack500，不用母图划分，
# 这样 15 个配置的四集均值可以直接和表里的 Mean 列对上。
MEAN_DS = ['deepcrack', 'crack500', 'camcrack789', 'crackmap']
DATASETS = MEAN_DS + ['crack500_mother']
# Table 4 的 Init 列。None 之外的一切（IN-1k / IN-21k / ADE20K）都算载入了预训练。
INIT = {
    'convnext_xlarge': 'IN-21k', 'segformer_mit_b5': 'ADE20K',
    'hrnet_w48': 'ADE20K', 'deeplabv3plus_r50': 'ADE20K',
    'rindnet': 'IN-1k', 'dtrcnet': 'IN-1k',
    'unet': 'None', 'shuttlenet_v2': 'None', 'carnet': 'None',
    'tangoseg': 'None', 'deepcrack_fast_fuse': 'None', 'restormixer': 'None',
    'mixercseg_improved': 'None', 'mixercseg_faithful': 'None',
    'mambavision': 'None',
}
# 5 轮时被摁在全背景解上的三格（稿件 sec:budget 已逐个点名）。分层若只靠它们撑着
# 就没有意义，所以必须做剔除后的敏感性检查。
COLLAPSED_AT_5EP = ['mixercseg_faithful', 'deepcrack_fast_fuse', 'carnet']
CAVEATS = [
    'One seed per cell: group differences include seed variability and are not '
    'matched contrasts. The repeated-run evidence is the three-architecture, '
    'three-seed budget comparison on DeepCrack.',
    'The groups are not randomly assigned and also differ in model family and '
    'scale. Counted GFLOPs include lower-bound entries. U-Net illustrates the '
    'pattern but cannot isolate initialization from capacity.',
    'Six models against nine: this describes the matrix, not a pretraining '
    'effect. The matched TANGO initialization contrasts answer a different '
    'question and do not identify a causal component of this group gap.',
    'The 54 cross-group comparisons reuse 15 models; they are not 54 independent '
    'runs or matched experimental pairs.',
]


def pair_wins(a, b):
    """(a 组, b 组) 的两两配对中 a 更大的对数；并列各计半对。Mann-Whitney 的计数形式。

    用配对数而不是 AUC 报数，是因为 n=6/9 时 "54 对里 52 对" 读者可以自己核，
    0.96 这个小数点会让人以为精度比实际高。
    """
    w = sum((x > y) + 0.5 * (x == y) for x, y in product(a, b))
    return w, len(a) * len(b)


def summarize(gains, models):
    pre = [gains[m] for m in models if INIT[m] != 'None']
    scr = [gains[m] for m in models if INIT[m] == 'None']
    wins, total = pair_wins(scr, pre)
    # 汇总量存四位小数, 不是别处惯用的三位: 稿件按两位显示, 而 DeepCrack 那一格的
    # 组间差是 0.30516..., 三位会写成 0.305, 正好卡在两位的进位边界上无法判读。
    return {
        'pretrained': {'n': len(pre), 'mean': round(sum(pre) / len(pre), 4),
                       'min': round(min(pre), 4), 'max': round(max(pre), 4)},
        'scratch': {'n': len(scr), 'mean': round(sum(scr) / len(scr), 4),
                    'min': round(min(scr), 4), 'max': round(max(scr), 4)},
        'mean_difference': round(sum(scr) / len(scr) - sum(pre) / len(pre), 4),
        'pairs_favouring_scratch': wins,
        'pairs_total': total,
        'groups_disjoint': min(scr) > max(pre),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--budget-matrix', type=Path)
    ap.add_argument('--params', type=Path)
    ap.add_argument('--bench-dir', type=Path, default=BENCH)
    ap.add_argument('--out', type=Path, default=OUT)
    args = ap.parse_args()
    d5, d50 = {}, {}
    if args.budget_matrix:
        matrix = json.loads(args.budget_matrix.read_text(encoding='utf-8'))
        for ds in DATASETS:
            row = matrix['datasets'][ds]
            for budget, target in [(5, d5), (50, d50)]:
                target[ds] = {'num_eval': row['num_eval'], 'split': row['split']}
                target[ds].update({m: {'ODS': v[f'ODS_{budget}']}
                                   for m, v in row['models'].items()})
    else:
        for ds in DATASETS:
            p5 = args.bench_dir / f'public_ods_final_{ds}.json'
            p50 = args.bench_dir / f'public_ods_final50_{ds}.json'
            if not p5.exists() or not p50.exists():
                raise SystemExit(f'Missing {p5.name} or {p50.name}')
            d5[ds] = json.loads(p5.read_text(encoding='utf-8'))
            d50[ds] = json.loads(p50.read_text(encoding='utf-8'))
            for key in ('num_eval', 'split'):
                if d5[ds].get(key) != d50[ds].get(key):
                    raise SystemExit(f'{ds}: inconsistent {key}')

    models = sorted(k for k, v in d50[DATASETS[0]].items() if isinstance(v, dict))
    unlabelled = [m for m in models if m not in INIT]
    if unlabelled:
        raise SystemExit(f'Missing initialization labels: {unlabelled}')
    if len(models) != len(INIT):
        raise SystemExit(f'Model roster length {len(models)} differs from labels {len(INIT)}')

    keep = [m for m in models if m not in COLLAPSED_AT_5EP]
    out = {
        'note': ('Whole-matrix 5-vs-50-epoch gains stratified by the '
                 'initialization loaded before crack training, which is the '
                 'Init. column of the comparative model matrix.'),
        'caveats': CAVEATS,
        'init_labels': INIT,
        'collapsed_at_5ep': COLLAPSED_AT_5EP,
        'sources': {ds: {'5': f'public_ods_final_{ds}.json',
                         '50': f'public_ods_final50_{ds}.json'} for ds in DATASETS},
        'per_dataset': {},
        'mean_over': MEAN_DS,
    }

    for ds in DATASETS:
        g = {m: d50[ds][m]['ODS'] - d5[ds][m]['ODS'] for m in models}
        out['per_dataset'][ds] = {
            'split': d50[ds]['split'], 'num_eval': d50[ds]['num_eval'],
            'gains': {m: round(g[m], 3) for m in models},
            'all_models': summarize(g, models),
            'excluding_collapsed': summarize(g, keep),
        }

    gm = {m: sum(d50[ds][m]['ODS'] - d5[ds][m]['ODS'] for ds in MEAN_DS) / len(MEAN_DS)
          for m in models}
    out['four_dataset_mean'] = {
        'gains': {m: round(gm[m], 3) for m in models},
        'all_models': summarize(gm, models),
        'excluding_collapsed': summarize(gm, keep),
    }

    # 组间落差本身随预算变多少 —— 这才是稿件要引的量：短预算把初始化差距放大了几倍。
    # 与 sec:init 的受控对比（同架构 ±IN-1k，3 种子，50 轮，末轮差 +1.30 / +0.12）
    # 同量级的应当是 50 轮那一栏，不是 5 轮那一栏。
    def level(blob):
        v = {m: sum(blob[ds][m]['ODS'] for ds in MEAN_DS) / len(MEAN_DS) for m in models}
        pre = [v[m] for m in models if INIT[m] != 'None']
        scr = [v[m] for m in models if INIT[m] == 'None']
        return {'pretrained_mean': round(sum(pre) / len(pre), 4),
                'scratch_mean': round(sum(scr) / len(scr), 4),
                'gap': round(sum(pre) / len(pre) - sum(scr) / len(scr), 4)}

    lv5, lv50 = level(d5), level(d50)

    # 四集均值的名次位移。稿件要引 U-Net 是全表上移最多的一个, 这里算出来备查;
    # 破平规则与 matrix_stats.py / build_budget_matrix.py 一致 (分数降序, 键名破平)。
    def mean_ranks(blob):
        v = {m: sum(blob[ds][m]['ODS'] for ds in MEAN_DS) / len(MEAN_DS) for m in models}
        return {m: i + 1 for i, m in enumerate(sorted(v, key=lambda m: (-v[m], m)))}

    r5, r50 = mean_ranks(d5), mean_ranks(d50)
    move = {m: r5[m] - r50[m] for m in models}          # 正数 = 上移
    top = max(move, key=lambda m: move[m])
    out['mean_rank_move'] = {
        'per_model': {m: {'rank_5': r5[m], 'rank_50': r50[m], 'move': move[m]}
                      for m in models},
        'largest_rise': {'model': top, 'move': move[top]},
        'is_also_largest_absolute': abs(move[top]) == max(abs(x) for x in move.values()),
    }

    out['group_gap_by_budget'] = {
        '5': lv5, '50': lv50,
        'gap_ratio': round(lv5['gap'] / lv50['gap'], 2),
        'comment': ('The gap between the two initialization groups at each '
                    'budget, in four-dataset mean pixel-ODS. The short budget '
                    'reports a gap several times the one the 50-epoch budget '
                    'reports. This is a cross-model association and does not isolate '
                    'initialization, capacity, model family or convergence speed.'),
    }

    # 混杂检查的数字，供稿件引用：两组的算力中位数，以及 U-Net 比几个预训练模型重。
    params_path = args.params or args.bench_dir / 'params_flops_512_merged.json'
    pf = json.loads(params_path.read_text(encoding='utf-8'))['models']
    pre_g = [pf[m]['gflops'] for m in models if INIT[m] != 'None']
    out['capacity_confound'] = {
        'pretrained_median_gflops': round(median(pre_g), 1),
        'scratch_median_gflops': round(
            median([pf[m]['gflops'] for m in models if INIT[m] == 'None']), 1),
        'unet_gflops': pf['unet']['gflops'],
        'unet_heavier_than_n_pretrained': sum(1 for x in pre_g if x < pf['unet']['gflops']),
        'n_pretrained': len(pre_g),
    }

    if args.budget_matrix:
        out['derived_from'] = {'budget_matrix': args.budget_matrix.name,
                               'params_flops': params_path.name}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, ensure_ascii=False) + '\n',
                   encoding='utf-8', newline='\n')

    for ds in DATASETS + ['four_dataset_mean']:
        row = out['four_dataset_mean'] if ds == 'four_dataset_mean' else out['per_dataset'][ds]
        for tag in ['all_models', 'excluding_collapsed']:
            s = row[tag]
            print(f"{ds} {tag}: pretrained={s['pretrained']['mean']:.4f}; "
                  f"scratch={s['scratch']['mean']:.4f}; "
                  f"difference={s['mean_difference']:.4f}; "
                  f"pairs={s['pairs_favouring_scratch']}/{s['pairs_total']}; "
                  f"ranges_disjoint={s['groups_disjoint']}")
    print('Group gaps:', out['group_gap_by_budget'])
    print('Largest rank rise:', out['mean_rank_move']['largest_rise'])
    print('Counted operation summaries:', out['capacity_confound'])
    print('Output:', args.out)

    return 0


if __name__ == '__main__':
    sys.exit(main())
