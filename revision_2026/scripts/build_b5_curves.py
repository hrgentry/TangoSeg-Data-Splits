#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""B5 的“更快收敛还是更好终点”（CACAIE 一审 R2-7）。纯 CPU。

R2-7 要求按优化更新次数画精度，区分预训练只是收敛更快、还是终点更好。B5 的 36 个运行
（从零 / ADE20K / IN-1k × 4 个数据集 × 种子 42–44，.228 上同一队列）每 5 轮在验证集上记一次
前景 IoU（阈值 0.5，训练框架自带的验证），由 .228 的 vis_data/scalars.json 抽出存为
work_dirs/rev1/b5_curves/scalars_val.txt（每块以 "### <run>" 开头）。这里对每个数据集、每种
预训练、每个验证轮次，算三种子平均的“预训练 − 从零”差。DeepCrack 的验证文件就是测试文件，
曲线只用来看形状，不用于任何阈值或模型选择。
"""
import argparse
import json
import statistics
from pathlib import Path

R = Path(__file__).resolve().parents[2]
DSS = ['deepcrack', 'crack500_mother', 'camcrack789', 'crackmap_parent']
SEEDS = [42, 43, 44]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', default=str(R / 'work_dirs/rev1/b5_curves/scalars_val.txt'))
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    cur = {}
    for blk in Path(a.src).read_text(encoding='utf-8').split('### ')[1:]:
        lines = blk.strip().splitlines()
        vals = [json.loads(x) for x in lines[1:]]
        cur[lines[0]] = {int(v['step']): v['fgIoU'] for v in vals}
    out = dict(note=__doc__.strip().splitlines()[0], seeds=SEEDS, metric='validation foreground IoU at 0.5',
               curves=cur, gap={})
    for ds in DSS:
        eps = sorted(cur[f'tangoseg_{ds}_s42'])
        out['gap'][ds] = {}
        for arm in ('ade20k', 'in1k'):
            g = {e: statistics.mean(cur[f'tangoseg_{arm}_{ds}_s{s}'][e] - cur[f'tangoseg_{ds}_s{s}'][e] for s in SEEDS)
                 for e in eps}
            out['gap'][ds][arm] = dict(epochs=eps, gap=[g[e] for e in eps], first=g[eps[0]], last=g[eps[-1]])
    Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False) + '\n', encoding='utf-8', newline='\n')
    for ds, v in out['gap'].items():
        print(ds, {k: (round(x['first'], 2), round(x['last'], 2)) for k, x in v.items()})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
