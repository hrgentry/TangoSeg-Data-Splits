#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""“取最后一轮”规则在未触碰数据上的检验（CACAIE 一审 R2-1 最后一问）。

稿件把 checkpoint 规则从“验证集 fgIoU@0.5 最优”改成“固定取第 50 轮”，理由是前者在
DeepCrack 上用的验证文件就是测试文件、在短预算下又会退化成取第一轮。R2 要求在未触碰的
数据上评估这条改动后的规则。这里对种子 42 的每个 (数据集, 模型)：
  - 最优 checkpoint（best_fgIoU_epoch_*.pth，按验证集 fgIoU@0.5 选）与末轮 checkpoint
    各自在**验证集**上选阈值，冻结后在**测试集**上算 F1 —— 选点与选阈值都只用验证集，
    测试集从头到尾没参与任何选择；
  - 报两者之差（最优 − 末轮）。正值表示按验证集选点在留出数据上确有收益。
有独立验证集的四列（Crack500 两种划分、CamCrack789、CrackMap 源照片互斥）是检验本身；
DeepCrack 的验证文件即测试文件，选点泄漏了测试集，单独列出、不计入检验。
"""
import argparse
import importlib.util
import json
import statistics
import sys
from pathlib import Path

R = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location('_stats', R / 'projects/mixercseg_baselines/rev1_bins_stats.py')
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)
B = R / 'work_dirs/rev1/bins'
HELD = [('crack500', 'crack500__test', 'crack500__val'),
        ('crack500_mother', 'crack500_mother__test', 'crack500_mother__val'),
        ('camcrack789', 'camcrack789__test', 'camcrack789__val'),
        ('crackmap_parent', 'crackmap_parent__test', 'crackmap_parent__val')]


def heldout(fam, test, val, m):
    t, v = S.load(B / fam / test / f'{m}.npz'), S.load(B / fam / val / f'{m}.npz')
    _, b = S.ods(v['pos'].sum(0), v['neg'].sum(0))
    ps, ns = t['pos'].sum(0), t['neg'].sum(0)
    return S.at_bin(ps, ns, b)['F1'], S.ods(ps, ns)[0], int(json.loads(json.dumps(t['meta']))['ckpt'].split('epoch_')[-1].split('.')[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = dict(note=__doc__.strip().splitlines()[0], seed=42, datasets={})
    for ds, test, val in HELD:
        rows = {}
        for p in sorted((B / 'best50_s42' / test).glob('*.npz')):
            m = p.stem
            if not (B / 'm50_s42' / test / f'{m}.npz').exists():
                continue
            fb, ob, eb = heldout('best50_s42', test, val, m)
            ff, of, _ = heldout('m50_s42', test, val, m)
            # 保留全精度: 稿件报两位小数的中位数, 而偶数个格子的中位数是两个差值的平均, 恰好
            # 落在进位边界上时 (Crack500 母图互斥列即是), 用三位小数的值会进错方向。
            rows[m] = dict(best_epoch=eb, heldout_F1_best=fb, heldout_F1_final=ff,
                           diff=fb - ff, oracle_ODS_best=ob, oracle_ODS_final=of)
        d = [r['diff'] for r in rows.values()]
        same = sum(1 for r in rows.values() if r['best_epoch'] == 50)
        out['datasets'][ds] = dict(models=rows, n=len(d), median_diff=statistics.median(d) if d else None,
                                   mean_diff=round(statistics.mean(d), 3) if d else None,
                                   n_best_better=sum(x > 0 for x in d), n_best_is_final=same,
                                   min_diff=round(min(d), 3) if d else None, max_diff=round(max(d), 3) if d else None)
    allv = [r['diff'] for v in out['datasets'].values() for r in v['models'].values()]
    out['pooled'] = dict(n=len(allv), median_diff=statistics.median(allv),
                         n_best_better=sum(x > 0 for x in allv), n_tie=sum(x == 0 for x in allv))
    Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False) + '\n', encoding='utf-8', newline='\n')
    for ds, v in out['datasets'].items():
        print('%-16s n=%2d median %+.2f mean %+.2f range %+.2f..%+.2f best>final %d best=final(epoch50) %d'
              % (ds, v['n'], v['median_diff'], v['mean_diff'], v['min_diff'], v['max_diff'],
                 v['n_best_better'], v['n_best_is_final']))
    print('pooled', out['pooled'])


if __name__ == '__main__':
    sys.exit(main())
