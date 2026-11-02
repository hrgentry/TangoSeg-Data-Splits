#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""TANGO 匹配对照的几何指标（CACAIE 一审 A8; R1-m2 的核心）。

R1-m2 要求在评估方向、传播、拓扑这类裂缝先验时补连续性与中心线指标。这里对 Table 6 的全部
运行 (参照 + 4 个先验开关 + IN-1k, 种子 42–46; DeepCrack 与母图互斥 Crack500) 跑
eval_rev1_geometry.py, 阈值同样是独立阈值 (Crack500 用验证集, DeepCrack 用测试集对半交叉拟合)。
checkpoint 清单直接取自 rev1_eval_queue.py 的 jobs(), 与直方图评测用的是同一批文件。
"""
import importlib.util
import subprocess
import sys
from pathlib import Path

R = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('q', R / 'projects/mixercseg_baselines/rev1_eval_queue.py')
Q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(Q)

DS = {'deepcrack': ['--data-root', 'data/DeepCrack/dataset', '--test-split', 'splits/test.txt'],
      'crack500_mother': ['--data-root', 'data/Crack500/dataset', '--test-split', 'splits_mother/test.txt',
                          '--val-split', 'splits_mother/val.txt']}
for ds, extra in DS.items():
    models = []
    for fam, d, _val, label, cfg, ckpt, _m in Q.jobs():
        if fam == 'tango' and d == ds and not label.startswith('tango_5ep') and ckpt.exists():
            models += ['--model', label, cfg, str(ckpt)]
    print(ds, len(models) // 4, 'runs', flush=True)
    cmd = [sys.executable, str(R / 'projects/mixercseg_baselines/eval_rev1_geometry.py'), '--dataset', ds,
           '--family', 'tango', '--out-dir', str(R / 'work_dirs/rev1/geometry_tango' / ds)] + extra + models
    subprocess.call(cmd, cwd=str(R))
print('TANGO_GEOMETRY_DONE')
