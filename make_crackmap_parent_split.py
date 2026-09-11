#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# VENDORED VERBATIM - do not rewrite.
#
# Source: https://github.com/hrgentry/tangoseg
#         code/projects/mixercseg_baselines/tools/make_crackmap_parent_split.py
#         (last modified in commit 7830730)
#
# Published unchanged so that the partition it regenerates is byte-identical to
# splits/crackmap_source_disjoint/ in this repository. Grouping is by the GoPro
# source-file prefix of each crop name (GOPR0315_(3).png -> GOPR0315), and the
# longest-processing-time assignment breaks ties by RATIO in descending order
# (train > test > val); any "equivalent" rewrite that breaks ties differently
# yields a different partition and so fails to reproduce the published indices.
#
"""为 CrackMap 生成源照片互斥的 train/val/test 划分。

背景 (2026-09-11 对抗性审稿 T0-1): CrackMap 发行包只有 images/ + masks/ 各 120 张,
无官方划分; 本仓的 84/12/24 是 tools/build_splits.py 图像级随机回退分支
(random.Random(42).shuffle) 的产物。而文件名 `GOPR0315_(3).png` 的前缀是 GoPro
源文件号、括号里是同一源的第几个裁块 —— 按前缀分组: 54 个源文件, 18 个跨 split,
**测试集 24 张里 15 张 (62.5%) 的源文件也出现在训练集** (GOPR0315 的 10 张拆成
6 train / 1 val / 3 test)。这正是稿件 §2.1 论证过的"哈希查不出、只能按源照片
分组查出"的泄漏, 而 Table 1 写的是 "No overlap"。

本脚本与 make_crack500_mother_split.py 同一规则, 刻意保持一致以便稿件把两者写成
同一个控制:
- 分配单位从图像改为源文件 (GOPR 前缀);
- 图像集合完全相同 (120 张不增不减), 比例仍按 7:1:2;
- 不移动任何文件, 新 split 跨 train_img/ val_img/ test_img/ 引用原位文件
  (DeepCrackPairDataset 直接拼 data_root + 相对路径);
- LPT 贪心: 源文件按裁块数降序, 逐个投给当前最欠额的 split; 平局按 RATIO 降序
  (train > test > val) 破。完全确定, 不依赖种子。

与 Crack500 的一处差别: 54 个源文件里最大的贡献 10 张 (GOPR0315), 占 120 的 8%,
所以三个 split 的实际比例会偏离 7:1:2 更多; 脚本打印实际比例并断言互斥。

输出 splits_parent/{train,val,test}.txt, 不动原 splits/。
"""
import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

# `GOPR0315_(3).png` / `GOPR0104_(1).png`: 前缀 GOPRxxxx 是源文件, `_(n)` 是裁块序。
PARENT_RE = re.compile(r'^(GOPR\d+)(?:_\(\d+\))?$')
RATIO = dict(train=0.70, val=0.10, test=0.20)
PHASES = ('train', 'val', 'test')


def parent_of(stem):
    m = PARENT_RE.match(stem)
    if not m:
        raise ValueError(f'无法从 {stem} 解析源文件号: 不是 GOPRxxxx[_(n)] 形式')
    return m.group(1)


def collect(root):
    """扫描三个物理目录, 返回 {parent: [(img_rel, lab_rel), ...]}。"""
    groups = defaultdict(list)
    n = 0
    # 本机 CrackMap 的 120 张全在 train_img/ 下 (build_splits 的随机回退分支只写索引、
    # 不搬文件), val_img/ test_img/ 不存在; 三个目录哪个有就扫哪个。
    for ph in PHASES:
        img_dir = root / f'{ph}_img'
        lab_dir = root / f'{ph}_lab'
        if not img_dir.is_dir():
            continue
        for img in sorted(img_dir.iterdir()):
            if not img.is_file() or img.suffix.lower() not in {'.png', '.jpg', '.jpeg'}:
                continue
            lab = lab_dir / (img.stem + '.png')
            if not lab.is_file():
                raise SystemExit(f'{img} 没有对应掩码 {lab}')
            groups[parent_of(img.stem)].append(
                (f'{img_dir.name}/{img.name}', f'{lab_dir.name}/{lab.name}'))
            n += 1
    return groups, n


def lpt_assign(groups, total):
    """最长处理时间优先: 大源文件先投, 投给 (目标 - 当前)/目标 最欠额的 split。"""
    target = {ph: RATIO[ph] * total for ph in PHASES}
    filled = {ph: 0 for ph in PHASES}
    assign = {}
    # 大组优先; 同大小按源文件号排序, 使输出完全确定
    for parent in sorted(groups, key=lambda p: (-len(groups[p]), p)):
        # 欠额比例最大者; 平局按 RATIO 降序 (train > test > val)
        ph = max(PHASES, key=lambda x: ((target[x] - filled[x]) / target[x], RATIO[x]))
        assign[parent] = ph
        filled[ph] += len(groups[parent])
    return assign, filled


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=Path('data/CrackMap/dataset'))
    ap.add_argument('--out', default='splits_parent')
    args = ap.parse_args()
    root = args.root.resolve()
    groups, total = collect(root)
    assign, filled = lpt_assign(groups, total)

    lines = {ph: [] for ph in PHASES}
    for parent in sorted(groups):
        for img_rel, lab_rel in groups[parent]:
            lines[assign[parent]].append(f'{img_rel} {lab_rel}')

    # 断言: 源文件三两互斥, 图像一张不丢
    parents_by = {ph: {p for p, a in assign.items() if a == ph} for ph in PHASES}
    for a in PHASES:
        for b in PHASES:
            if a < b:
                assert not (parents_by[a] & parents_by[b]), f'{a}/{b} 源文件重叠'
    assert sum(len(v) for v in lines.values()) == total

    out_dir = root / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    for ph in PHASES:
        (out_dir / f'{ph}.txt').write_text('\n'.join(lines[ph]) + '\n',
                                          encoding='utf-8', newline='\n')
    print(f'图像 {total} 张, 源文件 {len(groups)} 个 (最大 {max(len(v) for v in groups.values())} 张)')
    for ph in PHASES:
        print(f'  {ph:5s} {len(lines[ph]):3d} 张 ({100 * len(lines[ph]) / total:4.1f}%), '
              f'{len(parents_by[ph]):2d} 个源文件  [目标 {100 * RATIO[ph]:.0f}%]')
    print(f'-> {out_dir}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
