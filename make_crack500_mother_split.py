#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# VENDORED VERBATIM — do not rewrite.
#
# Source: https://github.com/hrgentry/tangoseg
#         code/projects/mixercseg_baselines/tools/make_crack500_mother_split.py
#         (last modified in commit dba06157)
#
# Published unchanged so that the partition it regenerates is byte-identical to
# splits/crack500_parent_disjoint/ in this repository. The longest-processing-
# time assignment breaks ties by RATIO in descending order (train > test > val);
# any "equivalent" rewrite that breaks ties differently — alphabetically, for
# instance — yields a different partition and so fails to reproduce the
# published indices.
#
"""为 Crack500 生成母图互斥的 train/val/test 划分。

背景:我们用的 Crack500 来自 MixerCSeg 官方 Google Drive 分发包
(data/_downloads/Crack500.zip),其 2357/336/675 的划分是在**裁块**层面随机
分配的。Crack500 的每张 2000x1500 母图被切成 640x360 的网格裁块,文件名
`<YYYYMMDD_HHMMSS>_<x>_<y>.jpg` 里前缀是母图、后两段是裁块左上角坐标。实测
test 的 356 张母图**全部 356 张**也出现在 train——即每个测试裁块都来自模型
训练过的那张照片,只是换了个窗口。绝对精度因此被系统性抬高。

本脚本只改**一个**变量:分配单位从裁块改成母图。刻意保持不变的有:
- 裁块集合完全相同(3368 张,一张不增不减);
- 比例仍为 7:1:2 —— 官方分发包正是 2357/336/675 = 70.0/10.0/20.0%,照抄该
  比例才能把"裁块级 vs 母图级分配"孤立成唯一差异;
- 不移动任何文件。DeepCrackPairDataset 的 data_prefix 为空、路径直接拼
  data_root,所以新 split 可以跨 train_img/ val_img/ test_img/ 引用原位文件。

分配用 LPT(最长处理时间优先)贪心:母图按裁块数降序,逐个投给当前**最欠额**
的 split。选它而不是随机打散,是因为它完全确定(无需记种子即可复现)且比随机
更贴近目标比例;母图大小在三个 split 间天然被摊匀,不会把大母图堆到某一侧。

输出 splits_mother/{train,val,test}.txt,不动原 splits/。
"""
import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

# `<mother>_<x>_<y>[_(n)].<ext>`：末尾两段纯数字是裁块左上角坐标，剥掉即得母图 ID。
# `(n)` 后缀在分发包里有 4 例，且它们的无后缀同名文件位于**别的** split
# （如 train/..._1281_721_(2).jpg 对 test/..._1281_721.jpg），即同一母图同一
# 网格块的两个版本被拆到了两侧；按母图分组会把它们一并收拢，故只需解析掉后缀，
# 不必特殊处理或丢弃。内容哈希全库无重复，所以它们是重编码/重标注版而非副本。
#
# 分隔符两种都认（`_(n)` 与 ` (n)`）：原始分发包用空格，但空格会破坏索引文件的
# `line.split()` 解析——DeepCrackPairDataset 按空白切分，一行会被切成 4 段、路径
# 被截断。.80 上的副本早已把空格改成下划线，本机 2026-07-27 已同步重命名，两机
# 现在一致；正则保留对空格的兼容，只为让脚本在未清理的副本上仍能正确分组。
CROP_RE = re.compile(r'^(.*?)_(\d+)_(\d+)(?:[ _]\(\d+\))?$')
RATIO = dict(train=0.70, val=0.10, test=0.20)
# 物理目录：原分发包把裁块分放在三个目录里，新划分要跨目录引用
PHASES = ('train', 'val', 'test')


def mother_of(stem):
    m = CROP_RE.match(stem)
    if not m:
        raise ValueError(f'无法从 {stem} 解析母图 ID：文件名不含 _<x>_<y> 后缀')
    return m.group(1)


def collect(root):
    """扫描三个物理目录，返回 {mother: [(img_rel, lab_rel), ...]}。"""
    groups = defaultdict(list)
    for ph in PHASES:
        img_dir = root / f'{ph}_img'
        lab_dir = root / f'{ph}_lab'
        if not img_dir.is_dir():
            raise SystemExit(f'缺少目录 {img_dir}')
        for img in sorted(img_dir.iterdir()):
            if not img.is_file():
                continue
            lab = lab_dir / (img.stem + '.png')
            if not lab.is_file():
                raise SystemExit(f'{img.name} 找不到对应标注 {lab}')
            groups[mother_of(img.stem)].append(
                (f'{ph}_img/{img.name}', f'{ph}_lab/{lab.name}'))
    return groups


def assign(groups):
    """LPT 贪心：母图按裁块数降序，逐个投给最欠额的 split。"""
    total = sum(len(v) for v in groups.values())
    target = {k: v * total for k, v in RATIO.items()}
    have = {k: 0 for k in RATIO}
    out = {k: [] for k in RATIO}
    order = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    for mother, crops in order:
        # 缺口最大者优先；平局时按 train>test>val 的固定顺序，保持确定性
        pick = max(RATIO, key=lambda k: (target[k] - have[k], RATIO[k]))
        out[pick].extend(crops)
        have[pick] += len(crops)
    return out, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='data/Crack500/dataset',
                    help='含 {train,val,test}_{img,lab} 的目录')
    ap.add_argument('--out-dir', default='splits_mother',
                    help='相对 --root 的输出目录；原 splits/ 不受影响')
    a = ap.parse_args()

    root = Path(a.root)
    groups = collect(root)
    out, total = assign(groups)

    # 硬校验：母图必须三两互斥，裁块必须一张不丢不重
    seen = {}
    for split, crops in out.items():
        for img, _ in crops:
            mo = mother_of(Path(img).stem)
            if seen.setdefault(mo, split) != split:
                raise SystemExit(f'母图 {mo} 同时落在 {seen[mo]} 与 {split}')
    n_out = sum(len(v) for v in out.values())
    if n_out != total:
        raise SystemExit(f'裁块数不符：输入 {total} 输出 {n_out}')

    # 索引文件是空白分隔的 `<img> <mask>`，DeepCrackPairDataset 用 line.split()
    # 解析。路径里只要有一个空格，该行就被切成 4 段、路径被截断，且报错发生在
    # DataLoader worker 里，只显示被截断的文件名——2026-07-27 因此白等了 8 小时
    # 的排队。写盘前直接拦下，比事后从 worker 栈回溯便宜得多。
    bad = [p for crops in out.values() for pair in crops for p in pair
           if any(c.isspace() for c in p)]
    if bad:
        raise SystemExit(
            f'{len(bad)} 个路径含空白字符，无法写入空白分隔的索引文件；'
            f'请先重命名文件。例：{bad[0]!r}')

    dst = root / a.out_dir
    dst.mkdir(parents=True, exist_ok=True)
    for split, crops in out.items():
        with open(dst / f'{split}.txt', 'w', encoding='utf-8', newline='\n') as f:
            for img, lab in sorted(crops):
                f.write(f'{img} {lab}\n')

    print(f'母图 {len(groups)} 张 / 裁块 {total} 张 -> {dst}')
    for split in PHASES:
        crops = out[split]
        mothers = {mother_of(Path(i).stem) for i, _ in crops}
        print(f'  {split:<6}裁块 {len(crops):>5} ({len(crops)/total:6.2%}, '
              f'目标 {RATIO[split]:.0%})  母图 {len(mothers):>4}')
    print('校验通过：三个 split 母图互斥，裁块无丢失无重复')
    return 0


if __name__ == '__main__':
    sys.exit(main())
