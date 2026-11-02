#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CACAIE 一审 B2：把“母图/源照片重叠”与“测试集构成”拆开的划分。

R1-M4 与 R2-2 指出：稿件 3.1 节的两组划分对照在改变分配单位的同时也换掉了测试集
（Crack500 两臂只共享 674 个测试裁块中的 141 个，CrackMap 只共享 24 张中的 3 张），
所以分数差混合了“训练集是否见过测试母图”和“测试集换了一批图”两件事。本脚本生成两组
测试集固定的划分，让两臂之间唯一的差别是训练集与测试集是否共享源照片。

Crack500（每个分组种子 g 一套，splits_rev1_iso_g<g>/）
  从母图互斥划分 splits_mother/ 出发：
  - 108 个测试母图的裁块逐母图随机对半：T_a 留作**两臂共用的固定测试集**，T_b 作
    “泄漏供体”。只有一个裁块的母图整块留在 T_a，否则那张母图在测试集里不存在，
    谈不上泄漏。
  - 验证集 V（62 个母图，与训练、测试母图都不相交）作“干净供体”。
  - 从训练集随机取出 k 个裁块 R，k = min(|T_b|, |V|)，较大的供体集随机下采样到 k。
  - train_leak  = train − R + T_b   （新增 k 个裁块，来自测试母图）
    train_clean = train − R + V     （新增 k 个裁块，来自与测试无关的新母图）
  两臂训练集一样大（2,357，奇偶与母图划分相同，沿用其各模型的批次设置），都新增 k 个
  “新母图”的裁块，区别只在这些母图是否也出现在测试集里。这样控制住了“训练多见了
  一批新场景”这个混杂：如果干净臂用原训练集、泄漏臂换入 T_b，泄漏臂会同时多出场景
  多样性，差值就分不清是泄漏还是多样性。
  - val = R：只用于训练日志（结果一律取第 50 轮，不按验证集选点），R 出自训练母图，
    它的验证分数不代表泛化，诊断时不要引用。

CrackMap（splits_rev1_cv_{group,random}_s<s>/fold<k>/）
  CrackMap 只有 120 张图，固定 24 张的测试集太小，只能交叉验证：
  - group 臂：54 个源照片用 LPT 贪心分成 5 折，每折约 24 张，源照片不跨折。
  - random 臂：120 张图随机打散后切 5 折，每折 24 张（源照片会跨折 = 泄漏）。
  每折 test = 该折，train = 其余 4 折（约 96 张）。两臂的测试集合并后都是**同一组
  120 张图**，测试集构成因此完全相同，训练集大小相同（96 张）。
  - val = 训练集中按文件名排序的前 12 张，同样只用于日志。

输出全部是新目录，不改任何已有划分。完全确定：同一份输入、同一组种子，输出逐字节相同。
"""
import argparse
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

C500_PARENT_RE = re.compile(r'^(.*?)_(\d+)_(\d+)(?:[ _]\(\d+\))?$')
CMAP_PARENT_RE = re.compile(r'^(GOPR\d+)(?:_\(\d+\))?$')


def stem(line):
    return Path(line.split()[0]).stem


def c500_parent(line):
    m = C500_PARENT_RE.match(stem(line))
    if not m:
        raise ValueError(f'无法解析 Crack500 母图: {line}')
    return m.group(1)


def cmap_parent(line):
    m = CMAP_PARENT_RE.match(stem(line))
    if not m:
        raise ValueError(f'无法解析 CrackMap 源照片: {line}')
    return m.group(1)


def read(p):
    return [ln.strip() for ln in p.read_text(encoding='utf-8').splitlines() if ln.strip()]


def write(p, lines):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')


def group(lines, key):
    g = defaultdict(list)
    for ln in lines:
        g[key(ln)].append(ln)
    return g


def make_crack500(root, g):
    src = root / 'splits_mother'
    train, val, test = (read(src / f'{ph}.txt') for ph in ('train', 'val', 'test'))
    rng = random.Random(20261030 + g)

    t_a, t_b = [], []
    for parent, crops in sorted(group(test, c500_parent).items()):
        crops = sorted(crops)
        if len(crops) == 1:
            t_a += crops
            continue
        rng.shuffle(crops)
        # 奇数个时多出的那一块随机落在哪一侧，避免 T_a 系统性偏大
        half = len(crops) // 2 + (len(crops) % 2 if rng.random() < 0.5 else 0)
        t_a += crops[:half]
        t_b += crops[half:]

    k = min(len(t_b), len(val))
    donors_leak = sorted(rng.sample(sorted(t_b), k))
    donors_clean = sorted(rng.sample(sorted(val), k))
    removed = set(rng.sample(sorted(train), k))
    kept = [ln for ln in train if ln not in removed]
    train_leak = sorted(kept + donors_leak)
    train_clean = sorted(kept + donors_clean)
    # 没被选作供体的 T_b 裁块不进任何集合：放回测试集会让 T_a 依赖 k，放进训练只有泄漏臂有
    test_fixed = sorted(t_a)

    # ---- 断言：设计意图逐条落实 ----
    test_parents = {c500_parent(x) for x in test_fixed}
    assert len(train_leak) == len(train_clean) == len(train)
    assert not ({c500_parent(x) for x in train_clean} & test_parents), '干净臂与测试母图相交'
    leak_parents = {c500_parent(x) for x in donors_leak}
    assert leak_parents <= test_parents, '泄漏供体的母图不在测试集里'
    assert not (set(train_leak) & set(test_fixed)) and not (set(train_clean) & set(test_fixed))
    assert not ({c500_parent(x) for x in kept} & test_parents)

    out = root / f'splits_rev1_iso_g{g}'
    write(out / 'train_leak.txt', train_leak)
    write(out / 'train_clean.txt', train_clean)
    write(out / 'val.txt', sorted(removed))
    write(out / 'test.txt', test_fixed)
    print(f'Crack500 g={g}: test {len(test_fixed)} 裁块 / {len(test_parents)} 母图; '
          f'k={k}（T_b {len(t_b)}, V {len(val)}）; 泄漏供体覆盖 {len(leak_parents)} 个测试母图; '
          f'train {len(train_leak)} -> {out}')


def make_crackmap(root, s, n_folds=5):
    items = sorted(sum((read(root / 'splits_parent' / f'{ph}.txt')
                        for ph in ('train', 'val', 'test')), []))
    assert len(items) == len(set(items)) == 120, len(items)
    groups = group(items, cmap_parent)

    # group 臂：分组种子 s 通过打乱同大小源照片的先后影响 LPT 的平局
    rng = random.Random(20261031 + s)
    order = sorted(groups)
    rng.shuffle(order)
    rank = {p: i for i, p in enumerate(order)}
    filled = [0] * n_folds
    g_assign = {}
    for parent in sorted(groups, key=lambda p: (-len(groups[p]), rank[p])):
        f = min(range(n_folds), key=lambda i: (filled[i], i))
        g_assign[parent] = f
        filled[f] += len(groups[parent])
    folds_group = [sorted(x for x in items if g_assign[cmap_parent(x)] == f) for f in range(n_folds)]

    shuffled = items[:]
    rng.shuffle(shuffled)
    size = len(items) // n_folds
    folds_random = [sorted(shuffled[f * size:(f + 1) * size]) for f in range(n_folds)]

    for arm, folds in (('group', folds_group), ('random', folds_random)):
        assert sorted(sum(folds, [])) == items, f'{arm} 折没有恰好覆盖 120 张'
        leaked = 0
        for f, test in enumerate(folds):
            train = sorted(x for x in items if x not in set(test))
            tp = {cmap_parent(x) for x in test}
            n_leak = sum(cmap_parent(x) in {cmap_parent(y) for y in train} for x in test)
            leaked += n_leak
            if arm == 'group':
                assert n_leak == 0, f'group 臂第 {f} 折有源照片跨折'
            out = root / f'splits_rev1_cv_{arm}_s{s}' / f'fold{f}'
            write(out / 'train.txt', train)
            write(out / 'val.txt', train[:12])
            write(out / 'test.txt', test)
        print(f'CrackMap {arm} s={s}: 折大小 {[len(x) for x in folds]}; '
              f'测试图中源照片也在该折训练集里的共 {leaked}/120')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--crack500-root', type=Path, default=Path('data/Crack500/dataset'))
    ap.add_argument('--crackmap-root', type=Path, default=Path('data/CrackMap/dataset'))
    ap.add_argument('--groups', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--cv-seeds', type=int, nargs='+', default=[0])
    a = ap.parse_args()
    for g in a.groups:
        make_crack500(a.crack500_root, g)
    for s in a.cv_seeds:
        make_crackmap(a.crackmap_root, s)
    return 0


if __name__ == '__main__':
    sys.exit(main())
