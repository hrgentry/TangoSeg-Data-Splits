#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""由 eval_rev1_bins.py 的逐图逐 bin 直方图计算一审要求的统计量（纯 CPU，不推理）。

每个模型一份 npz：pos[N, B+1]、neg[N, B+1]，分别是第 i 张图落在第 b 个概率 bin 里的
前景、背景像素数。阈值 bin b 的预测 = {bin >= b}，与 eval_crack_v5_domain_metrics.evaluate
完全相同，所以这里的 ODS 与正文逐位一致。

子命令
  table  一组模型（同一测试划分）的主表数字：
         - pixel-ODS 与 t*（测试集选阈值，正文现行口径，作上界）；
         - AP（与阈值规则无关）；
         - held-out：--cal 给出标定划分（验证集）时，在标定集上选 t*、冻结、报测试 F1/IoU
           （R2-1 的主数字）；
         - cross-fit：--halves R 时把测试集按组随机对半 R 次，各半用另一半选的阈值，两半
           计数合并后算 F1；报 R 次的均值与 SD（DeepCrack 无独立验证集，R2-1）；
         - bootstrap：--boot N 时按组（母图/源照片）有放回重抽样，分别给出
           (a) 固定全集 t* 的 F1 区间（正文现行口径）与
           (b) 嵌套口径：每个重抽样里重新选阈值的 ODS 区间（R2-1、R2-5）；
           以及相邻名次两两之差的嵌套区间。
  cv     B2 CrackMap 5 折：每个臂给出 5 个折目录，同名模型的计数逐折拼接成 120 张图，
         报合并 pixel-ODS、cross-fit（每折用其余 4 折合并曲线选的阈值）F1 与 AP；两臂
         同时给出时另报逐模型差值（--arm 顺序：前减后）。
  paired 两个目录（同一组测试图，例如 B2 Crack500 的泄漏臂与干净臂）逐模型配对：差值的
         嵌套 bootstrap 区间。

分组：crack500（<日期>_<时刻>_x_y → 母图）、crackmap（GOPRxxxx_(n) → 源照片）、none
（逐图，DeepCrack 与 CamCrack789）。与 build_ops_eval.py 的 PARENT_OF 相同。
随机种子固定，区间可复算。
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

SEED = 20261030


# ---------------- 读取与基本量 ----------------
def load(path):
    z = np.load(path)
    return dict(pos=z['pos'].astype(np.int64), neg=z['neg'].astype(np.int64),
                names=[str(x) for x in z['names']], meta=json.loads(str(z['meta'])))


def load_dir(d):
    return {p.stem: load(p) for p in sorted(Path(d).glob('*.npz'))}


def curve(ps, ns):
    """ps, ns: [..., B+1] 的计数和 → 各阈值 bin 的 (F1, P, R, tp, fp, fn)。"""
    tp = np.cumsum(ps[..., ::-1], -1)[..., ::-1]
    fp = np.cumsum(ns[..., ::-1], -1)[..., ::-1]
    tot = ps.sum(-1, keepdims=True)
    P = tp / np.maximum(tp + fp, 1)
    R = tp / np.maximum(tot, 1)
    F1 = 2 * P * R / np.maximum(P + R, 1e-9)
    return F1, P, R, tp, fp, tot - tp


def ods(ps, ns):
    F1 = curve(ps, ns)[0]
    b = int(np.argmax(F1))
    return 100 * float(F1[b]), b


def at_bin(ps, ns, b):
    F1, P, R, tp, fp, fn = curve(ps, ns)
    return dict(F1=100 * float(F1[b]), P=100 * float(P[b]), R=100 * float(R[b]),
                IoU=100 * float(tp[b] / max(tp[b] + fp[b] + fn[b], 1)))


def average_precision(ps, ns):
    """全点插值 AP，与 eval_crack_v5_domain_metrics.average_precision 相同。"""
    _, P, R, *_ = curve(ps, ns)
    order = np.argsort(R, kind='stable')
    r = R[order]
    p = np.maximum.accumulate(P[order][::-1])[::-1]
    return 100 * float(np.sum(np.diff(np.concatenate([[0.0], r])) * p))


# ---------------- 分组 ----------------
GROUP_RULES = {
    'crack500': lambda s: re.match(r'^(.*?)_(\d+)_(\d+)(?:[ _]\(\d+\))?$', Path(s).stem).group(1),
    'crackmap': lambda s: re.match(r'^(GOPR\d+)', Path(s).stem).group(1),
    'none': lambda s: s,
}


def group_index(names, rule):
    keys = [GROUP_RULES[rule](n) for n in names]
    uniq = sorted(set(keys))
    pos = {k: i for i, k in enumerate(uniq)}
    return np.array([pos[k] for k in keys]), len(uniq)


def group_sums(x, gidx, ng):
    out = np.zeros((ng, x.shape[1]), np.int64)
    np.add.at(out, gidx, x)
    return out


def boot_weights(ng, n_boot, rng):
    """每个重抽样里每个组被抽中的次数（多项分布），[n_boot, ng]。"""
    return rng.multinomial(ng, np.full(ng, 1.0 / ng), size=n_boot).astype(np.float64)


def ods_rows(PS, NS):
    """PS, NS: [n, B+1] 每行一个重抽样的合并计数 → 每行的 (ODS, 最优 bin)。"""
    F1 = curve(PS, NS)[0]
    b = F1.argmax(1)
    return 100 * F1[np.arange(len(b)), b], b


def ci(x):
    lo, hi = np.percentile(x, [2.5, 97.5])
    return [round(float(lo), 3), round(float(hi), 3)]


# ---------------- table ----------------
def check_same_images(models):
    names = None
    for k, m in models.items():
        if names is None:
            names = m['names']
        elif m['names'] != names:
            raise SystemExit(f'{k} 的图像列表与其他模型不同，不能放进同一张表')
    return names


def cmd_table(a):
    test = load_dir(a.test)
    names = check_same_images(test)
    cal = load_dir(a.cal) if a.cal else {}
    gidx, ng = group_index(names, a.groups)
    rng = np.random.default_rng(SEED)
    W = boot_weights(ng, a.boot, rng) if a.boot else None
    halves = []
    if a.halves:
        hr = np.random.default_rng(SEED + 1)
        for _ in range(a.halves):
            perm = hr.permutation(ng)
            ga = np.zeros(ng, bool)
            ga[perm[: ng // 2]] = True
            halves.append(ga[gidx])

    out = dict(test=str(a.test), cal=str(a.cal) if a.cal else None, groups=a.groups,
               num_images=len(names), num_groups=ng, boot=a.boot, halves=a.halves,
               seed=SEED, models={})
    boot_ods = {}
    for k, m in test.items():
        ps, ns = m['pos'].sum(0), m['neg'].sum(0)
        o, b = ods(ps, ns)
        r = dict(ODS=round(o, 3), ODS_thr=b / 1000, AP=round(average_precision(ps, ns), 3),
                 IoU_at_ODS=round(at_bin(ps, ns, b)['IoU'], 3),
                 F1_at_0p5=round(at_bin(ps, ns, 500)['F1'], 3),
                 IoU_at_0p5=round(at_bin(ps, ns, 500)['IoU'], 3))
        if k in cal:
            _, bc = ods(cal[k]['pos'].sum(0), cal[k]['neg'].sum(0))
            h = at_bin(ps, ns, bc)
            r.update(heldout_thr=bc / 1000, heldout_F1=round(h['F1'], 3),
                     heldout_IoU=round(h['IoU'], 3), heldout_penalty=round(o - h['F1'], 3))
        if halves:
            f, iou = [], []
            for mask in halves:
                pa, na = m['pos'][mask].sum(0), m['neg'][mask].sum(0)
                pb, nb = m['pos'][~mask].sum(0), m['neg'][~mask].sum(0)
                _, ba = ods(pa, na)
                _, bb = ods(pb, nb)
                # A 半用 B 半的阈值、B 半用 A 半的阈值，两半计数相加后算 F1 与前景 IoU
                _, _, _, tpa, fpa, fna = curve(pa, na)
                _, _, _, tpb, fpb, fnb = curve(pb, nb)
                tp, fp, fn = tpa[bb] + tpb[ba], fpa[bb] + fpb[ba], fna[bb] + fnb[ba]
                f.append(200 * tp / max(2 * tp + fp + fn, 1))
                iou.append(100 * tp / max(tp + fp + fn, 1))
            r.update(crossfit_F1_mean=round(float(np.mean(f)), 3),
                     crossfit_F1_sd=round(float(np.std(f, ddof=1)), 3) if len(f) > 1 else None,
                     crossfit_IoU_mean=round(float(np.mean(iou)), 3))
        if W is not None:
            GP, GN = group_sums(m['pos'], gidx, ng), group_sums(m['neg'], gidx, ng)
            PS, NS = W @ GP, W @ GN
            nested, _ = ods_rows(PS, NS)
            tp = PS[:, b:].sum(1)
            fp = NS[:, b:].sum(1)
            fn = PS.sum(1) - tp
            fixed = 200 * tp / np.maximum(2 * tp + fp + fn, 1)
            boot_ods[k] = nested
            r.update(boot_fixed_thr_F1_ci=ci(fixed),
                     boot_fixed_thr_halfwidth=round(float(np.diff(ci(fixed))[0] / 2), 3),
                     boot_nested_ODS_ci=ci(nested),
                     boot_nested_halfwidth=round(float(np.diff(ci(nested))[0] / 2), 3))
        out['models'][k] = r

    if boot_ods:
        order = sorted(out['models'], key=lambda k: -out['models'][k]['ODS'])
        adj = []
        for hi_, lo_ in zip(order, order[1:]):
            d = boot_ods[hi_] - boot_ods[lo_]
            c = ci(d)
            adj.append(dict(pair=[hi_, lo_],
                            diff=round(out['models'][hi_]['ODS'] - out['models'][lo_]['ODS'], 3),
                            nested_ci=c, separated=bool(c[0] > 0)))
        out['adjacent_pairs_nested'] = adj
        # 全部模型对 (不止相邻): 补充材料 S8 报“多宽的差能分开”, 用同一组嵌套重抽样
        ks = list(boot_ods)
        sep = sum(1 for i in range(len(ks)) for j in range(i + 1, len(ks))
                  if (lambda c: c[0] > 0 or c[1] < 0)(ci(boot_ods[ks[i]] - boot_ods[ks[j]])))
        out['all_pairs_nested'] = dict(n_pairs=len(ks) * (len(ks) - 1) // 2, n_separated=sep)
    return out


# ---------------- cv ----------------
def cmd_cv(a):
    arms = {}
    for spec in a.arm:
        name, dirs = spec.split('=', 1)
        folds = [load_dir(d) for d in dirs.split(',')]
        labels = sorted(set.intersection(*[set(f) for f in folds]))
        res = {}
        for k in labels:
            P = [f[k]['pos'] for f in folds]
            N = [f[k]['neg'] for f in folds]
            ps = sum(p.sum(0) for p in P)
            ns = sum(n.sum(0) for n in N)
            o, b = ods(ps, ns)
            tp = fp = fn = 0
            for i in range(len(folds)):
                # 第 i 折用其余各折合并曲线选的阈值（其余折是不同模型的样本外预测）
                _, bi = ods(ps - P[i].sum(0), ns - N[i].sum(0))
                _, _, _, t, f_, n_ = curve(P[i].sum(0), N[i].sum(0))
                tp, fp, fn = tp + t[bi], fp + f_[bi], fn + n_[bi]
            res[k] = dict(pooled_ODS=round(o, 3), pooled_thr=b / 1000,
                          crossfit_F1=round(float(200 * tp / max(2 * tp + fp + fn, 1)), 3),
                          AP=round(average_precision(ps, ns), 3),
                          num_images=int(sum(len(p) for p in P)))
        names = sorted(sum((f[labels[0]]['names'] for f in folds), [])) if labels else []
        arms[name] = dict(folds=dirs.split(','), models=res, images=names)
    out = dict(arms={k: dict(folds=v['folds'], models=v['models']) for k, v in arms.items()})
    if len(arms) == 2:
        (n1, A1), (n2, A2) = arms.items()
        if A1['images'] != A2['images']:
            raise SystemExit('两臂合并后的测试图不同，不满足固定测试集的设计')
        common = sorted(set(A1['models']) & set(A2['models']))
        diffs = {k: dict(pooled_ODS=round(A1['models'][k]['pooled_ODS'] - A2['models'][k]['pooled_ODS'], 3),
                         crossfit_F1=round(A1['models'][k]['crossfit_F1'] - A2['models'][k]['crossfit_F1'], 3))
                 for k in common}
        v = [d['pooled_ODS'] for d in diffs.values()]
        out['diff'] = dict(order=f'{n1} - {n2}', models=diffs,
                           median=round(float(np.median(v)), 3) if v else None,
                           mean=round(float(np.mean(v)), 3) if v else None,
                           n_positive=int(sum(x > 0 for x in v)), n=len(v))
    return out


# ---------------- paired ----------------
def cmd_paired(a):
    A, B = load_dir(a.a), load_dir(a.b)
    common = sorted(set(A) & set(B))
    names = A[common[0]]['names']
    for k in common:
        if A[k]['names'] != names or B[k]['names'] != names:
            raise SystemExit(f'{k}: 两臂测试图不同')
    gidx, ng = group_index(names, a.groups)
    W = boot_weights(ng, a.boot, np.random.default_rng(SEED))
    out = dict(a=str(a.a), b=str(a.b), order='a - b', groups=a.groups, boot=a.boot,
               num_images=len(names), num_groups=ng, seed=SEED, models={})
    for k in common:
        oa, _ = ods(A[k]['pos'].sum(0), A[k]['neg'].sum(0))
        ob, _ = ods(B[k]['pos'].sum(0), B[k]['neg'].sum(0))
        da, _ = ods_rows(W @ group_sums(A[k]['pos'], gidx, ng), W @ group_sums(A[k]['neg'], gidx, ng))
        db, _ = ods_rows(W @ group_sums(B[k]['pos'], gidx, ng), W @ group_sums(B[k]['neg'], gidx, ng))
        out['models'][k] = dict(ODS_a=round(oa, 3), ODS_b=round(ob, 3), diff=round(oa - ob, 3),
                                nested_ci=ci(da - db))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest='cmd', required=True)
    t = sp.add_parser('table')
    t.add_argument('--test', required=True)
    t.add_argument('--cal')
    t.add_argument('--groups', default='none', choices=list(GROUP_RULES))
    t.add_argument('--halves', type=int, default=0)
    t.add_argument('--boot', type=int, default=0)
    c = sp.add_parser('cv')
    c.add_argument('--arm', action='append', required=True,
                   help='name=fold0_dir,fold1_dir,...（可给两次）')
    p = sp.add_parser('paired')
    p.add_argument('--a', required=True)
    p.add_argument('--b', required=True)
    p.add_argument('--groups', default='none', choices=list(GROUP_RULES))
    p.add_argument('--boot', type=int, default=10000)
    for q in (t, c, p):
        q.add_argument('--out', required=True)
    a = ap.parse_args()
    out = dict(table=cmd_table, cv=cmd_cv, paired=cmd_paired)[a.cmd](a)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding='utf-8')
    print(f'-> {a.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
