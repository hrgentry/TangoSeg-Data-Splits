#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""裂缝几何指标评测（CACAIE 一审 A8; R1-m2, R2-6）。

R1-m2: 评估方向、传播、拓扑这类裂缝先验, 应补连续性、中心线、拓扑完整性指标, 不能只看
pixel-ODS 与 IoU。R2-6: 去掉 centerline-Dice 后 pixel-ODS 升、中心线 Dice 降, 说明像素分数
支撑不了几何结论; 要求端点与分叉误差、断裂路径、长度与宽度偏差、对接缝/阴影的误检, 并按
裂缝宽度与对比度分层, 全部在独立标定的阈值下。

每个模型在一个测试划分上跑一遍前向（前向、读图、量化与 eval_rev1_bins.py 相同, 来自同一个
指标模块）, 按**独立阈值**二值化: 有验证集的数据集用 work_dirs/rev1/bins 里该模型验证集
直方图选出的 t*; DeepCrack 的验证文件即测试文件, 把测试集按固定种子随机对半, 每半用另一半
选的 t*。然后逐图计算（512x512 的评测分辨率下, 宽度以像素计）:
  clDice                    中心线 Dice（与稿件 Table 6 同一实现）
  len_rel_err               预测骨架长度相对真值骨架长度的误差
  width_bias_px             平均宽度偏差: 预测面积/预测骨架长 − 真值面积/真值骨架长
  frag_excess               断裂: 落在真值裂缝 3 px 走廊内的预测连通块数 − 真值连通块数
  broken_frac / missed_frac 骨架长 ≥20 px 的真值裂缝（8 连通分量）中, 被 ≥2 个预测连通块覆盖
                            (断开) 与完全未被覆盖 (漏检) 的比例; 覆盖指预测落在该分量 3 px 走廊内
  endpoint_excess           剪掉 ≤10 px 的骨架毛刺后, 预测端点数 − 真值端点数（断裂会多出端点）
  junction_excess           同样剪枝后, 预测分叉点数 − 真值分叉点数
                            （不剪枝时真值掩膜较宽, 骨架上的短毛刺主导这两个数, 不代表断裂）
  spurious_count / _area    与真值 5 px 邻域不相交的预测连通块的个数与面积占预测面积的比例
                            （远离任何标注裂缝的误检, 接缝、阴影、纹理都会落在这里; 公开数据集
                            没有接缝或阴影标签, 所以只能报这一类, 不能按成因分）
  recall_by_width           真值骨架像素按局部宽度 (2×到边界距离) 分 <3 / 3–7 / >7 px 三档,
                            每档里被 2 px 内的预测覆盖的比例（逐档汇总像素, 不是逐图平均）
  contrast                  图像级对比度: 真值裂缝外 2–6 px 环带灰度均值减裂缝像素灰度均值,
                            除以环带灰度标准差; 分析时按数据集内三分位分层
输出 work_dirs/rev1/geometry/<数据集>/<label>.json（逐图值 + 汇总）。可反复运行, 已有的跳过。
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy import ndimage
from skimage.morphology import skeletonize

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = _load('_metrics', 'projects/crack_baselines/eval_crack_v5_domain_metrics.py')
P = _load('_public', 'projects/mixercseg_baselines/eval_public_ods.py')
EB = _load('_bins', 'projects/mixercseg_baselines/eval_rev1_bins.py')
ST = _load('_stats', 'projects/mixercseg_baselines/rev1_bins_stats.py')
BINS = ROOT / 'work_dirs/rev1/bins'
WIDTH_EDGES = (3.0, 7.0)
SPUR_PX = 10
MIN_COMP_SKEL = 20
K8 = np.ones((3, 3), np.uint8)


def thresholds(fam, ds, test_split, val_split, label, n_img):
    """每张测试图使用的阈值 bin: 有验证集用验证集 t*; 否则测试集固定对半交叉拟合。"""
    if val_split:
        v = ST.load(BINS / fam / f'{ds}__{Path(val_split).stem}' / f'{label}.npz')
        return np.full(n_img, ST.ods(v['pos'].sum(0), v['neg'].sum(0))[1]), 'validation'
    t = ST.load(BINS / fam / f'{ds}__{Path(test_split).stem}' / f'{label}.npz')
    rng = np.random.default_rng(ST.SEED + 7)
    a = np.zeros(n_img, bool)
    a[rng.permutation(n_img)[: n_img // 2]] = True
    ba = ST.ods(t['pos'][a].sum(0), t['neg'][a].sum(0))[1]
    bb = ST.ods(t['pos'][~a].sum(0), t['neg'][~a].sum(0))[1]
    return np.where(a, bb, ba), 'test halves (cross-fit)'


def _nbrs(sk):
    nb = cv2.filter2D(sk.astype(np.uint8), -1, K8, borderType=cv2.BORDER_CONSTANT) - sk.astype(np.uint8)
    return nb * sk


def prune(sk, length=SPUR_PX):
    """删掉长度 ≤length 像素、一头悬空的骨架毛刺。

    逐个剥端点的做法会在毛刺根部留下一个紧贴主干的像素（它有 3 个邻居, 永远不是端点）,
    于是分叉点照数不误。这里按枝处理: 去掉分叉团及其一圈邻域把骨架拆成若干枝, 含原始端点
    且像素数 ≤length 的枝整段删除, 再细化一次。
    """
    sk = sk.astype(bool)
    for _ in range(3):                      # 删掉一层毛刺可能暴露出下一层, 迭代几次
        nb = _nbrs(sk)
        ends = (nb == 1) & sk
        junc = cv2.dilate(((nb >= 3) & sk).astype(np.uint8), K8).astype(bool)
        lab, n = ndimage.label(sk & ~junc, structure=K8)
        if n == 0:
            break
        size = np.bincount(lab.ravel(), minlength=n + 1)
        has_end = np.bincount(lab[ends], minlength=n + 1) > 0
        drop = (size <= length) & has_end
        drop[0] = False
        if not drop.any():
            break
        sk = skeletonize(sk & ~drop[lab])
    # 收尾: 分叉团邻域里的端点是被拆枝时归入分叉区、没被删掉的毛刺残根（主干端点不会紧贴
    # 分叉点）, 逐个剥掉, 至多 3 像素
    for _ in range(3):
        nb = _nbrs(sk)
        near_j = cv2.dilate(((nb >= 3) & sk).astype(np.uint8), K8, iterations=2).astype(bool)
        stub = (nb == 1) & sk & near_j
        if not stub.any():
            break
        sk = sk & ~stub
    return sk


def endpoints_junctions(sk):
    sk = prune(sk)
    nb = _nbrs(sk)
    # 8 邻域下一个分叉周围会有几个像素邻居数都 ≥3, 逐像素数会重复计, 所以按团计;
    # 并且只有至少 3 条长 ≥3 px 的枝从团里伸出时才算分叉 —— 单像素凸起在 8 邻域下也会
    # 让几个像素的邻居数到 3, 但它只把主干分成两段, 不是分叉。
    lab_j, n_j = ndimage.label((nb >= 3) & sk, structure=K8)
    n_jn = 0
    for k in range(1, n_j + 1):
        core = cv2.dilate((lab_j == k).astype(np.uint8), K8, iterations=2).astype(bool)
        ring = cv2.dilate(core.astype(np.uint8), K8).astype(bool) & ~core
        seg, ns = ndimage.label(sk & ~core, structure=K8)
        if ns == 0:
            continue
        sizes = np.bincount(seg.ravel(), minlength=ns + 1)
        arms = {int(i) for i in np.unique(seg[ring & sk]) if i and sizes[i] >= 3}
        n_jn += len(arms) >= 3
    return int(((nb == 1) & sk).sum()), int(n_jn)


def per_image(pred, gt, gray):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    r = {}
    if not gt.any():
        lab, n = ndimage.label(pred, structure=K8)
        r.update(gt_empty=True, spurious_count=int(n), pred_area=int(pred.sum()))
        return r, None
    sg, sp = skeletonize(gt), skeletonize(pred) if pred.any() else np.zeros_like(gt)
    lg, lp = int(sg.sum()), int(sp.sum())
    c = M.cldice_one(pred.astype(np.uint8), gt.astype(np.uint8))
    r['clDice'] = None if c is None else float(c)
    r['len_rel_err'] = (lp - lg) / max(lg, 1)
    r['width_bias_px'] = (pred.sum() / lp - gt.sum() / lg) if lp else None
    corridor = cv2.dilate(gt.astype(np.uint8), K8, iterations=3).astype(bool)
    _, n_gt = ndimage.label(gt, structure=K8)
    lab_p, n_p = ndimage.label(pred, structure=K8)
    touch = set(np.unique(lab_p[corridor & pred])) - {0}
    r['frag_excess'] = len(touch) - n_gt
    lab_g, _ = ndimage.label(gt, structure=K8)
    broken = missed = big = 0
    for k in range(1, n_gt + 1):
        comp = lab_g == k
        if (sg & comp).sum() < MIN_COMP_SKEL:
            continue
        big += 1
        corr = cv2.dilate(comp.astype(np.uint8), K8, iterations=3).astype(bool)
        hit = set(np.unique(lab_p[corr & pred])) - {0}
        broken += len(hit) >= 2
        missed += len(hit) == 0
    r['n_gt_long'] = big
    r['broken'] = broken
    r['missed'] = missed
    ep_g, jn_g = endpoints_junctions(sg)
    ep_p, jn_p = endpoints_junctions(sp)
    r['endpoint_excess'] = ep_p - ep_g
    r['junction_excess'] = jn_p - jn_g
    near = cv2.dilate(gt.astype(np.uint8), K8, iterations=5).astype(bool)
    spur = [k for k in range(1, n_p + 1) if not (near & (lab_p == k)).any()]
    r['spurious_count'] = len(spur)
    r['spurious_area'] = float(np.isin(lab_p, spur).sum() / max(pred.sum(), 1)) if spur else 0.0
    dt = ndimage.distance_transform_edt(gt)
    width = 2.0 * dt[sg]
    covered = cv2.dilate(pred.astype(np.uint8), K8, iterations=2).astype(bool)[sg]
    strata = np.digitize(width, WIDTH_EDGES)          # 0: <3, 1: 3–7, 2: >7
    cnt = {k: [int((strata == i).sum()), int(((strata == i) & covered).sum())] for i, k in enumerate(('thin', 'medium', 'wide'))}
    ring = cv2.dilate(gt.astype(np.uint8), K8, iterations=6).astype(bool) & ~cv2.dilate(gt.astype(np.uint8), K8, iterations=2).astype(bool)
    r['contrast'] = float((gray[ring].mean() - gray[gt].mean()) / max(gray[ring].std(), 1e-6)) if ring.any() else None
    return r, cnt


def summarize(rows, counts):
    def med(k):
        v = [r[k] for r in rows if r.get(k) is not None and not r.get('gt_empty')]
        return float(np.median(v)) if v else None

    def mean(k):
        v = [r[k] for r in rows if r.get(k) is not None and not r.get('gt_empty')]
        return float(np.mean(v)) if v else None
    tot = {k: [sum(c[k][0] for c in counts), sum(c[k][1] for c in counts)] for k in ('thin', 'medium', 'wide')}
    return dict(
        clDice_mean=mean('clDice'), len_rel_err_median=med('len_rel_err'),
        width_bias_px_median=med('width_bias_px'), frag_excess_mean=mean('frag_excess'),
        endpoint_excess_mean=mean('endpoint_excess'), junction_excess_mean=mean('junction_excess'),
        spurious_count_mean=float(np.mean([r['spurious_count'] for r in rows])),
        broken_frac=(sum(r.get('broken', 0) for r in rows) / max(sum(r.get('n_gt_long', 0) for r in rows), 1)),
        missed_frac=(sum(r.get('missed', 0) for r in rows) / max(sum(r.get('n_gt_long', 0) for r in rows), 1)),
        n_gt_long=sum(r.get('n_gt_long', 0) for r in rows),
        spurious_area_mean=mean('spurious_area'),
        recall_by_width={k: (v[1] / v[0] if v[0] else None) for k, v in tot.items()},
        skeleton_px_by_width={k: v[0] for k, v in tot.items()},
        n_images=len(rows), n_fg_images=sum(1 for r in rows if not r.get('gt_empty')))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', nargs=3, action='append', required=True, metavar=('LABEL', 'CONFIG', 'CKPT'))
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--test-split', required=True)
    ap.add_argument('--val-split', default=None)
    ap.add_argument('--family', default='m50_s42', help='直方图族, 用来取独立阈值')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--device', default='cuda:0')
    a = ap.parse_args()
    imgs, masks = P.resolve_split(a.data_root, a.test_split)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = dict(img_shape=(M.H, M.W), ori_shape=(M.H, M.W), pad_shape=(M.H, M.W), batch_input_shape=(M.H, M.W))
    for label, cfg, ckpt in a.model:
        dst = out / f'{label}.json'
        if dst.exists():
            print('SKIP', label, flush=True)
            continue
        t0 = time.time()
        bins, rule = thresholds(a.family, a.dataset, a.test_split, a.val_split, label, len(imgs))
        model, _ = EB.build_model_checked(cfg, ckpt, a.device)
        rows, counts = [], []
        with torch.no_grad():
            for k, (ip, mp) in enumerate(zip(imgs, masks)):
                bgr = M.load_bgr(ip)
                p = M.fg_prob(model, bgr, meta)
                pu = np.clip((p * M.BINS).astype(np.int32), 0, M.BINS)
                pred = pu >= bins[k]
                gt = M.load_gt(mp).astype(bool)
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
                r, c = per_image(pred, gt, gray)
                r['img'] = Path(ip).name
                rows.append(r)
                if c is not None:
                    counts.append(c)
        del model
        torch.cuda.empty_cache()
        res = dict(label=label, dataset=a.dataset, family=a.family, threshold_rule=rule,
                   ckpt=str(ckpt), seconds=round(time.time() - t0, 1),
                   summary=summarize(rows, counts), per_image=rows)
        dst.write_text(json.dumps(res, ensure_ascii=False) + '\n', encoding='utf-8')
        s = res['summary']
        print('%s %s: clDice %.3f broken %.3f missed %.3f ep %+.1f spur %.2f recall thin/med/wide %s (%.0f s)' % (
            a.dataset, label, s['clDice_mean'] or 0, s['broken_frac'], s['missed_frac'], s['endpoint_excess_mean'] or 0,
            s['spurious_count_mean'], {k: (round(v, 3) if v is not None else None) for k, v in s['recall_by_width'].items()},
            res['seconds']), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
