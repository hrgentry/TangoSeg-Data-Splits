#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""原始分辨率下的评测（CACAIE 一审 B7; R2-6 后半）。

R2-6: 检验保持纵横比或原生分辨率的处理是否改变几何指标或排名, 尤其是被拉变形的
CamCrack789（640x480 -> 512x512）与被放大的 CrackMap（256x256 -> 512x512）。

两臂都在**原始分辨率的真值**上比较, 否则比的是两个不同的标注:
  resize512  矩阵原有的模型（512x512 训练）: 输入缩放到 512 推理, 概率图双线性插值回原尺寸;
  native     B7 在原始分辨率上训练的模型: 直接在原尺寸推理。
每个模型先在验证集上扫阈值选 t*（独立阈值, 与主指标同口径）, 再在测试集上报:
  heldout_F1 / heldout_IoU、oracle ODS、AP, 以及 eval_rev1_geometry.py 的全部几何指标（同一实现）。
前向、量化 (1,000 bins) 与读掩码 (mask>127) 来自正文所用的同一个指标模块。
输出 work_dirs/rev1/native_eval/<数据集>/<mode>/<label>.json; 已有的跳过。
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
GEO = _load('_geo', 'projects/mixercseg_baselines/eval_rev1_geometry.py')


@torch.no_grad()
def prob_native(model, bgr, mode):
    h, w = bgr.shape[:2]
    if mode == 'resize512':
        x = cv2.resize(bgr, (M.W, M.H))
        meta = dict(img_shape=(M.H, M.W), ori_shape=(M.H, M.W), pad_shape=(M.H, M.W),
                    batch_input_shape=(M.H, M.W))
        p = M.fg_prob(model, x, meta)
        return cv2.resize(p, (w, h), interpolation=cv2.INTER_LINEAR)
    meta = dict(img_shape=(h, w), ori_shape=(h, w), pad_shape=(h, w), batch_input_shape=(h, w))
    return M.fg_prob(model, bgr, meta)


def read_gt(path):
    m = cv2.imread(M.to_local(path), cv2.IMREAD_GRAYSCALE)
    return (m > 127).astype(np.uint8)


def run_split(model, imgs, masks, mode, geom_bin=None):
    pos, neg, rows, counts = [], [], [], []
    for ip, mp in zip(imgs, masks):
        bgr = cv2.imread(M.to_local(ip), cv2.IMREAD_COLOR)
        gt = read_gt(mp)
        p = prob_native(model, bgr, mode)
        pu = np.clip((p * M.BINS).astype(np.int32), 0, M.BINS)
        pos.append(np.bincount(pu[gt == 1], minlength=M.BINS + 1))
        neg.append(np.bincount(pu[gt == 0], minlength=M.BINS + 1))
        if geom_bin is not None:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
            r, c = GEO.per_image(pu >= geom_bin, gt.astype(bool), gray)
            r['img'] = Path(ip).name
            rows.append(r)
            if c is not None:
                counts.append(c)
    return np.array(pos, np.int64), np.array(neg, np.int64), rows, counts


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', nargs=3, action='append', required=True, metavar=('LABEL', 'CONFIG', 'CKPT'))
    ap.add_argument('--mode', required=True, choices=['resize512', 'native'])
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--test-split', required=True)
    ap.add_argument('--val-split', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--device', default='cuda:0')
    a = ap.parse_args()
    ti, tm = P.resolve_split(a.data_root, a.test_split)
    vi, vm = P.resolve_split(a.data_root, a.val_split)
    out = Path(a.out_dir) / a.dataset / a.mode
    out.mkdir(parents=True, exist_ok=True)
    for label, cfg, ckpt in a.model:
        dst = out / f'{label}.json'
        if dst.exists():
            print('SKIP', label, flush=True)
            continue
        if not Path(ckpt).exists():
            print('MISSING', label, ckpt, flush=True)
            continue
        t0 = time.time()
        model, _ = EB.build_model_checked(cfg, ckpt, a.device)
        vp, vn, _, _ = run_split(model, vi, vm, a.mode)
        _, b = ST.ods(vp.sum(0), vn.sum(0))
        tp_, tn_, rows, counts = run_split(model, ti, tm, a.mode, geom_bin=b)
        del model
        torch.cuda.empty_cache()
        ps, ns = tp_.sum(0), tn_.sum(0)
        h = ST.at_bin(ps, ns, b)
        ods, ob = ST.ods(ps, ns)
        res = dict(label=label, dataset=a.dataset, mode=a.mode, ckpt=str(ckpt), cfg=cfg,
                   heldout_thr=b / 1000, heldout_F1=h['F1'], heldout_IoU=h['IoU'],
                   ODS=ods, ODS_thr=ob / 1000, AP=ST.average_precision(ps, ns),
                   geometry=GEO.summarize(rows, counts), n_test=len(ti), n_val=len(vi),
                   seconds=round(time.time() - t0, 1))
        dst.write_text(json.dumps(res, ensure_ascii=False) + '\n', encoding='utf-8')
        g = res['geometry']
        print('%s %s %s: heldout F1 %.2f (ODS %.2f) clDice %.3f broken %.3f thin-recall %s (%.0f s)' % (
            a.dataset, a.mode, label, res['heldout_F1'], ods, g['clDice_mean'] or 0, g['broken_frac'],
            None if g['recall_by_width']['thin'] is None else round(g['recall_by_width']['thin'], 3), res['seconds']),
            flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
