#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""逐图逐 bin 直方图评测（CACAIE-D-26-01086 一审）。

为什么另写一个，而不是给 eval_public_ods.py 再加开关：那个脚本只落盘全测试集的聚合
值（以及 ODS 阈值处的逐图 tp/fp/fn），一审要的几样东西都要求能在**任意图像子集**上、
在**任意阈值**下重算：
  - R2-1 独立标定阈值：在验证集（或 DeepCrack 测试集的一半）上选 t*，冻结后在测试集报
    F1/IoU；DeepCrack 要多次随机对半。
  - R2-1/R2-5 嵌套 bootstrap：每个重抽样里重新选阈值，而不是把全集 t* 固定住。
  - B2 CrackMap 5 折：五个折各是一个模型，合并 120 张图的计数再算一条曲线。
  - R2-2 共同图像上的配对比较、B2 逐图配对。
这些都只需要每张图在每个 bin 上的正、负像素数。本脚本对每个模型跑一遍前向，存
pos[N, BINS+1] 与 neg[N, BINS+1]（像素数，uint32）。之后的统计全部在 CPU 上由
rev1_bins_stats.py 完成，不再推理。

口径与正文完全一致，因为前向、读图、读掩码、量化全部直接调用 eval_public_ods.py 所用
的同一个模块（projects/crack_baselines/eval_crack_v5_domain_metrics.py）：
  p = fg_prob(...)；bin = clip(int(p * BINS), 0, BINS)；gt = mask > 127。
所以由直方图算出的 pixel-ODS 与 eval_public_ods.py 逐位相同（--check 可当场核对）。

不能由直方图得到的是 clDice 与几何指标（端点、断裂、宽度偏差……），它们需要二值图；
等阈值定下来以后再单独跑一遍，不在这里。

输出：<out-dir>/<label>.npz，键 pos、neg、names，外加 meta（JSON 字符串：配置、
checkpoint、其 SHA-256 前 16 位、数据根、划分、BINS、torch/mmcv 版本、主机、耗时）。
已存在且 meta 中的 checkpoint 指纹一致时跳过，可反复调用。
"""
import argparse
import hashlib
import json
import platform
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import importlib.util as _ilu  # noqa: E402

_SRC = ROOT / 'projects' / 'crack_baselines' / 'eval_crack_v5_domain_metrics.py'
_spec = _ilu.spec_from_file_location('_crack_v5_metrics', _SRC)
_m = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_m)

_SRC2 = ROOT / 'projects' / 'mixercseg_baselines' / 'eval_public_ods.py'
_spec2 = _ilu.spec_from_file_location('_eval_public_ods', _SRC2)
_p = _ilu.module_from_spec(_spec2)
_spec2.loader.exec_module(_p)

BINS, H, W = _m.BINS, _m.H, _m.W


def build_model_checked(cfg_path, ckpt, device):
    """同 _m.build_model（模型构建与加载方式一致），但不容忍缺键。

    _m.build_model 用 load_state_dict(strict=False) 且不报告结果：代码树与 checkpoint
    对不上时（例如 .228 的两棵树、改过名的模块），模型会带着随机初始化的层被静默打分。
    这里把缺键与多余键数记进 meta；模型里有参数在 checkpoint 中找不到即报错。多余键
    （如训练期的辅助头）允许存在。
    """
    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmseg.registry import MODELS
    cfg = Config.fromfile(cfg_path)
    init_default_scope(cfg.get('default_scope', 'mmseg'))
    cfg.model.pop('pretrained', None)
    cfg.model.pop('train_cfg', None)
    model = MODELS.build(cfg.model)
    sd = torch.load(ckpt, map_location='cpu', weights_only=False)
    res = model.load_state_dict(sd.get('state_dict', sd), strict=False)
    if res.missing_keys:
        raise RuntimeError(f'{len(res.missing_keys)} model keys missing from checkpoint, '
                           f'e.g. {res.missing_keys[:3]}')
    model.to(device).eval()
    return model, len(res.unexpected_keys)


def ckpt_fingerprint(path):
    """前 64 MB 的 SHA-256：足以区分不同的 checkpoint，又不必把 5 GB 读完。"""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        h.update(f.read(64 << 20))
    h.update(str(Path(path).stat().st_size).encode())
    return h.hexdigest()[:16]


@torch.no_grad()
def histograms(model, imgs, masks):
    meta = dict(img_shape=(H, W), ori_shape=(H, W), pad_shape=(H, W),
                batch_input_shape=(H, W))
    pos = np.zeros((len(imgs), BINS + 1), np.uint32)
    neg = np.zeros((len(imgs), BINS + 1), np.uint32)
    for k, (ip, mp) in enumerate(zip(imgs, masks)):
        p = _m.fg_prob(model, _m.load_bgr(ip), meta)
        gt = _m.load_gt(mp)
        pu = np.clip((p * BINS).astype(np.int32), 0, BINS)
        pos[k] = np.bincount(pu[gt == 1], minlength=BINS + 1)
        neg[k] = np.bincount(pu[gt == 0], minlength=BINS + 1)
    return pos, neg


def pooled_ods(pos, neg):
    F1, _, _ = _m.f1_curve(pos.sum(0).astype(np.int64), neg.sum(0).astype(np.int64))
    bi = int(np.argmax(F1))
    return float(F1[bi] * 100), bi / BINS


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', nargs=3, action='append', required=True,
                    metavar=('LABEL', 'CONFIG', 'CKPT'))
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--split', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--check', default=None,
                    help='已有的 eval_public_ods.py 结果 JSON；同名 label 的 ODS 与'
                         '本次直方图算出的 ODS 不一致（>0.001）时报错退出')
    a = ap.parse_args()

    imgs, masks = _p.resolve_split(a.data_root, a.split)
    names = np.array([Path(x).name for x in imgs])
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ref = json.load(open(a.check, encoding='utf-8')) if a.check else None
    print(f'{len(imgs)} images from {a.data_root}/{a.split} -> {out}', flush=True)

    bad = []
    for label, cfg, ckpt in a.model:
        dst = out / f'{label}.npz'
        if not Path(ckpt).exists():
            print(f'SKIP {label}: checkpoint missing {ckpt}', flush=True)
            continue
        fp = ckpt_fingerprint(ckpt)
        if dst.exists():
            old = json.loads(str(np.load(dst)['meta']))
            if old.get('ckpt_sha16') == fp and old.get('split') == a.split:
                print(f'SKIP {label}: up to date', flush=True)
                continue
        t0 = time.time()
        try:
            model, n_unexpected = build_model_checked(cfg, ckpt, a.device)
            pos, neg = histograms(model, imgs, masks)
        except Exception as exc:  # 记下来、评下一个；结尾以非零码退出
            print(f'FAILED {label}: {type(exc).__name__}: {str(exc)[:300]}', flush=True)
            bad.append(label)
            torch.cuda.empty_cache()
            continue
        del model
        torch.cuda.empty_cache()
        ods, thr = pooled_ods(pos, neg)
        meta = dict(label=label, config=cfg, ckpt=str(ckpt), ckpt_sha16=fp,
                    data_root=a.data_root, split=a.split, num_images=len(imgs),
                    bins=BINS, input_hw=[H, W], ODS=round(ods, 3), ODS_thr=thr,
                    missing_keys=0, unexpected_keys=n_unexpected,
                    torch=torch.__version__, host=socket.gethostname(),
                    platform=platform.platform(), seconds=round(time.time() - t0, 1),
                    date=time.strftime('%Y-%m-%dT%H:%M:%S'))
        try:
            import mmcv
            meta['mmcv'] = mmcv.__version__
        except ImportError:
            pass
        np.savez_compressed(dst, pos=pos, neg=neg, names=names, meta=json.dumps(meta))
        msg = f'{label}: ODS {ods:.3f} @ {thr:.3f} ({meta["seconds"]} s)'
        if ref is not None and label in ref and 'ODS' in ref[label]:
            d = ods - ref[label]['ODS']
            msg += f' | ref {ref[label]["ODS"]:.3f} diff {d:+.4f}'
            if abs(d) > 1e-3:
                bad.append(label)
        print(msg, flush=True)
    if bad:
        raise SystemExit(f'failed or ODS mismatch vs --check: {bad}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
