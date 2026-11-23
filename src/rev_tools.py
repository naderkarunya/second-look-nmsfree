"""rev_tools.py — helpers for the experiments added in the revised version of the Second-Look paper.

  * hard_nms / soft_nms        : standard suppression applied to the SAME K=500 one-to-one candidates the head receives
  * write_odgt / score_odgt    : CrowdDet-format detection files and the official AP / MR^-2 / JI scorer
  * per_image_matches / bootstrap_ci : paired bootstrap over the 4,370 validation images for differences in AP and MR^-2,
                                 using the official per-image matching of CrowdDet (the full-set values reproduce
                                 compute_APMR exactly, see check_against_official)
  * Timer helpers for the complete-pipeline latency measurement
  * cp_to_coco                 : converts cp_eval.py detections to the COCO result format of the official CityPersons evaluator
"""
from __future__ import annotations
import json, os, subprocess, sys, time
from pathlib import Path
import numpy as np

# ------------------------------------------------------------------------------------------------ suppression
def hard_nms(boxes, scores, iou_thr=0.5):
    """boxes (K,4) xyxy, scores (K,) torch tensors -> kept boxes, scores (torchvision NMS, the Ultralytics default routine)."""
    import torch
    from torchvision.ops import nms
    keep = nms(boxes.float(), scores.float(), iou_thr)
    return boxes[keep], scores[keep]


def soft_nms(boxes, scores, sigma=0.5, score_thr=0.001, method="gaussian", iou_thr=0.3):
    """Soft-NMS (Bodla et al., ICCV 2017) on numpy arrays. boxes (K,4) xyxy, scores (K,).
    method 'gaussian': s <- s * exp(-iou^2 / sigma); 'linear': s <- s * (1 - iou) for iou > iou_thr.
    Returns boxes and decayed scores of every candidate whose final score is above score_thr."""
    b = np.asarray(boxes, dtype=np.float64).copy(); s = np.asarray(scores, dtype=np.float64).copy()
    N = len(s)
    x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    areas = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    idx = np.arange(N); out_b, out_s = [], []
    while idx.size:
        m = idx[np.argmax(s[idx])]
        out_b.append(b[m]); out_s.append(s[m])
        idx = idx[idx != m]
        if not idx.size: break
        xx1 = np.maximum(x1[m], x1[idx]); yy1 = np.maximum(y1[m], y1[idx])
        xx2 = np.minimum(x2[m], x2[idx]); yy2 = np.minimum(y2[m], y2[idx])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / np.maximum(areas[m] + areas[idx] - inter, 1e-9)
        if method == "gaussian":
            s[idx] = s[idx] * np.exp(-(iou ** 2) / sigma)
        else:
            s[idx] = np.where(iou > iou_thr, s[idx] * (1 - iou), s[idx])
        idx = idx[s[idx] > score_thr]
    if not out_b:
        return np.zeros((0, 4)), np.zeros(0)
    return np.array(out_b), np.array(out_s)


# ------------------------------------------------------------------------------------------------ CrowdDet files
def gt_tags(record):
    return [{"box": g["fbox"], "tag": -1 if (g["tag"] != "person" or g.get("extra", {}).get("ignore", 0) == 1) else 1}
            for g in record["gtboxes"]]


def dt_line(record, W, H, boxes_xyxy, scores):
    dts = [{"box": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)], "score": float(s), "tag": 1}
           for (x1, y1, x2, y2), s in zip(np.asarray(boxes_xyxy).tolist(), np.asarray(scores).tolist())]
    return json.dumps({"ID": record["ID"], "height": H, "width": W, "dtboxes": dts, "gtboxes": gt_tags(record)})


def write_odgt(out_dir, dt_lines, gt_records):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "dt.odgt").write_text("\n".join(dt_lines) + "\n")
    (out / "gt_used.odgt").write_text("\n".join(json.dumps(r) for r in gt_records) + "\n")
    return out


def score_odgt(out_dir, crowddet, ji=True):
    """Official CrowdDet scorer (same call as export_eval.py / score_subsets.py). Writes <out>/metrics.json."""
    out = Path(out_dir); lib = Path(crowddet) / "lib"
    code = ("import json,sys;sys.path.insert(0,'.');from evaluate import compute_JI, compute_APMR;"
            f"AP,MR=compute_APMR.compute_APMR(r'{(out/'dt.odgt').resolve()}', r'{(out/'gt_used.odgt').resolve()}', 'box');"
            + (f"_,JI=compute_JI.evaluation_all(r'{(out/'dt.odgt').resolve()}', 'box');" if ji else "JI=None;")
            + f"json.dump(dict(AP=AP,MR=MR,JI=JI),open(r'{(out/'metrics.json').resolve()}','w'),indent=2);"
            "print('AP %.4f  MR %.4f  JI %s'%(AP,MR,JI))")
    subprocess.run([sys.executable, "-c", code], cwd=lib, check=True)
    return json.load(open(out / "metrics.json"))


# ------------------------------------------------------------------------------------------------ bootstrap
MR_REF = np.array([0.0100, 0.0178, 0.03160, 0.0562, 0.1000, 0.1778, 0.3162, 0.5623, 1.000])


def per_image_matches(dt_odgt, gt_odgt, crowddet, thres=0.5):
    """Runs the official per-image Caltech matching of CrowdDet once and returns, per image ID,
    (scores array, tp-flag array, number of non-ignored ground-truth persons)."""
    sys.path.insert(0, str(Path(crowddet) / "lib"))
    from evaluate.APMRToolkits.database import Database
    db = Database(str(gt_odgt), str(dt_odgt), "box", None, 0)
    per = {}
    for ID, img in db.images.items():
        res = img.compare_caltech(thres)
        sc = np.array([r[0][-1] for r in res], dtype=np.float64); tp = np.array([r[1] for r in res], dtype=np.int8)
        per[ID] = (sc, tp, int(img._gtNum - img._ignNum))
    return per


def ap_mr_from_pool(scores, tps, n_gt, n_img):
    """AP (trapezoid over the PR sequence) and MR^-2 exactly as CrowdDet's eval_AP / eval_MR compute them."""
    order = np.argsort(-scores, kind="stable")
    tp = np.cumsum(tps[order] == 1).astype(np.float64); fp = np.cumsum(tps[order] == 0).astype(np.float64)
    recall = tp / n_gt; precision = tp / (tp + fp)
    ap = float(np.sum((recall[1:] - recall[:-1]) * (precision[1:] + precision[:-1]) / 2))
    fppi = fp / n_img; miss = 1.0 - recall
    idx = np.searchsorted(fppi, MR_REF, side="left"); idx = np.minimum(idx, len(fppi) - 1)
    mr = float(np.exp(np.log(miss[idx]).mean()))
    return ap, mr


def eval_full(per):
    ids = list(per)
    sc = np.concatenate([per[i][0] for i in ids]); tp = np.concatenate([per[i][1] for i in ids])
    return ap_mr_from_pool(sc, tp, sum(per[i][2] for i in ids), len(ids))


class BootPool:
    """Per-image matches of one system, globally sorted once by score so that a bootstrap resample needs no sorting:
    a resample is a multiplicity vector over images, applied as per-detection repeat counts."""
    def __init__(self, per, ids):
        self.ids = ids; n = len(ids)
        sc = np.concatenate([per[i][0] for i in ids]); tp = np.concatenate([per[i][1] for i in ids])
        img = np.concatenate([np.full(len(per[i][0]), k, dtype=np.int32) for k, i in enumerate(ids)])
        order = np.argsort(-sc, kind="stable")
        self.tp = (tp[order] == 1).astype(np.float64); self.fp = (tp[order] == 0).astype(np.float64); self.img = img[order]
        self.gt = np.array([per[i][2] for i in ids], dtype=np.float64)
    def ap_mr(self, counts):
        w = counts[self.img]
        if w.sum() == 0: return float("nan"), float("nan")
        rep = np.repeat(np.arange(len(w)), w)
        tp = np.cumsum(self.tp[rep]); fp = np.cumsum(self.fp[rep])
        n_gt = float((self.gt * counts).sum()); n_img = float(counts.sum())
        recall = tp / n_gt; precision = tp / (tp + fp)
        ap = float(np.sum((recall[1:] - recall[:-1]) * (precision[1:] + precision[:-1]) / 2))
        fppi = fp / n_img; miss = 1.0 - recall
        idx = np.minimum(np.searchsorted(fppi, MR_REF, side="left"), len(fppi) - 1)
        return ap, float(np.exp(np.log(miss[idx]).mean()))


def bootstrap_ci(per_a, per_b, n_boot=1000, seed=0, alpha=0.05):
    """Paired bootstrap over images for the difference (B - A) in AP and MR^-2.
    per_a, per_b: outputs of per_image_matches for the two systems on the same images.
    Returns dict with the full-set values, the mean bootstrap difference and percentile confidence intervals (all in %)."""
    ids = sorted(set(per_a) & set(per_b)); n = len(ids)
    PA, PB = BootPool(per_a, ids), BootPool(per_b, ids)
    rng = np.random.default_rng(seed)
    dap, dmr = np.zeros(n_boot), np.zeros(n_boot)
    for t in range(n_boot):
        counts = np.bincount(rng.integers(0, n, n), minlength=n)
        a = PA.ap_mr(counts); b = PB.ap_mr(counts)
        dap[t] = (b[0] - a[0]) * 100; dmr[t] = (b[1] - a[1]) * 100
    ones = np.ones(n, dtype=np.int64)
    (apA, mrA), (apB, mrB) = PA.ap_mr(ones), PB.ap_mr(ones)
    q = [100 * alpha / 2, 100 * (1 - alpha / 2)]
    return dict(n_images=n, n_boot=n_boot, A=dict(AP=apA * 100, MR=mrA * 100), B=dict(AP=apB * 100, MR=mrB * 100),
                dAP=dict(point=(apB - apA) * 100, mean=float(dap.mean()), ci=[float(x) for x in np.percentile(dap, q)], frac_pos=float((dap > 0).mean())),
                dMR=dict(point=(mrB - mrA) * 100, mean=float(dmr.mean()), ci=[float(x) for x in np.percentile(dmr, q)], frac_neg=float((dmr < 0).mean())))


def check_against_official(dt_odgt, gt_odgt, crowddet):
    """Sanity check: the pooled per-image matching must reproduce compute_APMR on the full set."""
    per = per_image_matches(dt_odgt, gt_odgt, crowddet)
    ap, mr = eval_full(per)
    lib = Path(crowddet) / "lib"
    code = ("import sys;sys.path.insert(0,'.');from evaluate import compute_APMR;"
            f"AP,MR=compute_APMR.compute_APMR(r'{Path(dt_odgt).resolve()}', r'{Path(gt_odgt).resolve()}', 'box');print(AP, MR)")
    o = subprocess.run([sys.executable, "-c", code], cwd=lib, check=True, capture_output=True, text=True).stdout.split()[-2:]
    ap_o, mr_o = float(o[0]), float(o[1])
    print(f"pooled: AP {ap:.6f} MR {mr:.6f} | official: AP {ap_o:.6f} MR {mr_o:.6f}")
    assert abs(ap - ap_o) < 1e-6 and abs(mr - mr_o) < 1e-6, "mismatch with the official evaluator"
    return per


# ------------------------------------------------------------------------------------------------ latency
def letterbox_uint8(im, new=640, stride=32):
    """Same geometry as refine.letterbox (minimal-rectangle letterbox, centre padding, value 114) but returns the padded
    uint8 HWC image instead of a float tensor, so that normalisation can be done on the GPU."""
    import cv2, math
    H, W = im.shape[:2]
    r = min(new / H, new / W); nw, nh = round(W * r), round(H * r)
    pw, ph = (math.ceil(nw / stride) * stride - nw), (math.ceil(nh / stride) * stride - nh)
    left, top = pw // 2, ph // 2
    img = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    img = cv2.copyMakeBorder(img, top, ph - top, left, pw - left, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return img, r, (left, top)


def to_input(img_u8, dev, half=True):
    """uint8 BGR HWC (numpy) -> (1,3,H,W) RGB tensor in [0,1] on the GPU; the uint8 copy is 4x smaller than a float one."""
    import torch
    t = torch.from_numpy(img_u8).to(dev, non_blocking=True).permute(2, 0, 1)[None][:, [2, 1, 0]]
    return (t.half() if half else t.float()) / 255


class CudaTimer:
    def __init__(self):
        import torch; self.torch = torch
    def __enter__(self):
        self.torch.cuda.synchronize(); self.t0 = time.perf_counter(); return self
    def __exit__(self, *a):
        self.torch.cuda.synchronize(); self.ms = (time.perf_counter() - self.t0) * 1000


def summarize_ms(v):
    v = np.asarray(v, dtype=np.float64)
    return dict(n=int(v.size), mean=round(float(v.mean()), 2), median=round(float(np.median(v)), 2),
                p95=round(float(np.percentile(v, 95)), 2), p99=round(float(np.percentile(v, 99)), 2), max=round(float(v.max()), 2))


def patch_ultralytics_nms(log):
    """Wraps ultralytics.utils.ops.non_max_suppression so that, for every call, the number of boxes above the confidence
    threshold BEFORE suppression and the time spent inside the routine are appended to `log` (list of dicts).
    Used to check whether slow images on the NMS path are caused by the NMS routine itself."""
    import torch
    try:
        from ultralytics.utils import nms as ops      # ultralytics >= 8.4: NMS lives in ultralytics.utils.nms
    except ImportError:
        from ultralytics.utils import ops             # older releases
    orig = ops.non_max_suppression
    def wrapped(prediction, conf_thres=0.25, *args, **kwargs):
        p = prediction[0] if isinstance(prediction, (list, tuple)) else prediction
        try:
            nc = kwargs.get("nc", 0) or (p.shape[1] - 4)
            n_above = int((p[:, 4:4 + nc].amax(1) > conf_thres).sum())
        except Exception:
            n_above = -1
        torch.cuda.synchronize(); t0 = time.perf_counter()
        out = orig(prediction, conf_thres, *args, **kwargs)
        torch.cuda.synchronize()
        log.append(dict(n_above_conf=n_above, nms_ms=(time.perf_counter() - t0) * 1000))
        return out
    ops.non_max_suppression = wrapped
    return orig


# ------------------------------------------------------------------------------------------------ CityPersons
def cp_to_coco(dt_cp_json, val_gt_json, out_json):
    """cp_eval.py writes dt_cp.json = {stem: [[x,y,w,h], score], ...}; the official evaluator wants a COCO result list
    with the image ids of val_gt.json (im_name = <stem>_leftImg8bit.png)."""
    dts = json.load(open(dt_cp_json)); gt = json.load(open(val_gt_json))
    ids = {im["im_name"].replace("_leftImg8bit.png", ""): im["id"] for im in gt["images"]}
    res, missing = [], 0
    for stem, lst in dts.items():
        if stem not in ids: missing += 1; continue
        for b, s in lst:
            res.append(dict(image_id=ids[stem], category_id=1, bbox=[float(v) for v in b], score=float(s)))
    json.dump(res, open(out_json, "w"))
    print(f"{out_json}: {len(res)} detections on {len(dts) - missing} images ({missing} stems not in val_gt.json)")
    return out_json


def patch_cp_eval_script(eval_script_dir):
    """The official script was written for Python 2 / old numpy; two mechanical fixes make it run on current numpy
    (no change to the evaluation logic): integer count in np.linspace, and np.float -> np.float64."""
    import re
    for name in ("eval_MR_multisetup.py", "coco.py"):
        p = Path(eval_script_dir) / name; t = p.read_text()
        t2 = t.replace("np.linspace(.0, 1.00, np.round((1.00 - .0) / .01) + 1, endpoint=True)",
                       "np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)")
        t2 = re.sub(r"np\.float\b", "np.float64", t2); t2 = re.sub(r"np\.int\b", "np.int64", t2); t2 = re.sub(r"np\.bool\b", "bool", t2)
        if t2 != t: p.write_text(t2); print("patched", p)


def run_official_cp(eval_script_dir, val_gt_json, res_json, tag="run"):
    """Runs the official CityPersons evaluator (cvgroup-njust/CityPersons/evaluation/eval_script) for its four setups:
    Reasonable, Reasonable_small, Reasonable_occ=heavy, All. Returns {setup: MR^-2 in %}."""
    patch_cp_eval_script(eval_script_dir)
    txt = Path(eval_script_dir) / f"results_{tag}.txt"
    if txt.exists(): txt.unlink()
    code = f"""
import sys; sys.path.insert(0, r'{eval_script_dir}')
from coco import COCO
from eval_MR_multisetup import COCOeval
f = open(r'{txt}', 'w')
for id_setup in range(4):
    cocoGt = COCO(r'{val_gt_json}'); cocoDt = cocoGt.loadRes(r'{res_json}')
    E = COCOeval(cocoGt, cocoDt, 'bbox'); E.params.imgIds = sorted(cocoGt.getImgIds())
    E.evaluate(id_setup); E.accumulate(); E.summarize(id_setup, f)
f.close()
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
    out = {}
    for line in txt.read_text().splitlines():
        if "@" in line and "=" in line:
            name = line.split("@")[1].split("[")[0].strip(); out[name] = float(line.rsplit("=", 1)[1].strip().rstrip("%"))
    return out
