"""cp_eval.py — CityPersons evaluation: log-average miss rate MR^-2 on the standard subsets.

Protocol (CityPersons / Caltech convention, as in the widely used eval_MR_multisetup):
  * only 'pedestrian' objects are positives; all other labels (rider, sitting person, person (other), person group)
    are ignore regions; positives outside the subset's height / visibility range are ignored too
  * detections whose height is outside [h_min / 1.25, h_max * 1.25] of the subset are discarded (expFilter)
  * greedy matching in descending score at IoU >= 0.5 (non-ignored GT preferred); a detection matched only to an
    ignore GT is neither TP nor FP; a detection matching nothing is a FP
  * MR^-2 = exp(mean(log(miss rate))) over 9 FPPI reference points log-spaced in [1e-2, 1e0]; the miss rate at a
    reference point is 1 - recall of the last detection whose FPPI <= reference (1 if there is none)
Subsets (h = full-box height in pixels, v = vis-box area / full-box area):
  Reasonable: h>=50, v>=0.65 | Small: 50<=h<75, v>=0.65 | Heavy: h>=50, 0.20<=v<0.65 |
  Partial: h>=50, 0.65<=v<0.90 | Bare: h>=50, v>=0.90 | All: h>=20, v>=0.20

Usage
  predict + score (Ultralytics path, NMS or NMS-free):
    python cp_eval.py --weights best.pt --root /data/citypersons --out results/cp_x --imgsz 1280 [--nms-free]
  frozen detector + SLR head (NMS-free):
    python cp_eval.py --weights best.pt --head slr.pt --root /data/citypersons --out results/cp_x --imgsz 1280
"""
import argparse, json, math, os, sys
from pathlib import Path
import numpy as np

SETUPS = {  # name: (h_min, h_max, v_min, v_max)
    "reasonable": (50, 1e9, 0.65, 1.01), "small": (50, 75, 0.65, 1.01), "heavy": (50, 1e9, 0.20, 0.65),
    "partial": (50, 1e9, 0.65, 0.90), "bare": (50, 1e9, 0.90, 1.01), "all": (20, 1e9, 0.20, 1.01)}
EXP = 1.25
FPPI_REF = np.logspace(-2, 0, 9)


def load_gt(root, split="val"):
    """Returns dict stem -> list of (xywh, vis_ratio, is_pedestrian) and the image path."""
    root = Path(root); gts = {}
    for jf in sorted((root / "gtBboxCityPersons" / split).glob("*/*_gtBboxCityPersons.json")):
        stem = jf.name.replace("_gtBboxCityPersons.json", "")
        img = root / "leftImg8bit" / split / jf.parent.name / f"{stem}_leftImg8bit.png"
        objs = []
        for o in json.load(open(jf))["objects"]:
            b, vb = o["bbox"], o.get("bboxVis", o["bbox"])
            v = (vb[2] * vb[3]) / max(b[2] * b[3], 1e-6)
            objs.append((b, min(1.0, v), o["label"] == "pedestrian"))
        gts[stem] = (img, objs)
    return gts


def iou_xywh(d, g):
    dx1, dy1, dx2, dy2 = d[0], d[1], d[0] + d[2], d[1] + d[3]
    gx1, gy1, gx2, gy2 = g[0], g[1], g[0] + g[2], g[1] + g[3]
    iw = max(0.0, min(dx2, gx2) - max(dx1, gx1)); ih = max(0.0, min(dy2, gy2) - max(dy1, gy1)); inter = iw * ih
    return inter / max(d[2] * d[3] + g[2] * g[3] - inter, 1e-9)


def iof_xywh(d, g):   # intersection over detection area (for ignore regions)
    dx1, dy1, dx2, dy2 = d[0], d[1], d[0] + d[2], d[1] + d[3]
    gx1, gy1, gx2, gy2 = g[0], g[1], g[0] + g[2], g[1] + g[3]
    iw = max(0.0, min(dx2, gx2) - max(dx1, gx1)); ih = max(0.0, min(dy2, gy2) - max(dy1, gy1))
    return iw * ih / max(d[2] * d[3], 1e-9)


def mr2(dts, gts, setup):
    """dts: dict stem -> list of (xywh, score); gts: dict stem -> list of (xywh, v, is_ped). Returns MR^-2 (%)."""
    hmin, hmax, vmin, vmax = SETUPS[setup]
    scores, tps, n_pos = [], [], 0
    for stem, g in gts.items():
        pos = [b for b, v, ped in g if ped and hmin <= b[3] < hmax and vmin <= v < vmax]
        ign = [b for b, v, ped in g if not (ped and hmin <= b[3] < hmax and vmin <= v < vmax)]
        n_pos += len(pos)
        d = [(b, s) for b, s in dts.get(stem, []) if hmin / EXP <= b[3] < hmax * EXP]
        d.sort(key=lambda x: -x[1])
        used = [False] * len(pos)
        for b, s in d:
            best, bi = 0.5, -1
            for i, p in enumerate(pos):
                if used[i]: continue
                u = iou_xywh(b, p)
                if u >= best: best, bi = u, i
            if bi >= 0:
                used[bi] = True; scores.append(s); tps.append(1); continue
            if any(iof_xywh(b, q) >= 0.5 for q in ign):
                continue                                   # matched an ignore region: not counted
            scores.append(s); tps.append(0)
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-np.array(scores), kind="stable")
    tp = np.cumsum(np.array(tps)[order]); fp = np.cumsum(1 - np.array(tps)[order])
    recall = tp / n_pos; fppi = fp / len(gts)
    mrs = []
    for ref in FPPI_REF:
        idx = np.searchsorted(fppi, ref, side="right") - 1
        mrs.append(1.0 if idx < 0 else 1.0 - recall[idx])
    mrs = np.clip(np.array(mrs), 1e-10, 1.0)
    return float(np.exp(np.mean(np.log(mrs))) * 100)


def predict_ultralytics(weights, gts, imgsz, conf, max_det, device, nms_free):
    import vgyolo  # noqa: F401
    from ultralytics import YOLO
    model = YOLO(weights); dts = {}
    for i, (stem, (img, _)) in enumerate(gts.items()):
        res = model.predict(str(img), imgsz=imgsz, conf=conf, max_det=max_det, device=device, verbose=False,
                            **({"nms": False} if nms_free else {}))[0]
        b = res.boxes
        xyxy = b.xyxy.cpu().numpy() if len(b) else np.zeros((0, 4)); sc = b.conf.cpu().numpy() if len(b) else np.zeros(0)
        dts[stem] = [([float(x1), float(y1), float(x2 - x1), float(y2 - y1)], float(s)) for (x1, y1, x2, y2), s in zip(xyxy, sc)]
        if (i + 1) % 100 == 0: print(f"{i+1}/{len(gts)}", flush=True)
    return dts


def predict_with_head(weights, head_path, gts, imgsz, conf, k, device):
    import torch, cv2
    import refine
    dev = refine.pick_device(device)
    base = refine.FrozenBase(weights, dev, k=k)
    ck = torch.load(head_path, map_location="cpu", weights_only=False)
    head = refine.SLRHead(ck["chs"], ck["strides"], occ=ck["occ"], score_only=ck.get("score_only", False), rel=ck.get("rel", False)).to(dev).eval()
    head.load_state_dict(ck["head"])
    dts_base, dts_head = {}, {}
    for i, (stem, (img, _)) in enumerate(gts.items()):
        im = cv2.imread(str(img)); H, W = im.shape[:2]
        x, ratio, (left, top) = refine.letterbox(im, imgsz, max(base.strides)); x = x.to(dev)
        with torch.no_grad(), torch.autocast(device_type=dev.type, enabled=dev.type == "cuda", dtype=torch.float16):
            feats, boxes, scores = base(x); dbox, dscore, _ = head(feats, boxes, scores)
        rb = head.decode(boxes, dbox.float())[0]; rs = head.refine_scores(scores.float(), dscore.float())[0]
        for store, bx, sc in ((dts_base, boxes[0], scores[0]), (dts_head, rb, rs)):
            b = bx.clone(); b[:, [0, 2]] -= left; b[:, [1, 3]] -= top; b /= ratio
            b[:, [0, 2]] = b[:, [0, 2]].clamp(0, W); b[:, [1, 3]] = b[:, [1, 3]].clamp(0, H)
            keep = sc > conf
            store[stem] = [([x1, y1, x2 - x1, y2 - y1], s) for (x1, y1, x2, y2), s in zip(b[keep].tolist(), sc[keep].tolist())]
        if (i + 1) % 100 == 0: print(f"{i+1}/{len(gts)}", flush=True)
    return dts_base, dts_head


def score_all(dts, gts, out):
    g = {s: objs for s, (_, objs) in gts.items()}
    res = {k: round(mr2(dts, g, k), 2) for k in SETUPS}
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / "metrics_cp.json").write_text(json.dumps(res, indent=2))
    (Path(out) / "dt_cp.json").write_text(json.dumps({s: d for s, d in dts.items()}))
    print(out, "MR^-2 (%):", res)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True); ap.add_argument("--root", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--head", default=None); ap.add_argument("--split", default="val")
    ap.add_argument("--imgsz", type=int, default=1280); ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--max-det", type=int, default=500); ap.add_argument("--device", default="0")
    ap.add_argument("--nms-free", action="store_true"); ap.add_argument("--only-existing", action="store_true")
    a = ap.parse_args()
    gts = load_gt(a.root, a.split)
    if a.only_existing:
        gts = {s: v for s, v in gts.items() if v[0].exists()}
    print(f"CityPersons {a.split}: {len(gts)} images")
    if a.head:
        dts_b, dts_h = predict_with_head(a.weights, a.head, gts, a.imgsz, a.conf, a.max_det, a.device)
        score_all(dts_b, gts, a.out + "_base"); score_all(dts_h, gts, a.out)
    else:
        dts = predict_ultralytics(a.weights, gts, a.imgsz, a.conf, a.max_det, a.device, a.nms_free)
        score_all(dts, gts, a.out)
