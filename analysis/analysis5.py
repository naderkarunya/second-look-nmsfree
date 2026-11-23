"""analysis5.py — Section 5 analyses from the real detection files (CrowdHuman val, 4,370 images).

Inputs (uploads): dt.txt (B, one-to-many + NMS), slr_on_B_noocc_basedt.txt (B, one-to-one NMS-free),
slr_on_B_nooccdt.txt (B + SLR head).  GT: annotation_val.odgt (full).
Outputs: analysis5.json (operating thresholds, error categories, score-shift statistics, MR-FPPI curves,
per-crowd-density AP), used by figs5_real.py and build5.
"""
import json, sys, os, math
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from export_eval import vis_ratio

UP = Path(os.environ.get("SLR_DATA_DIR", Path(__file__).resolve().parent.parent / "data"))   # see README: detections + annotation_val.odgt
GT_PATH = UP / "annotation_val.odgt"
DT = {"nms": UP / "dt.txt", "free": UP / "slr_on_B_noocc_basedt.txt", "slr": UP / "slr_on_B_nooccdt.txt"}
OUT = Path(__file__).parent / "analysis5.json"


def load_gt():
    gts = {}
    for l in open(GT_PATH):
        r = json.loads(l)
        boxes, ign, vis, hs = [], [], [], []
        for g in r["gtboxes"]:
            fb = g["fbox"]
            is_ign = g["tag"] != "person" or g.get("extra", {}).get("ignore", 0) == 1
            boxes.append(fb); ign.append(is_ign); hs.append(fb[3])
            vis.append(vis_ratio(fb, g.get("vbox", fb), 10**6, 10**6) if not is_ign else 1.0)
        gts[r["ID"]] = dict(boxes=np.array(boxes, float).reshape(-1, 4), ign=np.array(ign, bool),
                            vis=np.array(vis, float), h=np.array(hs, float))
    return gts


def clip_xywh(b, W, H):
    """Official evaluator clips every box to the image before matching (x1 in [0,W-1], y1 in [0,H-1], x2 in [0,W], y2 in [0,H])."""
    if len(b) == 0: return b
    x1 = np.clip(b[:, 0], 0, W - 1); y1 = np.clip(b[:, 1], 0, H - 1)
    x2 = np.clip(b[:, 0] + b[:, 2], 0, W); y2 = np.clip(b[:, 1] + b[:, 3], 0, H)
    return np.stack([x1, y1, np.maximum(x2 - x1, 0), np.maximum(y2 - y1, 0)], 1)


def load_dt(path):
    dts = {}
    for l in open(path):
        r = json.loads(l)
        b = np.array([d["box"] for d in r["dtboxes"]], float).reshape(-1, 4)
        s = np.array([d["score"] for d in r["dtboxes"]], float)
        dts[r["ID"]] = (clip_xywh(b, r["width"], r["height"]), s, r["width"], r["height"])
    return dts


def iou_matrix(a, b):  # xywh
    if len(a) == 0 or len(b) == 0: return np.zeros((len(a), len(b)))
    ax1, ay1, ax2, ay2 = a[:, 0], a[:, 1], a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax2[:, None], bx2[None]) - np.maximum(ax1[:, None], bx1[None]), 0, None)
    ih = np.clip(np.minimum(ay2[:, None], by2[None]) - np.maximum(ay1[:, None], by1[None]), 0, None)
    inter = iw * ih
    return inter / (a[:, 2:].prod(1)[:, None] + b[:, 2:].prod(1)[None] - inter + 1e-9)


def iof_matrix(a, b):  # intersection over area of a
    if len(a) == 0 or len(b) == 0: return np.zeros((len(a), len(b)))
    ax1, ay1, ax2, ay2 = a[:, 0], a[:, 1], a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax2[:, None], bx2[None]) - np.maximum(ax1[:, None], bx1[None]), 0, None)
    ih = np.clip(np.minimum(ay2[:, None], by2[None]) - np.maximum(ay1[:, None], by1[None]), 0, None)
    return iw * ih / (a[:, 2:].prod(1)[:, None] + 1e-9)


def match_image(db, ds, g, thr=0.5):
    """Greedy one-to-one matching in descending score (CrowdDet style).
    Returns per-dt: status (1 TP, 0 FP, -1 ignored), matched gt index; per-gt: matched dt index (-1 none).
    Also returns best IoU of each dt with any positive gt and the index of that gt (for error categories)."""
    order = np.argsort(-ds, kind="stable")
    pos = ~g["ign"]; gb = g["boxes"]
    iou = iou_matrix(db, gb)
    iof = iof_matrix(db, gb)
    status = np.zeros(len(db), int); mg = -np.ones(len(db), int); gd = -np.ones(len(gb), int)
    used = np.zeros(len(gb), bool)
    best_iou = np.zeros(len(db)); best_g = -np.ones(len(db), int)
    if len(gb):
        iou_pos = np.where(pos[None], iou, -1)
        best_g = iou_pos.argmax(1); best_iou = iou_pos.max(1)
    for k in order:
        if len(gb) == 0: status[k] = 0; continue
        row = np.where(used | ~pos, -1, iou[k])
        j = int(row.argmax())
        if row[j] >= thr:
            used[j] = True; status[k] = 1; mg[k] = j; gd[j] = k; continue
        # unmatched: ignored if it overlaps an ignore region (IoF >= 0.5) or an already-matched gt? -> only ignore regions
        if np.any((iof[k] >= 0.5) & g["ign"]):
            status[k] = -1
        else:
            status[k] = 0
    return status, mg, gd, best_iou, best_g, iou


def main():
    gts = load_gt(); ids = list(gts.keys())
    n_img = len(ids); n_pos = int(sum((~g["ign"]).sum() for g in gts.values()))
    print("images", n_img, "positives", n_pos)
    dts = {k: load_dt(p) for k, p in DT.items()}
    for i in ids:                                   # clip the GT boxes to the image as the official evaluator does
        _, _, W, H = dts["free"][i]; gts[i]["boxes"] = clip_xywh(gts[i]["boxes"], W, H)
    res = {"n_images": n_img, "n_positives": n_pos}

    # ---------- 1. full matching at conf 0.001 (as in the evaluation) -> curves, thresholds
    matched = {}
    for k in dts:
        sc_all, st_all = [], []
        per_img = {}
        for i in ids:
            db, ds, W, H = dts[k][i]; g = gts[i]
            status, mg, gd, biou, bg, iou = match_image(db, ds, g)
            per_img[i] = (status, mg, gd, biou, bg)
            keep = status >= 0
            sc_all.append(ds[keep]); st_all.append(status[keep])
        sc = np.concatenate(sc_all); st = np.concatenate(st_all)
        o = np.argsort(-sc, kind="stable"); sc, st = sc[o], st[o]
        tp = np.cumsum(st == 1); fp = np.cumsum(st == 0)
        rec = tp / n_pos; fppi = fp / n_img
        # operating threshold at 1 FPPI
        idx = np.searchsorted(fppi, 1.0, side="right") - 1
        thr1 = float(sc[idx]) if idx >= 0 else 1.0
        # MR^-2 (own implementation, for the curve figure; tables use the official code)
        ref = np.logspace(-2, 0, 9); mrs = []
        for r in ref:
            j = np.searchsorted(fppi, r, side="right") - 1
            mrs.append(1.0 if j < 0 else 1 - rec[j])
        mr2 = float(np.exp(np.mean(np.log(np.clip(mrs, 1e-10, 1)))))
        # AP (VOC-style continuous)
        prec = tp / np.maximum(tp + fp, 1)
        ap = float(np.trapezoid(np.maximum.accumulate(prec[::-1])[::-1], rec))
        # curve samples (log-spaced fppi)
        grid = np.logspace(-3, 1, 60); curve = []
        for r in grid:
            j = np.searchsorted(fppi, r, side="right") - 1
            curve.append(1.0 if j < 0 else float(1 - rec[j]))
        matched[k] = per_img
        res[k] = dict(thr_1fppi=thr1, recall_1fppi=float(rec[idx]) if idx >= 0 else 0.0, mr2_own=round(mr2 * 100, 2),
                      ap_own=round(ap * 100, 2), curve_fppi=grid.tolist(), curve_mr=curve)
        print(k, "thr@1FPPI %.3f recall %.3f MR2(own) %.2f AP(own) %.2f" % (thr1, res[k]["recall_1fppi"], mr2 * 100, ap * 100))

    # ---------- 1b. MR-FPPI curves on the Severe subset (others ignored)
    for k in dts:
        sc_all, st_all = [], []; n_sev = 0
        for i in ids:
            db, ds, W, H = dts[k][i]; g = gts[i]
            g2 = dict(g); sev = (~g["ign"]) & (g["vis"] < 0.3); n_sev += int(sev.sum())
            g2["ign"] = g["ign"] | ~sev
            status, *_ = match_image(db, ds, g2)
            keep = status >= 0; sc_all.append(ds[keep]); st_all.append(status[keep])
        sc = np.concatenate(sc_all); st = np.concatenate(st_all); o = np.argsort(-sc, kind="stable"); sc, st = sc[o], st[o]
        tp = np.cumsum(st == 1); fp = np.cumsum(st == 0); rec = tp / n_sev; fppi = fp / n_img
        grid = np.logspace(-3, 1, 60); curve = []
        for r in grid:
            j = np.searchsorted(fppi, r, side="right") - 1
            curve.append(1.0 if j < 0 else float(1 - rec[j]))
        res[k]["curve_mr_severe"] = curve
    res["n_severe"] = n_sev

    # ---------- 2. error analysis at 1 FPPI for 'slr' (final) and 'free' (base)
    for k in ("slr", "free", "nms"):
        thr = res[k]["thr_1fppi"]
        FN = dict(no_candidate=0, below_thr_small=0, below_thr_other=0, localisation=0, other=0)
        FP = dict(duplicate=0, spanning_two=0, near_ignore=0, background=0)
        n_fn = n_fp = 0
        for i in ids:
            db, ds, W, H = dts[k][i]; g = gts[i]
            above = ds >= thr
            status, mg, gd, biou, bg = match_image(db[above], ds[above], g)[:5]
            pos_idx = np.where(~g["ign"])[0]
            iou_all = iou_matrix(db, g["boxes"]) if len(db) else np.zeros((0, len(g["boxes"])))
            for j in pos_idx:
                if gd[j] >= 0: continue
                n_fn += 1
                best = iou_all[:, j].max() if len(db) else 0.0
                if best < 0.3: FN["no_candidate"] += 1
                elif best < 0.5: FN["localisation"] += 1
                else:
                    if g["h"][j] < 50: FN["below_thr_small"] += 1
                    else: FN["below_thr_other"] += 1
            # false positives above threshold
            dba = db[above]
            iou_a = iou_matrix(dba, g["boxes"]) if len(dba) else np.zeros((0, len(g["boxes"])))
            iof_a = iof_matrix(dba, g["boxes"]) if len(dba) else iou_a
            for t in np.where(status == 0)[0]:
                n_fp += 1
                row = np.where(~g["ign"], iou_a[t], 0)
                if row.max() >= 0.5: FP["duplicate"] += 1           # overlaps a person already claimed by a higher-scored box
                elif (row >= 0.3).sum() >= 2: FP["spanning_two"] += 1
                elif np.any((iof_a[t] >= 0.3) & g["ign"]): FP["near_ignore"] += 1
                else: FP["background"] += 1
        res[k]["errors_1fppi"] = dict(n_fn=n_fn, n_fp=n_fp, FN=FN, FP=FP,
                                      FN_pct={a: round(100 * b / max(n_fn, 1), 1) for a, b in FN.items()},
                                      FP_pct={a: round(100 * b / max(n_fp, 1), 1) for a, b in FP.items()})
        print(k, "errors@1FPPI", res[k]["errors_1fppi"])

    # ---------- 3. score shift of the head: base (free) vs slr, same candidate boxes
    logit = lambda s: np.log(np.clip(s, 1e-6, 1 - 1e-6) / (1 - np.clip(s, 1e-6, 1 - 1e-6)))
    cats = {"tp": [], "dup": [], "bg": [], "tp_hi": [], "dup_hi": [], "bg_hi": []}; vis_bins = {b: [] for b in ("<0.3", "0.3-0.65", ">=0.65")}
    sb = {"tp": [], "dup": [], "bg": []}; ss = {"tp": [], "dup": [], "bg": []}      # raw scores before / after
    for i in ids:
        db_b, ds_b, W, H = dts["free"][i]; db_s, ds_s, _, _ = dts["slr"][i]; g = gts[i]
        if len(db_b) == 0 or len(db_s) == 0: continue
        # map base boxes to slr boxes by exact coordinates
        key_s = {tuple(np.round(b, 3)): n for n, b in enumerate(db_s)}
        idx_s = np.array([key_s.get(tuple(np.round(b, 3)), -1) for b in db_b])
        ok = idx_s >= 0
        if ok.sum() == 0: continue
        status, mg, gd, biou, bg = match_image(db_b, ds_b, g)[:5]      # categories defined on the BASE ranking
        d = logit(ds_s[idx_s[ok]]) - logit(ds_b[ok])
        st = status[ok]; bi = biou[ok]; bgi = bg[ok]
        sbo = ds_b[ok]; sso = ds_s[idx_s[ok]]
        for n in range(len(d)):
            hi = sbo[n] >= 0.3
            if st[n] == 1:
                cats["tp"].append(d[n]); sb["tp"].append(sbo[n]); ss["tp"].append(sso[n]); v = g["vis"][bgi[n]]
                vis_bins["<0.3" if v < 0.3 else ("0.3-0.65" if v < 0.65 else ">=0.65")].append(d[n])
                if hi: cats["tp_hi"].append(d[n])
            elif st[n] == 0 and bi[n] >= 0.5:
                cats["dup"].append(d[n]); sb["dup"].append(sbo[n]); ss["dup"].append(sso[n])
                if hi: cats["dup_hi"].append(d[n])
            elif st[n] == 0:
                cats["bg"].append(d[n]); sb["bg"].append(sbo[n]); ss["bg"].append(sso[n])
                if hi: cats["bg_hi"].append(d[n])
    def summ(x):
        x = np.array(x); return dict(n=int(len(x)), median=float(np.median(x)), mean=float(x.mean()), q25=float(np.percentile(x, 25)), q75=float(np.percentile(x, 75)),
                                     hist=np.histogram(x, bins=np.linspace(-6, 4, 41))[0].tolist())
    res["score_shift"] = {c: summ(v) for c, v in cats.items()}
    res["score_shift_by_vis"] = {c: summ(v) for c, v in vis_bins.items()}
    res["score_shift_bins"] = np.linspace(-6, 4, 41).tolist()
    def shist(x): return np.histogram(np.array(x), bins=np.linspace(0, 1, 41))[0].tolist()
    def auc(pos, neg):   # probability that a random TP outranks a random duplicate (Mann-Whitney)
        pos, neg = np.array(pos), np.array(neg)
        if len(pos) > 20000: pos = np.random.default_rng(0).choice(pos, 20000, replace=False)
        if len(neg) > 20000: neg = np.random.default_rng(0).choice(neg, 20000, replace=False)
        return float((pos[:, None] > neg[None]).mean() + 0.5 * (pos[:, None] == neg[None]).mean())
    res["scores_before"] = {c: dict(hist=shist(v), median=float(np.median(v)), mean=float(np.mean(v))) for c, v in sb.items()}
    res["scores_after"] = {c: dict(hist=shist(v), median=float(np.median(v)), mean=float(np.mean(v))) for c, v in ss.items()}
    res["auc_tp_vs_dup"] = {"before": auc(sb["tp"], sb["dup"]), "after": auc(ss["tp"], ss["dup"])}
    res["auc_tp_vs_bg"] = {"before": auc(sb["tp"], sb["bg"]), "after": auc(ss["tp"], ss["bg"])}
    print("median scores before/after:", {c: (round(res["scores_before"][c]["median"], 3), round(res["scores_after"][c]["median"], 3)) for c in sb},
          "AUC tp vs dup", res["auc_tp_vs_dup"], "tp vs bg", res["auc_tp_vs_bg"])
    print("score shift medians (base>=0.3): tp %.2f dup %.2f bg %.2f | n" % tuple(res["score_shift"][c]["median"] for c in ("tp_hi", "dup_hi", "bg_hi")), {c: res["score_shift"][c]["n"] for c in cats})
    print("score shift medians: tp %.2f dup %.2f bg %.2f | by vis" % tuple(res["score_shift"][c]["median"] for c in ("tp", "dup", "bg")),
          {c: round(v["median"], 2) for c, v in res["score_shift_by_vis"].items()})

    # ---------- 4. accuracy vs crowd density (persons per image) — recall at the 1-FPPI threshold and FP per image
    bins = [(0, 10), (10, 20), (20, 40), (40, 400)]
    dens = {k: [] for k in dts}
    for k in dts:
        thr = res[k]["thr_1fppi"]
        for lo, hi in bins:
            tp = fp = npos = nimg = 0
            for i in ids:
                g = gts[i]; n = int((~g["ign"]).sum())
                if not (lo <= n < hi): continue
                db, ds, W, H = dts[k][i]; above = ds >= thr
                status, mg, gd, *_ = match_image(db[above], ds[above], g)
                tp += int((status == 1).sum()); fp += int((status == 0).sum()); npos += n; nimg += 1
            dens[k].append(dict(range=[lo, hi], images=nimg, recall=round(tp / max(npos, 1), 4), fp_per_image=round(fp / max(nimg, 1), 3)))
    res["density"] = dens
    print("density", {k: [(d["range"], d["recall"], d["fp_per_image"]) for d in v] for k, v in dens.items()})

    json.dump(res, open(OUT, "w"))
    print("written", OUT)


if __name__ == "__main__":
    main()
