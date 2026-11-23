"""Figs 10-12: REAL detections of the final model (B, B + SLR, B with NMS) on real CrowdHuman validation images.
Operating thresholds = 1 false positive per image on the full validation split (analysis5.json)."""
import json, os, sys, numpy as np
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analysis5 import load_gt, match_image, clip_xywh, iou_matrix, iof_matrix, DT
HERE = os.path.dirname(os.path.abspath(__file__)); UP = "../data"
AN = json.load(open(f"{HERE}/analysis5.json")); THR = {k: AN[k]["thr_1fppi"] for k in ("nms", "free", "slr")}
CW, CH_ = 440, 330
GREEN, RED, ORANGE, GREY, WHITE, BLUE = (0, 190, 70), (235, 40, 40), (255, 170, 0), (200, 200, 200), (255, 255, 255), (0, 120, 220)
def font(s, b=False): return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf" % ("-Bold" if b else ""), s)
IMG_DIRS = [f"{UP}/Images_val"]
def img_path(i):
    for d in IMG_DIRS:
        p = f"{d}/{i}.jpg"
        if os.path.exists(p): return p
    raise FileNotFoundError(i)
NEED = set(json.load(open(f"{HERE}/fig_image_choice.json"))["show"]) | {"282555,1972a000479a36d0", "284193,8adf00083e8ad37", "283554,3699400056a59b6c",
        "284193,5481000108ceae3", "273275,b929b0006234664e", "273275,147c4d0000c6f6fb8", "282555,ae07c0000d1669d8", "273278,119b49000145232b1", "282555,f7f3000097603e5c",
        "273278,e8c1300073571165", "282555,ec6d4000b8753a14"}
GT = load_gt()
def load_subset(path):
    out = {}
    for l in open(path):
        r = json.loads(l)
        if r["ID"] in NEED:
            b = np.array([d["box"] for d in r["dtboxes"]], float).reshape(-1, 4); s = np.array([d["score"] for d in r["dtboxes"]], float)
            out[r["ID"]] = (clip_xywh(b, r["width"], r["height"]), s, r["width"], r["height"])
    return out
D = {k: load_subset(p) for k, p in DT.items()}
for i in NEED:
    _, _, W, H = D["free"][i]; GT[i]["boxes"] = clip_xywh(GT[i]["boxes"], W, H)

def cover(img, focus=None, zoom=1.0):
    W, H = img.size; t = CW / CH_
    cw, ch = (int(H * t), H) if W / H > t else (W, int(W / t))
    cw, ch = int(cw / zoom), int(ch / zoom)
    fx, fy = focus if focus else (W / 2, H / 2)
    x0 = int(min(max(fx - cw / 2, 0), W - cw)); y0 = int(min(max(fy - ch / 2, 0), H - ch)); s = CW / cw
    m = lambda b: [(b[0] - x0) * s, (b[1] - y0) * s, b[2] * s, b[3] * s]
    return img.crop((x0, y0, x0 + cw, y0 + ch)).resize((CW, CH_), Image.LANCZOS), m
def rect(d, b, col, w=3, dash=False):
    x, y, bw, bh = b; x2, y2 = x + bw, y + bh
    if not dash: d.rectangle([x, y, x2, y2], outline=col, width=w); return
    seg = 7
    for (xa, ya, xb, yb) in [(x, y, x2, y), (x, y2, x2, y2), (x, y, x, y2), (x2, y, x2, y2)]:
        L = max(abs(xb - xa), abs(yb - ya)); n = max(int(L // seg), 1)
        for k in range(0, n, 2):
            t0, t1 = k / n, min((k + 1) / n, 1)
            d.line([xa + (xb - xa) * t0, ya + (yb - ya) * t0, xa + (xb - xa) * t1, ya + (yb - ya) * t1], fill=col, width=w)
def tag(d, b, txt, col, dark_text=True):
    if b[2] < 26: return
    x, y = max(b[0], 0), max(b[1] - 15, 0); tw = d.textlength(txt, font=font(11, True))
    d.rectangle([x, y, x + tw + 6, y + 14], fill=col); d.text((x + 3, y + 1), txt, fill="black" if dark_text else "white", font=font(11, True))
def label(d, text):
    tw = d.textlength(text, font=font(12, True)); d.rectangle([0, 0, tw + 10, 18], fill=(0, 0, 0)); d.text((5, 2), text, fill="white", font=font(12, True))
def panel_title(canvas, x, y, text):
    d = ImageDraw.Draw(canvas); d.text((x + CW / 2, y), text, fill="black", font=font(13, True), anchor="mm")
def grid(cols, rows, title_h=26, gap=6):
    return Image.new("RGB", (cols * CW + (cols - 1) * gap, rows * (CH_ + title_h) + (rows - 1) * gap), "white")

def crowd_view(img, g):
    """focus + zoom on the union of the annotated persons (keeps the whole crowd, drops empty sky/ceiling)"""
    W, H = img.size; b = g["boxes"][~g["ign"]]
    if len(b) == 0: return None, 1.0
    x1, y1 = b[:, 0].min(), b[:, 1].min(); x2, y2 = (b[:, 0] + b[:, 2]).max(), (b[:, 1] + b[:, 3]).max()
    uw, uh = max(x2 - x1, 1), max(y2 - y1, 1); t = CW / CH_
    cw, ch = (int(H * t), H) if W / H > t else (W, int(W / t))
    zoom = max(1.0, min(2.2, 0.92 * min(cw / uw, ch / uh)))
    return ((x1 + x2) / 2, (y1 + y2) / 2), zoom

def gt_panel(img, m, g, show_vis=True):
    d = ImageDraw.Draw(img)
    for b, ign in zip(g["boxes"], g["ign"]):
        if ign: rect(d, m(b), GREY, 2, dash=True)
    for b, ign, v in zip(g["boxes"], g["ign"], g["vis"]):
        if ign: continue
        rect(d, m(b), GREEN, 3)
    return img

def det_panel(img, m, g, db, ds, thr, show_scores=True, extra=None):
    """TP green (score), FP red (score), missed GT white dashed. Returns counts."""
    d = ImageDraw.Draw(img); a = ds >= thr
    st, mg, gd, biou, bg, iou = match_image(db[a], ds[a], g)
    for j, (b, ign) in enumerate(zip(g["boxes"], g["ign"])):
        if not ign and gd[j] < 0: rect(d, m(b), WHITE, 2, dash=True)
    for t, (b, s) in enumerate(zip(db[a], ds[a])):
        if st[t] == 1: rect(d, m(b), GREEN, 3); tag(d, m(b), f"{s:.2f}", GREEN) if show_scores else None
        elif st[t] == 0: rect(d, m(b), RED, 3); tag(d, m(b), f"{s:.2f}", RED, False) if show_scores else None
    tp = int((st == 1).sum()); fp = int((st == 0).sum()); fn = int(((gd < 0) & ~g["ign"]).sum())
    return tp, fp, fn

# ---------------------------------------------------------------- Fig 10: effect of the head (GT | candidates | refined)
import sys as _sys
TWO = "--two" in _sys.argv
show = ["273275,443d7000118a603d", "282555,e2d8d000c98f2299"] if TWO else ["273275,443d7000118a603d", "282555,e2d8d000c98f2299", "273278,4f2d80007164d216"]
canvas = grid(3, len(show)); TH = 26
for r, i in enumerate(show):
    g = GT[i]; img0 = Image.open(img_path(i)).convert("RGB"); W, H = img0.size
    db_f, ds_f, _, _ = D["free"][i]; db_s, ds_s, _, _ = D["slr"][i]
    # panel (a)
    fz = crowd_view(img0, g)
    ia, m = cover(img0.copy(), *fz); gt_panel(ia, m, g); ImageDraw.Draw(ia).text((5, CH_ - 16), f"{int((~g['ign']).sum())} annotated persons", fill="white", font=font(11, True))
    # panel (b): one-to-one candidates: those above the base threshold (TP green / FP red) and, dashed orange, the candidates below it that the head will promote
    ib, m = cover(img0.copy(), *fz); d = ImageDraw.Draw(ib)
    key_s = {tuple(np.round(b, 3)): n for n, b in enumerate(db_s)}
    a = ds_f >= THR["free"]; st, mg, gd, biou, bg, iou = match_image(db_f[a], ds_f[a], g)
    for j, (b, ign) in enumerate(zip(g["boxes"], g["ign"])):
        if not ign and gd[j] < 0: rect(d, m(b), WHITE, 2, dash=True)
    # promoted: candidates below base thr whose head score is above the head thr and that are TPs under the head output
    a_s = ds_s >= THR["slr"]; st_s, mg_s, gd_s, *_ = match_image(db_s[a_s], ds_s[a_s], g)
    promoted = set()
    for t in np.where(a_s)[0]:
        pass
    idx_s_above = np.where(a_s)[0]
    for n, t in enumerate(idx_s_above):
        if st_s[n] == 1:
            k = key_s.get(tuple(np.round(db_s[t], 3)))
            # find this box in the base list
            kk = np.where(np.all(np.isclose(db_f, db_s[t], atol=1e-3), axis=1))[0]
            if len(kk) and ds_f[kk[0]] < THR["free"]: promoted.add(kk[0])
    for t in promoted: rect(d, m(db_f[t]), ORANGE, 3, dash=True); tag(d, m(db_f[t]), f"{ds_f[t]:.2f}", ORANGE)
    for t, (b, s) in enumerate(zip(db_f[a], ds_f[a])):
        if st[t] == 1: rect(d, m(b), GREEN, 3)
        elif st[t] == 0: rect(d, m(b), RED, 3); tag(d, m(b), f"{s:.2f}", RED, False)
    tp, fp, fn = int((st == 1).sum()), int((st == 0).sum()), int(((gd < 0) & ~g["ign"]).sum())
    label(d, f"TP {tp} · FP {fp} · missed {fn}")
    # panel (c): refined output — tags only on the recovered people (orange) and on the false positives (red)
    ic, m = cover(img0.copy(), *fz); d = ImageDraw.Draw(ic)
    for j, (b, ign) in enumerate(zip(g["boxes"], g["ign"])):
        if not ign and gd_s[j] < 0: rect(d, m(b), WHITE, 2, dash=True)
    prom_boxes = {tuple(np.round(db_f[t], 3)) for t in promoted}
    for n, t in enumerate(idx_s_above):
        b, sc = db_s[t], ds_s[t]
        if st_s[n] == 1:
            if tuple(np.round(b, 3)) in prom_boxes: rect(d, m(b), ORANGE, 3); tag(d, m(b), f"{sc:.2f}", ORANGE)
            else: rect(d, m(b), GREEN, 3)
        elif st_s[n] == 0: rect(d, m(b), RED, 3); tag(d, m(b), f"{sc:.2f}", RED, False)
    tp2, fp2, fn2 = int((st_s == 1).sum()), int((st_s == 0).sum()), int(((gd_s < 0) & ~g["ign"]).sum()); label(d, f"TP {tp2} · FP {fp2} · missed {fn2}")
    for cidx, im in enumerate((ia, ib, ic)):
        x = cidx * (CW + 6); y = r * (CH_ + TH + 6)
        canvas.paste(im, (x, y + TH))
    if r == 0:
        for cidx, t in enumerate(["(a) ground truth", "(b) one-to-one candidates of B (NMS-free), s ≥ 0.44", "(c) B + SLR output, s ≥ 0.51"]):
            panel_title(canvas, cidx * (CW + 6), 12, t)
    else:
        pass
canvas.save(f"{HERE}/fig10_qualitative_2rows.png" if TWO else f"{HERE}/fig10_qualitative.png"); print("saved fig10")
# (two-row mode continues to Fig. 11)

# ---------------------------------------------------------------- Fig 11: grid GT | NMS | one-to-one | + SLR
rows_ids = ["284193,8adf00083e8ad37", "273271,14f0d00052515611", "283992,1217e000df9ee218"]
if TWO: rows_ids = rows_ids[:2]
canvas = grid(4, len(rows_ids))
for r, i in enumerate(rows_ids):
    g = GT[i]; img0 = Image.open(img_path(i)).convert("RGB")
    fz = crowd_view(img0, g)
    ia, m = cover(img0.copy(), *fz); gt_panel(ia, m, g); ImageDraw.Draw(ia).text((5, CH_ - 16), f"{int((~g['ign']).sum())} annotated persons", fill="white", font=font(11, True))
    panels = [ia]
    for k in ("nms", "free", "slr"):
        db, ds, _, _ = D[k][i]; im, m = cover(img0.copy(), *fz); tp, fp, fn = det_panel(im, m, g, db, ds, THR[k], show_scores=False); label(ImageDraw.Draw(im), f"TP {tp} · FP {fp} · missed {fn}"); panels.append(im)
    for cidx, im in enumerate(panels):
        canvas.paste(im, (cidx * (CW + 6), r * (CH_ + TH + 6) + TH))
    if r == 0:
        for cidx, t in enumerate(["(a) ground truth", "(b) one-to-many head + NMS", "(c) one-to-one head (NMS-free)", "(d) one-to-one + SLR (NMS-free)"]):
            panel_title(canvas, cidx * (CW + 6), 12, t)
canvas.save(f"{HERE}/fig11_grid_2rows.png" if TWO else f"{HERE}/fig11_grid.png"); print("saved fig11")

# ---------------------------------------------------------------- Fig 12: failure cases of the final model (B + SLR) at 1 FPPI
cases = [("273278,119b49000145232b1", "gt", 0, "(a) no candidate on a hidden person (v = 0.12)"),
         ("273275,b929b0006234664e", "dt", 3, "(b) surviving duplicate (IoU 0.80 with a detected person)"),
         ("282555,ae07c0000d1669d8", "dt", 0, "(c) detection inside an ignore region"),
         ("273275,147c4d0000c6f6fb8", "dt", 6, "(d) one box spanning two people")]
canvas = grid(4, 1)
for cidx, (i, kind, idx, title) in enumerate(cases):
    g = GT[i]; img0 = Image.open(img_path(i)).convert("RGB"); db, ds, _, _ = D["slr"][i]; a = ds >= THR["slr"]
    st, mg, gd, biou, bg, iou = match_image(db[a], ds[a], g)
    fb = g["boxes"][idx] if kind == "gt" else db[a][idx]
    focus = (fb[0] + fb[2] / 2, fb[1] + fb[3] / 2)
    W, H = img0.size; t = CW / CH_; cw, ch = (int(H * t), H) if W / H > t else (W, int(W / t))
    zoom = max(1.0, min(4.0 if kind == "dt" else 2.0, 0.55 * min(cw / max(fb[2] * 2.2, 1), ch / max(fb[3], 1))))
    if kind == "gt": focus = (focus[0], focus[1] - fb[3] * 0.4)
    im, m = cover(img0.copy(), focus, zoom); d = ImageDraw.Draw(im)
    pos_iou = np.where(~g["ign"], iou[idx] if kind == "dt" else 0, 0)
    if kind == "gt":                                   # (a) the missed person and the detected people around it
        for j in np.where(~g["ign"])[0]:
            if j != idx and gd[j] >= 0 and iou_matrix(g["boxes"][[j]], g["boxes"][[idx]])[0, 0] > 0.05: rect(d, m(g["boxes"][j]), GREEN, 2)
        rect(d, m(fb), ORANGE, 3, dash=True); tag(d, m(fb), f"missed person, v = {g['vis'][idx]:.2f}", ORANGE)
    else:
        for j in np.where(pos_iou >= 0.3)[0]:          # the people this box overlaps
            rect(d, m(g["boxes"][j]), GREEN, 2)
            if gd[j] >= 0: rect(d, m(db[a][gd[j]]), GREEN, 3)   # the detection that claimed them
        for j in np.where(g["ign"] & (iof_matrix(db[a][[idx]], g["boxes"])[0] >= 0.3))[0]: rect(d, m(g["boxes"][j]), WHITE, 3, dash=True); tag(d, [m(g["boxes"][j])[0], m(g["boxes"][j])[1] + m(g["boxes"][j])[3] + 15, 100, 10], "ignore region", WHITE)
        rect(d, m(fb), RED, 3); tag(d, m(fb), f"false positive, s = {ds[a][idx]:.2f}", RED, False)
    canvas.paste(im, (cidx * (CW + 6), TH)); panel_title(canvas, cidx * (CW + 6), 12, title)
canvas.save(f"{HERE}/fig12_failures.png"); print("saved fig12")
