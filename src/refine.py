"""
refine.py — Second-Look Refinement head (SLR) on top of a FROZEN, already-trained NMS-free YOLO26 detector.

The base detector produces its one-to-one candidates (top-K boxes + scores, no NMS). The SLR head re-examines every
candidate: it pools the neck features inside the box (multi-level RoIAlign), and predicts a box correction, a score
correction and, as an occlusion-aware auxiliary, the candidate's visibility. Training targets are ONE-TO-ONE: for each
person, only the best-overlapping candidate is positive; other candidates on the same person are negatives, so the head
also learns to suppress duplicates and the output stays NMS-free.

  train:  python refine.py train --weights base_best.pt --data /tmp/ch_yolo/crowdhuman.yaml --epochs 8 --out runs/slr_A
          [--no-occ]  (ablation: no visibility auxiliary, no visibility weighting)
  eval :  python refine.py eval --weights base_best.pt --head runs/slr_A/slr.pt --images Images_val
          --gt annotation_val.odgt --out results/slr_A --crowddet CrowdDet
          -> writes results for the base alone (<out>_base) and for base+SLR (<out>) from the SAME forward pass.
"""
from __future__ import annotations

import argparse, random, json, math, os, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align, sigmoid_focal_loss, box_iou, generalized_box_iou_loss

sys.path.insert(0, str(Path(__file__).parent))
import vgyolo  # noqa: E402  (registers the custom classes so checkpoints unpickle)

BBOX_STD = (0.1, 0.1, 0.2, 0.2)   # delta scaling as in Cascade R-CNN's later stages


def pick_device(d):
    """'cpu', '0', 'cuda:0' -> torch.device (falls back to CPU when CUDA is unavailable)."""
    if str(d) == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    d = str(d)
    return torch.device(f"cuda:{d}" if d.isdigit() else d)


# ----------------------------------------------------------------------------- base detector (frozen)
class FrozenBase:
    """Runs the trained detector once and returns its neck features and one-to-one candidates."""

    def __init__(self, weights, device, k=300):
        from ultralytics import YOLO
        self.model = YOLO(weights).model.to(device).float().eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.det = self.model.model[-1]
        assert getattr(self.det, "one2one_cv2", None) is not None, "base must be an NMS-free (end2end) model"
        self.det.end2end = True
        self.det.max_det = k
        self.det.export = False
        self.strides = [int(s) for s in self.det.stride.tolist()]
        self.chs = [seq[0].conv.in_channels for seq in self.det.cv2]
        self.nc = self.det.nc
        self._feats = None
        self.det.register_forward_pre_hook(lambda m, args: setattr(self, "_feats", list(args[0])))
        self.device = device

    @torch.no_grad()
    def __call__(self, img):
        y, _ = self.model(img)                      # y: (B, K, 6) = x1 y1 x2 y2 score cls  (input-pixel coords)
        y = y.float()
        if self.nc > 1:                             # multi-class checkpoint (plumbing tests): keep class 0 = person
            y[..., 4] = torch.where(y[..., 5] == 0, y[..., 4], torch.zeros_like(y[..., 4]))
        return [f.float() for f in self._feats], y[..., :4], y[..., 4]


# ----------------------------------------------------------------------------- the head
class SLRHead(nn.Module):
    def __init__(self, chs, strides, d=64, roi=7, hidden=256, occ=True, score_only=False, rel=False, app=True):
        super().__init__()
        self.strides, self.roi, self.occ, self.score_only, self.rel, self.app = strides, roi, occ, score_only, rel, app
        n_geo = 3 + (6 if rel else 0)
        if app:   # RoI appearance branch (the full head)
            self.lateral = nn.ModuleList(nn.Sequential(nn.Conv2d(c, d, 1), nn.ReLU(inplace=True)) for c in chs)
            self.conv = nn.Sequential(nn.Conv2d(d, d, 3, padding=1), nn.ReLU(inplace=True))
            self.fc1 = nn.Sequential(nn.Linear(d * roi * roi, hidden), nn.ReLU(inplace=True))
            self.fc2 = nn.Sequential(nn.Linear(hidden + n_geo, hidden), nn.ReLU(inplace=True))
        else:     # appearance-free variant: geometry (+ relation) features only
            self.fc0 = nn.Sequential(nn.Linear(n_geo, hidden), nn.ReLU(inplace=True))
            self.fc2 = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True))
        self.out = nn.Linear(hidden, 6)              # dx dy dw dh | dscore | visibility logit
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)   # starts as the identity refinement

    def level_of(self, boxes):
        """FPN-style level assignment by box size (k0 = level with stride 16)."""
        w = (boxes[:, 2] - boxes[:, 0]).clamp(min=1); h = (boxes[:, 3] - boxes[:, 1]).clamp(min=1)
        k = torch.floor(math.log2(16) + torch.log2(torch.sqrt(w * h) / 224 + 1e-6))
        lv = k - math.log2(self.strides[0])
        return lv.clamp(0, len(self.strides) - 1).long()

    def forward(self, feats, boxes, scores):
        """feats: list of (B,C,H,W); boxes: (B,K,4) xyxy input pixels; scores: (B,K)."""
        B, K, _ = boxes.shape
        dev = boxes.device
        flat_boxes = boxes.reshape(-1, 4)
        w = (flat_boxes[:, 2] - flat_boxes[:, 0]).clamp(min=1); h = (flat_boxes[:, 3] - flat_boxes[:, 1]).clamp(min=1)
        s = scores.reshape(-1).clamp(1e-4, 1 - 1e-4)
        geo = torch.stack((torch.logit(s), torch.log(w / 640), torch.log(h / 640)), 1)
        if self.rel:
            geo = torch.cat((geo, self.relation_feats(boxes, scores).reshape(B * K, 6)), 1)
        if not self.app:
            o = self.out(self.fc2(self.fc0(geo))).view(B, K, 6)
            dbox = o[..., :4] * 0 if self.score_only else o[..., :4]
            return dbox, o[..., 4], o[..., 5]
        lat = [m(f) for m, f in zip(self.lateral, feats)]
        bidx = torch.arange(B, device=dev).repeat_interleave(K).float()
        rois = torch.cat((bidx[:, None], flat_boxes), 1)
        lv = self.level_of(flat_boxes)
        pooled = torch.zeros((B * K, lat[0].shape[1], self.roi, self.roi), device=dev, dtype=lat[0].dtype)
        for l, (f, s_) in enumerate(zip(lat, self.strides)):
            sel = (lv == l).nonzero().flatten()
            if sel.numel():
                r = roi_align(f.float(), rois[sel].float(), self.roi, spatial_scale=1.0 / s_, sampling_ratio=2, aligned=True)
                pooled[sel] = r.to(pooled.dtype)
        z = self.fc1(self.conv(pooled).flatten(1))
        o = self.out(self.fc2(torch.cat((z, geo), 1))).view(B, K, 6)
        dbox = o[..., :4] * 0 if self.score_only else o[..., :4]
        return dbox, o[..., 4], o[..., 5]

    @staticmethod
    @torch.no_grad()
    def relation_feats(boxes, scores):
        """For each candidate: its strongest HIGHER-SCORED overlapping neighbour. Returns (B,K,6):
        [max IoU with a higher-scored box, that box's score, dx/w, dy/h, log(w'/w), log(h'/h)] (zeros if none).
        This is the information a duplicate decision needs: 'is there a more confident box on the same person?'"""
        B, K, _ = boxes.shape
        out = torch.zeros((B, K, 6), device=boxes.device, dtype=torch.float32)
        for b in range(B):
            bx = boxes[b].float(); sc = scores[b].float()
            iou = box_iou(bx, bx)
            higher = sc[None, :] > sc[:, None]                      # (k, j): j has a higher score than k
            iou = torch.where(higher, iou, torch.zeros_like(iou))
            miou, j = iou.max(1)
            w = (bx[:, 2] - bx[:, 0]).clamp(min=1); h = (bx[:, 3] - bx[:, 1]).clamp(min=1)
            cx = bx[:, 0] + w / 2; cy = bx[:, 1] + h / 2
            wj, hj, cxj, cyj = w[j], h[j], cx[j], cy[j]
            f = torch.stack((miou, sc[j], (cxj - cx) / w, (cyj - cy) / h, torch.log(wj / w), torch.log(hj / h)), 1)
            out[b] = torch.where((miou > 0)[:, None], f, torch.zeros_like(f))
        return out

    @staticmethod
    def decode(boxes, d):
        w = boxes[..., 2] - boxes[..., 0]; h = boxes[..., 3] - boxes[..., 1]
        cx = boxes[..., 0] + w / 2; cy = boxes[..., 1] + h / 2
        dx, dy, dw, dh = (d[..., i] * BBOX_STD[i] for i in range(4))
        dw, dh = dw.clamp(max=math.log(4)), dh.clamp(max=math.log(4))
        cx, cy, w, h = cx + dx * w, cy + dy * h, w * torch.exp(dw), h * torch.exp(dh)
        return torch.stack((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), -1)

    @staticmethod
    def encode(boxes, gt):
        w = boxes[..., 2] - boxes[..., 0]; h = boxes[..., 3] - boxes[..., 1]
        cx = boxes[..., 0] + w / 2; cy = boxes[..., 1] + h / 2
        gw = gt[..., 2] - gt[..., 0]; gh = gt[..., 3] - gt[..., 1]
        gcx = gt[..., 0] + gw / 2; gcy = gt[..., 1] + gh / 2
        t = torch.stack(((gcx - cx) / w, (gcy - cy) / h, torch.log(gw / w), torch.log(gh / h)), -1)
        return t / torch.tensor(BBOX_STD, device=t.device)

    @staticmethod
    def refine_scores(scores, ds):
        s = scores.clamp(1e-4, 1 - 1e-4)
        return torch.sigmoid(torch.logit(s) + ds)


# ----------------------------------------------------------------------------- targets
@torch.no_grad()
def one_to_one_targets(boxes, gt_boxes, gt_vis, iou_thr=0.5, scores=None):
    """Greedy one-to-one matching for ONE image, in the order the evaluation protocol uses:
    candidates are visited by DESCENDING SCORE and each takes the free ground truth it overlaps most (IoU >= thr).
    So the positive for a person is the detector's most confident box on that person; lower-scored boxes on the
    same person become negatives (duplicates). boxes (K,4), gt_boxes (G,4), gt_vis (G,), scores (K,) or None.
    Returns pos mask (K,), matched gt box (K,4), matched vis (K,), matched IoU (K,)."""
    K = boxes.shape[0]
    pos = torch.zeros(K, dtype=torch.bool, device=boxes.device)
    tgt = torch.zeros((K, 4), device=boxes.device)
    vis = torch.zeros(K, device=boxes.device)
    q = torch.zeros(K, device=boxes.device)
    if gt_boxes.numel() == 0:
        return pos, tgt, vis, q
    iou = box_iou(boxes, gt_boxes)                    # (K,G)
    if scores is None:
        order = iou.max(1).values.argsort(descending=True)
    else:
        order = scores.argsort(descending=True)
    used_g = torch.zeros(gt_boxes.shape[0], dtype=torch.bool, device=boxes.device)
    for k in order.tolist():
        row = iou[k].clone(); row[used_g] = -1
        g = int(row.argmax())
        if row[g] < iou_thr:
            continue
        used_g[g] = True
        pos[k] = True; tgt[k] = gt_boxes[g]; vis[k] = gt_vis[g]; q[k] = iou[k, g]
        if bool(used_g.all()):
            break
    return pos, tgt, vis, q


def compute_loss(head, dbox, dscore, dvis, boxes, scores, batch, imgsz, occ=True, beta=1.0, soft=False):
    B, K = scores.shape
    dev = scores.device
    xywh = batch["bboxes"].to(dev).float() * imgsz
    gt_all = torch.cat((xywh[:, :2] - xywh[:, 2:] / 2, xywh[:, :2] + xywh[:, 2:] / 2), 1)
    bi = batch["batch_idx"].to(dev).long()
    v_all = (batch["vis"].to(dev).view(-1).float() + 0.5) / vgyolo.VG_CFG["vis_bins"]
    pos = torch.zeros((B, K), dtype=torch.bool, device=dev)
    tgt = torch.zeros((B, K, 4), device=dev); vis = torch.zeros((B, K), device=dev); q = torch.zeros((B, K), device=dev)
    for b in range(B):
        m = bi == b
        pos[b], tgt[b], vis[b], q[b] = one_to_one_targets(boxes[b], gt_all[m], v_all[m], scores=scores[b])
    n_pos = pos.sum().clamp(min=1).float()
    w = torch.ones((B, K), device=dev)
    if occ:
        w = 1.0 + beta * (1.0 - vis)
        w = torch.where(pos, w / ((w * pos).sum() / n_pos), torch.ones_like(w))
    # score: focal loss over all candidates, positives weighted by w
    target = q if soft else pos.float()               # soft: matched candidates learn their IoU, not 1
    fl = sigmoid_focal_loss(torch.logit(scores.clamp(1e-4, 1 - 1e-4)) + dscore, target, alpha=0.25, gamma=2.0,
                            reduction="none")
    l_score = (fl * torch.where(pos, w, torch.ones_like(w))).sum() / n_pos
    if pos.any() and not head.score_only:
        t = head.encode(boxes[pos], tgt[pos])
        l_l1 = (F.smooth_l1_loss(dbox[pos], t, reduction="none", beta=1.0 / 9).sum(-1) * w[pos]).sum() / n_pos
        l_giou = (generalized_box_iou_loss(head.decode(boxes[pos], dbox[pos]), tgt[pos], reduction="none") * w[pos]).sum() / n_pos
        l_vis = (F.binary_cross_entropy_with_logits(dvis[pos], vis[pos], reduction="none") * w[pos]).sum() / n_pos if occ else dbox.sum() * 0
    else:
        l_l1 = l_giou = dbox.sum() * 0
        l_vis = (F.binary_cross_entropy_with_logits(dvis[pos], vis[pos], reduction="none") * w[pos]).sum() / n_pos if (occ and pos.any()) else dbox.sum() * 0
    total = l_score + l_l1 + 2.0 * l_giou + (0.5 * l_vis if occ else 0)
    return total, dict(score=float(l_score.detach()), l1=float(l_l1.detach()), giou=float(l_giou.detach()), vis=float(l_vis.detach()), npos=float(n_pos))


# ----------------------------------------------------------------------------- train
def train(a):
    from ultralytics.cfg import get_cfg
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.data import build_dataloader
    dev = pick_device(a.device)
    torch.manual_seed(a.seed); random.seed(a.seed)
    base = FrozenBase(a.weights, dev, k=a.k)
    head = SLRHead(base.chs, base.strides, occ=not a.no_occ, score_only=a.score_only, rel=a.rel, app=not a.no_app).to(dev)
    print(f"SLR head: {sum(p.numel() for p in head.parameters())/1e6:.3f}M params; levels {base.strides}; occ={not a.no_occ}")
    args = get_cfg(overrides=dict(data=a.data, imgsz=a.imgsz, batch=a.batch, workers=a.workers, device=str(dev),
                                  model="yolo26s-p2.yaml", mosaic=0.0, mixup=0.0, copy_paste=0.0, fraction=a.fraction))
    data = check_det_dataset(a.data)
    ds = vgyolo.build_vg_dataset(args, data["train"], a.batch, data, "train", stride=max(base.strides))
    dl = build_dataloader(ds, a.batch, a.workers, shuffle=True, rank=-1)
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * len(dl)
    warm = max(1, int(0.05 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: (i + 1) / warm if i < warm else 0.5 * (1 + math.cos(math.pi * min(1.0, (i - warm) / max(1, steps - warm)))))
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time(); it = 0
    for ep in range(a.epochs):
        head.train(); agg = {}
        for batch in dl:
            batch = vgyolo.split_vis(batch)
            img = batch["img"].to(dev, non_blocking=True).float() / 255
            with torch.autocast(device_type=dev.type, enabled=dev.type == "cuda" or os.environ.get("SLR_FORCE_AMP") == "1",
                                dtype=torch.float16 if dev.type == "cuda" else torch.bfloat16):
                feats, boxes, scores = base(img)
                dbox, dscore, dvis = head(feats, boxes, scores)
            loss, items = compute_loss(head, dbox.float(), dscore.float(), dvis.float(), boxes, scores, batch, a.imgsz,
                                       occ=not a.no_occ, soft=a.soft)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); nn.utils.clip_grad_norm_(head.parameters(), 10.0)
            scaler.step(opt); scaler.update(); sched.step(); it += 1
            for k, v in items.items():
                agg[k] = agg.get(k, 0) + v
            if a.max_hours and (time.time() - t0) > a.max_hours * 3600:
                print("MAX_HOURS reached - stopping"); break
        n = max(1, len(dl))
        print(f"epoch {ep+1}/{a.epochs} " + " ".join(f"{k}={v/n:.4f}" for k, v in agg.items()) +
              f"  ({(time.time()-t0)/3600:.2f} h)", flush=True)
        torch.save({"head": head.state_dict(), "chs": base.chs, "strides": base.strides, "occ": not a.no_occ, "k": a.k,
                    "score_only": a.score_only, "soft": a.soft, "rel": a.rel, "app": not a.no_app, "epochs_done": ep + 1}, out / "slr.pt")
        if a.max_hours and (time.time() - t0) > a.max_hours * 3600:
            break
    (out / "train_info.json").write_text(json.dumps(dict(weights=a.weights, epochs=a.epochs, fraction=a.fraction,
                                                          k=a.k, occ=not a.no_occ, score_only=a.score_only, soft=a.soft, rel=a.rel, app=not a.no_app, seed=a.seed,
                                                          hours=round((time.time()-t0)/3600, 3), iters=it), indent=2))


# ----------------------------------------------------------------------------- eval
def load_head(ck, dev):
    """Builds an SLRHead from a checkpoint dict (handles older checkpoints without the 'app' field)."""
    head = SLRHead(ck["chs"], ck["strides"], occ=ck["occ"], score_only=ck.get("score_only", False), rel=ck.get("rel", False),
                   app=ck.get("app", True)).to(dev).eval()
    head.load_state_dict(ck["head"])
    return head


def letterbox(im, new=640, stride=32):
    """Minimal-rectangle letterbox (as Ultralytics predict, auto=True). Returns tensor, ratio, (left, top)."""
    import cv2
    H, W = im.shape[:2]
    r = min(new / H, new / W)
    nw, nh = round(W * r), round(H * r)
    pw, ph = (math.ceil(nw / stride) * stride - nw), (math.ceil(nh / stride) * stride - nh)
    left, top = pw // 2, ph // 2
    img = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    img = cv2.copyMakeBorder(img, top, ph - top, left, pw - left, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    t = torch.from_numpy(img[:, :, ::-1].copy()).permute(2, 0, 1).float() / 255
    return t[None], r, (left, top)


def evaluate(a):
    import cv2
    from PIL import Image
    from export_eval import vis_ratio  # noqa: F401
    from score_subsets import score
    dev = pick_device(a.device)
    base = FrozenBase(a.weights, dev, k=a.k)
    ck = torch.load(a.head, map_location="cpu", weights_only=False)
    head = load_head(ck, dev)
    gts = [json.loads(l) for l in open(a.gt)]
    if a.only_existing:
        gts = [r for r in gts if (Path(a.images) / f"{r['ID']}.jpg").exists()]
    if a.limit:
        gts = gts[:a.limit]
    print(f"eval mode={a.mode} images={len(gts)} head={a.head}")
    lines = {"base": [], "slr": []}
    t0 = time.time()
    for i, r in enumerate(gts):
        im = cv2.imread(str(Path(a.images) / f"{r['ID']}.jpg"))
        H, W = im.shape[:2]
        x, ratio, (left, top) = letterbox(im, a.imgsz, max(base.strides))
        x = x.to(dev)
        with torch.no_grad(), torch.autocast(device_type=dev.type, enabled=dev.type == "cuda" or os.environ.get("SLR_FORCE_AMP") == "1",
                                             dtype=torch.float16 if dev.type == "cuda" else torch.bfloat16):
            feats, boxes, scores = base(x)
            dbox, dscore, _ = head(feats, boxes, scores)
        dbox, dscore = dbox.float(), dscore.float()
        if a.mode == "score": dbox = dbox * 0          # diagnostic: apply only the score correction
        if a.mode == "box": dscore = dscore * 0        # diagnostic: apply only the box correction
        rb = head.decode(boxes, dbox)[0]; rs = head.refine_scores(scores.float(), dscore)[0]
        gtb = [{"box": g["fbox"], "tag": -1 if (g["tag"] != "person" or g.get("extra", {}).get("ignore", 0) == 1) else 1}
               for g in r["gtboxes"]]
        for key, bx, sc in (("base", boxes[0], scores[0]), ("slr", rb, rs)):
            b = bx.clone(); b[:, [0, 2]] -= left; b[:, [1, 3]] -= top; b /= ratio
            b[:, [0, 2]] = b[:, [0, 2]].clamp(0, W); b[:, [1, 3]] = b[:, [1, 3]].clamp(0, H)
            keep = sc > a.conf
            dts = [{"box": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)], "score": float(s), "tag": 1}
                   for (x1, y1, x2, y2), s in zip(b[keep].tolist(), sc[keep].tolist())]
            lines[key].append(json.dumps({"ID": r["ID"], "height": H, "width": W, "dtboxes": dts, "gtboxes": gtb}))
        if (i + 1) % 500 == 0:
            print(f"{i+1}/{len(gts)} images ({(time.time()-t0)/60:.1f} min)", flush=True)
    gt_lines = [json.dumps(r) for r in gts]
    results = {}
    for key, out in (("base", Path(a.out + "_base")), ("slr", Path(a.out))):
        out.mkdir(parents=True, exist_ok=True)
        (out / "dt.odgt").write_text("\n".join(lines[key]) + "\n")
        (out / "gt_used.odgt").write_text("\n".join(gt_lines) + "\n")
        print(key, "all:", end=" "); score(out, a.crowddet)
        results[key] = json.load(open(out / "metrics.json"))
        if a.subsets:
            import subprocess
            subprocess.run([sys.executable, str(Path(__file__).parent / "score_subsets.py"), "--dt", str(out / "dt.odgt"),
                            "--gt", a.gt, "--out", str(out), "--crowddet", a.crowddet] + (["--no-ji"] if a.no_ji else []), check=True)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--weights", required=True); t.add_argument("--data", required=True); t.add_argument("--out", required=True)
    t.add_argument("--epochs", type=int, default=8); t.add_argument("--fraction", type=float, default=1.0)
    t.add_argument("--batch", type=int, default=16); t.add_argument("--workers", type=int, default=2)
    t.add_argument("--imgsz", type=int, default=640); t.add_argument("--k", type=int, default=300)
    t.add_argument("--lr", type=float, default=1e-3); t.add_argument("--device", default="0")
    t.add_argument("--no-occ", action="store_true"); t.add_argument("--max-hours", type=float, default=0.0)
    t.add_argument("--score-only", action="store_true", help="train only the score correction (box deltas fixed at 0)")
    t.add_argument("--soft", action="store_true", help="matched candidates learn their IoU as score target instead of 1")
    t.add_argument("--rel", action="store_true", help="add relation features (strongest higher-scored overlapping neighbour)")
    t.add_argument("--no-app", action="store_true", help="appearance-free head: no RoI branch, geometry (+ relation) features only")
    t.add_argument("--seed", type=int, default=0)
    e = sub.add_parser("eval")
    e.add_argument("--weights", required=True); e.add_argument("--head", required=True)
    e.add_argument("--images", required=True); e.add_argument("--gt", required=True); e.add_argument("--out", required=True)
    e.add_argument("--crowddet", required=True); e.add_argument("--imgsz", type=int, default=640)
    e.add_argument("--k", type=int, default=300); e.add_argument("--conf", type=float, default=0.001)
    e.add_argument("--device", default="0"); e.add_argument("--only-existing", action="store_true")
    e.add_argument("--subsets", action="store_true"); e.add_argument("--no-ji", action="store_true", help="skip JI on the subsets (faster)")
    e.add_argument("--mode", default="full", choices=["full", "score", "box"]); e.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    train(a) if a.cmd == "train" else evaluate(a)
