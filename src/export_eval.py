"""
export_eval.py — run a trained model on the CrowdHuman validation images and score it with the OFFICIAL CrowdDet
evaluation code (AP, MR^-2, JI), so the numbers are comparable with published papers (Section IV-B of the paper).

Usage
  python export_eval.py --weights runs/.../best.pt --images /data/CrowdHuman/Images_val \
      --gt /data/CrowdHuman/annotation_val.odgt --out results/vg_seed0 --imgsz 640 --crowddet /work/CrowdDet

  --crowddet : path to a clone of https://github.com/Purkialo/CrowdDet (only lib/evaluate and lib/utils are used)
  --subset   : optional visibility group (occlusion subsets): light | heavy | severe  (others become ignore)
  --oracle   : oracle experiment (variant G): the model's predicted visibility map is replaced by the
               ground-truth map V* built from the annotation file (same rule as training). VG models only.
               Also reports how close the model's own prediction is to V* (vis_mae_person, vis_mae_occluded).

Writes <out>/dt.odgt (one JSON line per image, CrowdDet format) and <out>/metrics.json.
Detections are made on the original (un-painted) validation images; ignore regions are handled by the evaluator.
"""
import argparse, json, os, subprocess, sys
from pathlib import Path

import numpy as np
from PIL import Image


def vis_ratio(fbox, vbox, W, H):
    def clip(b):
        x, y, w, h = b
        x1, y1, x2, y2 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
        return x1, y1, max(0, x2 - x1), max(0, y2 - y1)
    fx, fy, fw, fh = clip(fbox); vx, vy, vw, vh = clip(vbox)
    iw = max(0, min(fx + fw, vx + vw) - max(fx, vx)); ih = max(0, min(fy + fh, vy + vh) - max(fy, vy))
    return min(1.0, iw * ih / max(fw * fh, 1e-6))


NBINS = 20  # must match prepare_data.NBINS


def clip_box(b, W, H):
    x, y, w, h = b
    x1, y1, x2, y2 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    return x1, y1, x2, y2


class Oracle:
    """Forward pre-hook on the VG neck: builds V* for the current image at the P2 resolution and hands it to the
    model in place of the prediction. Letterbox mapping follows Ultralytics' LetterBox (centre padding)."""

    def __init__(self, neck, imgsz):
        self.neck, self.imgsz, self.persons, self.W, self.H = neck, imgsz, [], 1, 1
        self.err_sum = self.err_n = self.occ_sum = self.occ_n = 0.0
        neck.register_forward_pre_hook(self.hook)

    def set_image(self, gtboxes, W, H):
        self.W, self.H, self.persons = W, H, []
        for g in gtboxes:
            if g["tag"] != "person" or g.get("extra", {}).get("ignore", 0) == 1:
                continue
            v = vis_ratio(g["fbox"], g["vbox"], W, H)
            vc = (min(NBINS - 1, max(0, int(np.floor(v * NBINS)))) + 0.5) / NBINS   # bin centre, as in training
            f = clip_box(g["fbox"], W, H); vb = clip_box(g["vbox"], W, H)
            vb = (max(vb[0], f[0]), max(vb[1], f[1]), min(vb[2], f[2]), min(vb[3], f[3]))
            if vb[2] - vb[0] < 1 or vb[3] - vb[1] < 1:
                vb = f
            self.persons.append((f, vb, vc))

    def hook(self, module, args):
        import torch
        x0 = args[0][0]
        h, w = x0.shape[-2:]
        s = 4                                       # P2 stride
        Hin, Win = h * s, w * s
        r = min(self.imgsz / self.H, self.imgsz / self.W)
        nw, nh = round(self.W * r), round(self.H * r)
        left, top = round((Win - nw) / 2 - 0.1), round((Hin - nh) / 2 - 0.1)
        gx = (torch.arange(w, device=x0.device).float() + 0.5) * s
        gy = (torch.arange(h, device=x0.device).float() + 0.5) * s
        V = torch.ones((1, 1, h, w), device=x0.device)
        full = torch.zeros((h, w), dtype=torch.bool, device=x0.device)
        for f, vb, vc in self.persons:
            m = lambda b: ((gy >= b[1] * r + top) & (gy < b[3] * r + top))[:, None] & \
                          ((gx >= b[0] * r + left) & (gx < b[2] * r + left))[None, :]  # noqa: E731
            V[0, 0] = torch.where(m(vb), torch.minimum(V[0, 0], torch.tensor(vc, device=x0.device)), V[0, 0])
            full |= m(f)
        self.full = full
        module.oracle = V.to(x0.dtype)

    def record(self):
        """Compare the model's own prediction (still computed, then replaced) with V* inside person boxes."""
        import torch
        pred = torch.sigmoid(self.neck.cache["logit"].float())[0, 0]
        tgt = self.neck.oracle.float()[0, 0]
        if self.full.any():
            d = (pred - tgt).abs()
            self.err_sum += float(d[self.full].sum()); self.err_n += float(self.full.sum())
            occ = self.full & (tgt < 0.65)
            self.occ_sum += float(d[occ].sum()); self.occ_n += float(occ.sum())


def in_subset(v, subset):
    return {"light": v >= 0.65, "heavy": 0.3 <= v < 0.65, "severe": v < 0.3}.get(subset, True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True); ap.add_argument("--images", required=True)
    ap.add_argument("--gt", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--imgsz", type=int, default=640); ap.add_argument("--max-det", type=int, default=500)
    ap.add_argument("--conf", type=float, default=0.001); ap.add_argument("--device", default=None)
    ap.add_argument("--crowddet", default=None); ap.add_argument("--subset", default=None)
    ap.add_argument("--oracle", action="store_true", help="feed ground-truth visibility V* (VG models only)")
    ap.add_argument("--only-existing", action="store_true", help="evaluate only images present in --images (samples)")
    ap.add_argument("--nms-free", action="store_true", help="use the one-to-one head (NMS-free output, nms=False); "
                    "default = Ultralytics default = one-to-many head + NMS")
    a = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).parent))
    import vgyolo  # noqa: F401  (registers the VG classes so the checkpoint can be unpickled)
    from ultralytics import YOLO

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    model = YOLO(a.weights)
    gts = [json.loads(l) for l in open(a.gt)]
    if a.only_existing:
        gts = [r for r in gts if (Path(a.images) / f"{r['ID']}.jpg").exists()]
    oracle = None
    if a.oracle:
        # the predictor works on its own copy of the network: build it once, then hook that copy
        first = next(Path(a.images) / f"{r['ID']}.jpg" for r in gts)
        model.predict(str(first), imgsz=a.imgsz, device=a.device, verbose=False)
        net = model.predictor.model.model
        neck = getattr(net.model[-1], "vg", None)
        assert neck is not None and neck.use_vpb, "--oracle needs a VG model with a visibility branch"
        oracle = Oracle(neck, a.imgsz)
    dt_lines, gt_lines = [], []
    for r in gts:
        img = Path(a.images) / f"{r['ID']}.jpg"
        W, H = Image.open(img).size
        if oracle:
            oracle.set_image(r["gtboxes"], W, H)
        res = model.predict(str(img), imgsz=a.imgsz, conf=a.conf, max_det=a.max_det, device=a.device, verbose=False,
                            **({"nms": False} if a.nms_free else {}))[0]
        if oracle:
            oracle.record()
        b = res.boxes
        xyxy = b.xyxy.cpu().numpy() if len(b) else np.zeros((0, 4))
        sc = b.conf.cpu().numpy() if len(b) else np.zeros(0)
        dts = [{"box": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)], "score": float(s), "tag": 1}
               for (x1, y1, x2, y2), s in zip(xyxy, sc)]
        gtb, gt_rec = [], dict(r)
        for g in r["gtboxes"]:
            ign = g["tag"] != "person" or g.get("extra", {}).get("ignore", 0) == 1
            if not ign and a.subset:
                ign = not in_subset(vis_ratio(g["fbox"], g["vbox"], W, H), a.subset)
            gtb.append({"box": g["fbox"], "tag": -1 if ign else 1})
            if a.subset and ign and g["tag"] == "person":
                g = dict(g); g["extra"] = dict(g.get("extra", {}), ignore=1)
            gt_rec.setdefault("_g", []).append(g)
        gt_rec["gtboxes"] = gt_rec.pop("_g", [])
        dt_lines.append(json.dumps({"ID": r["ID"], "height": H, "width": W, "dtboxes": dts, "gtboxes": gtb}))
        gt_lines.append(json.dumps(gt_rec))
    (out / "dt.odgt").write_text("\n".join(dt_lines) + "\n")
    (out / "gt_used.odgt").write_text("\n".join(gt_lines) + "\n")
    print(f"wrote {len(dt_lines)} images to {out/'dt.odgt'}")
    if oracle:
        diag = dict(vis_mae_person=oracle.err_sum / max(oracle.err_n, 1), vis_mae_occluded=oracle.occ_sum / max(oracle.occ_n, 1))
        (out / "oracle_diag.json").write_text(json.dumps(diag, indent=2))
        print("visibility guess error (0 = perfect):", {k: round(v, 4) for k, v in diag.items()})

    if a.crowddet:
        lib = Path(a.crowddet) / "lib"
        code = (
            "import json,sys;sys.path.insert(0,'.');"
            "from evaluate import compute_JI, compute_APMR;"
            f"AP,MR=compute_APMR.compute_APMR(r'{(out/'dt.odgt').resolve()}', r'{(out/'gt_used.odgt').resolve()}', 'box');"
            f"_,JI=compute_JI.evaluation_all(r'{(out/'dt.odgt').resolve()}', 'box');"
            f"json.dump(dict(AP=AP,MR=MR,JI=JI),open(r'{(out/'metrics.json').resolve()}','w'),indent=2);"
            "print('AP %.4f  MR %.4f  JI %.4f'%(AP,MR,JI))"
        )
        subprocess.run([sys.executable, "-c", code], cwd=lib, check=True)


if __name__ == "__main__":
    main()
