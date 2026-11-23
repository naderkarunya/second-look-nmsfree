"""
vgyolo.py — Visibility-Guided YOLO (VG-YOLO) on top of Ultralytics (tested with ultralytics 8.4.x).

Adds to any Ultralytics detection model (YOLO26-P2, YOLOv8-P2, ...), without changing backbone/neck/head:
  * VPB   — Visibility Prediction Branch on the stride-4 neck feature N2      (feature-level visibility variant)
  * VGFFM — Visibility-Guided Feature Fusion Module before the detection head (feature-level visibility variant)
  * a joint loss  L = L_det + lambda_vis * L_vis                            

Labels come from prepare_data.py (YOLO pose format: class field = visibility bin, two "keypoints" =
visible-box corners). The trainer turns the class field into (class 0, visibility v) after augmentation.

Modes (feature-level visibility variants, pilot protocol):
  "baseline" : no extra modules (still reads the same labels, so data handling is identical)
  "vpb"      : VPB + visibility loss only, fusion unchanged                  (variant B)
  "asff"     : same fusion module, no visibility input, no prior, no VPB    (variant C)
  "vis_off"  : VGFFM + VPB, but V is replaced by 1 everywhere inside VGFFM  (variant D)
  "no_prior" : VGFFM + VPB with gamma fixed to 0                            (variant E)
  "vg"       : full model                                                   (variant F)
Oracle (variant G) is an evaluation-time option: set_oracle(model, V_star) before a forward pass.
  "vg_oracle": same network as "vg", but during TRAINING the fusion receives the ground-truth map V* instead of
               the prediction (the VPB is still trained with L_vis). Upper-bound diagnostic; evaluate it with
               export_eval.py --oracle.
  "vg_sched" : same network as "vg"; training starts with the ground-truth V* and hands over gradually to the
               model's own prediction (scheduled teacher forcing): each image uses V* with probability p, where
               p = 1 for the first 20% of epochs, falls linearly to 0 at 70%, and stays 0 afterwards.
               At test time only the prediction is used, so its numbers are fair.
"""
from __future__ import annotations

import math
from copy import copy, deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.data.dataset import YOLODataset
from ultralytics.data.utils import get_split_fraction  # noqa: F401  (kept for API parity)
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator
from ultralytics.nn.modules import Conv, Detect
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import colorstr
from ultralytics.utils.torch_utils import unwrap_model
from ultralytics.utils.loss import E2ELoss, v8DetectionLoss
from ultralytics.utils.tal import make_anchors, TaskAlignedAssigner
from functools import partial

# ----------------------------------------------------------------------------- configuration
VG_CFG = dict(
    mode="vg",          # baseline | vpb | asff | vis_off | no_prior | vg
    vis_bins=20,        # must match prepare_data.NBINS
    lambda_vis=1.0,     # Eq. (14)
    lambda_bg=0.1,      # Eq. (3)
    r=8,                # descriptor channels, Eq. (8)
    cm=64,              # common width of the aligned candidates (keeps VGFFM light)
    c_vpb=32,           # hidden width of the VPB
    rho=(1.0, 1 / 3, -1 / 3, -1.0),  # level prior, Eq. (9), shallow -> deep
    gamma_init=1.0,
    vis_weight=0.0,     # Option B: occlusion-aware loss weighting, beta in w = 1 + beta * (1 - v); 0 = off
    vis_weight_norm=True,
    vra="none",
    crm=False,          # Crowd Relation Module on the one-to-one (NMS-free) branch
    vba=False,          # Visible-Body Attention on the neck outputs (feeds BOTH heads)
    vba_r=4, lambda_vba=1.0,
    hg=False,           # Head-Guided module (head-centre heatmap on P2 guiding the one-to-one branch)
    hg_dim=64, lambda_head=1.0,
    crm_dim=64, crm_heads=4, crm_win=7,         # visible-region assignment: none | o2o (one-to-one branch only) | all (both branches)  # rescale the weights of positives to mean 1 (False: visible people keep weight 1)
)


# ----------------------------------------------------------------------------- modules
class VPB(nn.Module):
    """Visibility Prediction Branch, Eq. (4): two 3x3 Conv-BN-SiLU + 1x1 conv -> logits (sigmoid gives V)."""

    def __init__(self, c: int, ch: int = 32):
        super().__init__()
        self.body = nn.Sequential(Conv(c, ch, 3), Conv(ch, ch, 3))
        self.out = nn.Conv2d(ch, 1, 1)
        nn.init.constant_(self.out.bias, 2.0)  # start near V = 0.88 (most pixels are background, target 1)

    def forward(self, x):
        return self.out(self.body(x))


class LevelFusion(nn.Module):
    """VGFFM for one output level l (Eqs. 7–11)."""

    def __init__(self, chs, l, r, use_vis, use_prior, rho, gamma_init, cm=64):
        super().__init__()
        self.l, self.n = l, len(chs)
        cl = chs[l]
        # every level (including l itself) is projected to a common width cm (Eq. 7)
        self.align = nn.ModuleList([Conv(chs[k], cm, 1) for k in range(self.n)])
        self.desc = nn.ModuleList([nn.Conv2d(cm, r, 1) for _ in range(self.n)])
        self.use_vis, self.use_prior = use_vis, use_prior
        self.g = nn.Sequential(nn.Conv2d(self.n * r + int(use_vis), 16, 1), nn.SiLU(), nn.Conv2d(16, self.n, 1))
        # E: depth-wise 3x3 on the fused map, then a 1x1 back to c_l (Eq. 11)
        self.refine = nn.Sequential(Conv(cm, cm, 3, g=cm), Conv(cm, cl, 1, act=False))
        nn.init.zeros_(self.refine[1].bn.weight)  # E starts at 0  ->  F_l = N_l at initialisation
        self.register_buffer("rho", torch.tensor(rho[: self.n], dtype=torch.float32).view(1, -1, 1, 1))
        # gamma = softplus(gamma_bias) >= 0, initialised to gamma_init ("bias" in the name keeps it out of weight decay)
        self.gamma_bias = nn.Parameter(torch.tensor(math.log(math.expm1(gamma_init)) if gamma_init > 0 else -20.0))

    def gamma(self):
        return F.softplus(self.gamma_bias)

    def forward(self, xs, V_l):
        l = self.l
        h, w = xs[l].shape[-2:]
        cands = []
        for k, x in enumerate(xs):
            if k == l:
                c = self.align[k](x)
            elif k > l:  # deeper, smaller map: 1x1 conv then nearest upsample (the two commute)
                c = F.interpolate(self.align[k](x), size=(h, w), mode="nearest")
            else:  # shallower, larger map: max-pool down, then 1x1 conv
                c = self.align[k](F.adaptive_max_pool2d(x, (h, w)))
            cands.append(c)
        z = [self.desc[k](c) for k, c in enumerate(cands)]
        if self.use_vis:
            z.append(V_l)
        logits = self.g(torch.cat(z, 1))                                   # Eq. (8)
        if self.use_prior and V_l is not None:
            logits = logits + self.gamma() * (1.0 - V_l) * self.rho       # Eq. (9) numerator term
        alpha = logits.softmax(1)                                          # Eq. (9)
        fused = sum(alpha[:, k : k + 1] * cands[k] for k in range(self.n))  # Eq. (10)
        self.last_alpha = alpha.detach()
        return xs[l] + self.refine(fused)                                  # Eq. (11)


class VGNeck(nn.Module):
    """Holds VPB + one LevelFusion per detection level. Inputs are the neck outputs fed to Detect (shallow -> deep)."""

    def __init__(self, chs, mode, cfg):
        super().__init__()
        self.mode = mode
        self.use_vpb = mode in {"vpb", "vis_off", "no_prior", "vg"}
        self.use_fusion = mode in {"asff", "vis_off", "no_prior", "vg"}
        use_vis = mode in {"vis_off", "no_prior", "vg"}
        use_prior = mode in {"vis_off", "vg"}
        self.vpb = VPB(chs[0], cfg["c_vpb"]) if self.use_vpb else None
        self.fuse = (
            nn.ModuleList(
                [LevelFusion(chs, l, cfg["r"], use_vis, use_prior, cfg["rho"], cfg["gamma_init"], cfg["cm"])
                 for l in range(len(chs))]
            )
            if self.use_fusion
            else None
        )
        self.cache = {}
        self.oracle = None  # optional ground-truth V* for the oracle experiment (evaluation only)

    def forward(self, xs):
        xs = list(xs)
        V = None
        if self.use_vpb:
            logit = self.vpb(xs[0])
            self.cache["logit"] = logit
            V = torch.sigmoid(logit)
            if self.oracle is not None:
                mask = getattr(self, "oracle_mask", None)   # per-image choice (scheduled hand-over), else all images
                V = self.oracle.to(V) if mask is None else torch.where(mask.to(V.device), self.oracle.to(V), V)
            self.cache["V"] = V.detach()
        if self.use_fusion:
            outs = []
            for l, f in enumerate(self.fuse):
                if f.use_vis:
                    Vl = F.adaptive_avg_pool2d(V, xs[l].shape[-2:])  # Eq. (5): average pooling
                    if self.mode == "vis_off":
                        Vl = torch.ones_like(Vl)
                else:
                    Vl = None
                outs.append(f(xs, Vl))
            xs = outs
        return xs


class CrowdRelationModule(nn.Module):
    """Crowd Relation Module (CRM) for one level of the one-to-one (NMS-free) branch.

    Each location compares itself with its neighbours inside a local k x k window (multi-head self-attention with a
    learned relative-position bias), so candidates of the same person and of different, adjacent persons can be told
    apart before the one-to-one classification/regression. Reduce (1x1) -> window attention -> depth-wise conv FFN ->
    expand (1x1, zero-initialised, so the block starts as the identity) -> residual."""

    def __init__(self, c, d=64, heads=4, win=7):
        super().__init__()
        self.d, self.h, self.win = d, heads, win
        self.reduce = Conv(c, d, 1)
        self.qkv = nn.Conv2d(d, 3 * d, 1)
        self.proj = nn.Conv2d(d, d, 1)
        self.rpb = nn.Parameter(torch.zeros(heads, (2 * win - 1) ** 2))       # relative position bias
        nn.init.trunc_normal_(self.rpb, std=0.02)
        yy, xx = torch.meshgrid(torch.arange(win), torch.arange(win), indexing="ij")
        pts = torch.stack((yy.flatten(), xx.flatten()), 1)
        rel = pts[:, None] - pts[None] + win - 1
        self.register_buffer("rp_index", rel[..., 0] * (2 * win - 1) + rel[..., 1], persistent=False)
        self.ffn = nn.Sequential(Conv(d, d, 3, g=d), Conv(d, d, 1))
        self.expand = Conv(d, c, 1, act=False)
        nn.init.zeros_(self.expand.bn.weight)

    def attn(self, y):
        B, C, H, W = y.shape
        k = self.win
        ph, pw = (k - H % k) % k, (k - W % k) % k
        yp = F.pad(y, (0, pw, 0, ph))
        Hp, Wp = H + ph, W + pw
        q, kk, v = self.qkv(yp).chunk(3, 1)
        def win_split(t):  # (B,C,Hp,Wp) -> (B*nW, heads, k*k, C/heads)
            t = t.view(B, self.h, C // self.h, Hp // k, k, Wp // k, k)
            return t.permute(0, 3, 5, 1, 4, 6, 2).reshape(-1, self.h, k * k, C // self.h)
        q, kk, v = win_split(q), win_split(kk), win_split(v)
        a = (q * (C // self.h) ** -0.5) @ kk.transpose(-2, -1) + self.rpb[:, self.rp_index]
        # padded positions must not be attended to
        if ph or pw:
            m = F.pad(torch.zeros(1, 1, H, W, device=y.device), (0, pw, 0, ph), value=1.0)
            m = m.view(1, Hp // k, k, Wp // k, k).permute(0, 1, 3, 2, 4).reshape(-1, 1, 1, k * k)
            a = a.masked_fill(m.repeat(B, 1, 1, 1).bool(), float("-inf"))
        o = a.softmax(-1) @ v                                                  # (B*nW, heads, k*k, C/heads)
        o = o.view(B, Hp // k, Wp // k, self.h, k, k, C // self.h).permute(0, 3, 6, 1, 4, 2, 5)
        o = o.reshape(B, C, Hp, Wp)[..., :H, :W]
        return self.proj(o)

    def forward(self, x):
        y = self.reduce(x)
        y = y + self.attn(y)
        y = y + self.ffn(y)
        return x + self.expand(y)


class VisibleBodyAttention(nn.Module):
    """Visible-Body Attention (VBA) on every neck output level (P2-P5), before BOTH detection heads.

    A light branch per level (1x1 reduce -> depth-wise 3x3 -> 1x1) predicts a visible-body map M_l, supervised by the
    union of the visible-body boxes. The features are re-weighted as  F'_l = F_l * (1 + g_l * (2 M_l - 1)),
    boosting visible body parts and damping regions where a person is hidden behind someone else. The gate g_l is a
    learnable scalar initialised to 0, so the block starts as the identity while the map is already being learned
    from its own loss."""

    def __init__(self, chs, r=4):
        super().__init__()
        self.branch = nn.ModuleList(
            nn.Sequential(Conv(c, max(c // r, 16), 1), Conv(max(c // r, 16), max(c // r, 16), 3, g=max(c // r, 16)),
                          nn.Conv2d(max(c // r, 16), 1, 1)) for c in chs)
        self.gate_bias = nn.Parameter(torch.zeros(len(chs)))     # "bias" in the name keeps it out of weight decay
        self.cache = {}

    def forward(self, xs):
        logits = [b(x) for b, x in zip(self.branch, xs)]
        self.cache["logits"] = logits
        return [x * (1 + self.gate_bias[i] * (2 * torch.sigmoid(lg) - 1)) for i, (x, lg) in enumerate(zip(xs, logits))]


@torch.no_grad()
def visible_body_target(batch, shape, lambda_bg=0.1):
    """Binary visible-body target (union of visible boxes) and weight map (1 inside full boxes, lambda_bg elsewhere)
    at the resolution of `shape` (B,1,h,w). Visible box = keypoints 1-2 (min/max, flip-safe)."""
    B, _, h, w = shape
    dev = batch["img"].device
    T = torch.zeros((B, 1, h, w), device=dev)
    Wm = torch.full((B, 1, h, w), float(lambda_bg), device=dev)
    if batch["bboxes"].numel() == 0:
        return T, Wm
    xywh = batch["bboxes"].to(dev).float()
    fx1, fy1, fx2, fy2 = xywh[:, 0] - xywh[:, 2] / 2, xywh[:, 1] - xywh[:, 3] / 2, xywh[:, 0] + xywh[:, 2] / 2, xywh[:, 1] + xywh[:, 3] / 2
    kp = batch.get("keypoints")
    if kp is not None and kp.numel():
        kp = kp[:, :2, :2].to(dev).float()
        vx1, vx2 = torch.maximum(kp[:, :, 0].min(1).values, fx1), torch.minimum(kp[:, :, 0].max(1).values, fx2)
        vy1, vy2 = torch.maximum(kp[:, :, 1].min(1).values, fy1), torch.minimum(kp[:, :, 1].max(1).values, fy2)
        bad = (vx2 - vx1 < 1e-4) | (vy2 - vy1 < 1e-4)
        vx1, vy1, vx2, vy2 = [torch.where(bad, a, b) for a, b in ((fx1, vx1), (fy1, vy1), (fx2, vx2), (fy2, vy2))]
    else:
        vx1, vy1, vx2, vy2 = fx1, fy1, fx2, fy2
    gy = (torch.arange(h, device=dev).float() + 0.5) / h
    gx = (torch.arange(w, device=dev).float() + 0.5) / w
    bi = batch["batch_idx"].to(dev).long()
    for b in range(B):
        idx = (bi == b).nonzero().flatten()
        if idx.numel() == 0:
            continue
        inx = lambda a1, a2, g: (g[None] >= a1[:, None]) & (g[None] < a2[:, None])  # noqa: E731
        vis = (inx(vy1[idx], vy2[idx], gy)[:, :, None] & inx(vx1[idx], vx2[idx], gx)[:, None, :]).any(0)
        full = (inx(fy1[idx], fy2[idx], gy)[:, :, None] & inx(fx1[idx], fx2[idx], gx)[:, None, :]).any(0)
        T[b, 0] = vis.float()
        Wm[b, 0] = torch.where(full | vis, torch.ones_like(Wm[b, 0]), Wm[b, 0])
    return T, Wm


class HeadGuide(nn.Module):
    """Head-Guided module (HG).

    A light branch on the finest level (P2) predicts a head-centre heatmap (one peak per person, supervised by the
    CrowdHuman head boxes). Its features and heatmap are pooled to every detection level and added, through a
    zero-initialised 1x1 projection, to the features entering the one-to-one (NMS-free) branch. Heads are rarely
    occluded and rarely overlap, so the one-to-one branch receives an explicit cue of how many people are present
    in a crowded region. The heatmap branch is trained by its own loss; its outputs are detached before fusion, so
    the one-to-one loss trains only the fusion projections."""

    def __init__(self, chs, d=64):
        super().__init__()
        self.stem = nn.Sequential(Conv(chs[0], d, 3), Conv(d, d, 3))
        self.hm = nn.Conv2d(d, 1, 1)
        nn.init.constant_(self.hm.bias, -2.19)                                   # prior 0.1
        self.fuse = nn.ModuleList(Conv(d + 1, c, 1, act=False) for c in chs)
        for f in self.fuse:
            nn.init.zeros_(f.bn.weight)                                           # starts as identity
        self.cache = {}

    def forward(self, x, xo):
        f = self.stem(x[0])
        logit = self.hm(f)
        self.cache["logit"] = logit
        g = torch.cat((f, logit.sigmoid()), 1)
        if self.training:
            g = g.detach()
        out = []
        for l, (xi, fu) in enumerate(zip(xo, self.fuse)):
            gl = g if l == 0 else F.max_pool2d(g, 2 ** l, 2 ** l, ceil_mode=True)[..., : xi.shape[-2], : xi.shape[-1]]
            out.append(xi + fu(gl))
        return out


class VGDetect(Detect):
    """Detect head with optional VGNeck on its inputs and optional CRM in front of the one-to-one branch."""

    def forward(self, x):
        if hasattr(self, "vg"):
            x = self.vg(x)
        if hasattr(self, "vba"):
            x = self.vba(x)
        if not (hasattr(self, "crm") or hasattr(self, "hg")) or getattr(self, "one2one_cv2", None) is None:
            return super().forward(x)
        preds = self.forward_head(x, **self.one2many)
        xo = [xi.detach() for xi in x] if self.training else list(x)
        if hasattr(self, "hg"):
            xo = self.hg(x, xo)
        if hasattr(self, "crm"):
            xo = [m(xi) for m, xi in zip(self.crm, xo)]
        one2one = self.forward_head(xo, **self.one2one)
        preds = {"one2many": preds, "one2one": one2one}
        if self.training:
            return preds
        y = self._inference(preds["one2one" if self.end2end else "one2many"])
        if self.end2end:
            y = self.postprocess(y.permute(0, 2, 1))
        return y if self.export else (y, preds)


class VGLoss:
    """Wraps the base detection loss and adds the visibility loss (Eqs. 6 and 14)."""

    def __init__(self, model, base, cfg):
        self.model, self.base, self.cfg = model, base, cfg

    def __getattr__(self, k):  # pass 'updates', 'update', 'hyp', ... through to the base loss
        if k.startswith("__"):
            raise AttributeError(k)
        return getattr(object.__getattribute__(self, "base"), k)

    def __setattr__(self, k, v):
        if k in {"model", "base", "cfg"}:
            object.__setattr__(self, k, v)
        else:
            setattr(self.base, k, v)

    def __call__(self, preds, batch):
        total, items = self.base(preds, batch)
        det = self.model.model[-1]
        bs = batch["img"].shape[0]
        vg = getattr(det, "vg", None)
        if vg is not None and vg.use_vpb and "logit" in vg.cache and "vis" in batch:
            logit = vg.cache["logit"].float()
            target, weight = visibility_target(batch, logit.shape, self.cfg)
            bce = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
            lv = (bce * weight).sum() / weight.sum().clamp(min=1.0)
            items = dict(items)
            items["vis_loss"] = lv.detach()
            total = total + self.cfg["lambda_vis"] * lv * bs
        vba = getattr(det, "vba", None)
        if vba is not None and "logits" in vba.cache:
            lv, n = 0.0, 0
            for lg in vba.cache["logits"]:
                lg = lg.float()
                T, Wm = visible_body_target(batch, lg.shape, self.cfg["lambda_bg"])
                bce = F.binary_cross_entropy_with_logits(lg, T, reduction="none")
                lv = lv + (bce * Wm).sum() / Wm.sum().clamp(min=1.0)
                n += 1
            lv = lv / max(n, 1)
            items = dict(items)
            items["vba_loss"] = lv.detach()
            total = total + self.cfg["lambda_vba"] * lv * bs
        hg = getattr(det, "hg", None)
        if hg is not None and "logit" in hg.cache:
            logit = hg.cache["logit"].float()
            target = head_heatmap_target(batch, logit.shape)
            lh = heatmap_focal_loss(logit, target)
            items = dict(items)
            items["head_loss"] = lh.detach()
            total = total + self.cfg["lambda_head"] * lh * bs
        return total, items


@torch.no_grad()
def head_heatmap_target(batch, shape):
    """Gaussian head-centre heatmap (CenterNet style) at the resolution of `shape` (B,1,h,w).
    Head boxes are keypoints 3 and 4 (normalised corners, visibility flag > 0)."""
    B, _, h, w = shape
    kp = batch.get("keypoints")
    dev = batch["img"].device
    T = torch.zeros((B, 1, h, w), device=dev)
    if kp is None or kp.numel() == 0 or kp.shape[1] < 4:
        return T
    kp = kp.to(dev).float()
    ok = (kp[:, 2, 2] > 0) | (kp[:, 3, 2] > 0)          # keep heads cut by the image border (one corner inside)
    # corners may swap under horizontal flip, so take min / max
    x1, x2 = torch.minimum(kp[:, 2, 0], kp[:, 3, 0]) * w, torch.maximum(kp[:, 2, 0], kp[:, 3, 0]) * w
    y1, y2 = torch.minimum(kp[:, 2, 1], kp[:, 3, 1]) * h, torch.maximum(kp[:, 2, 1], kp[:, 3, 1]) * h
    ok &= (x2 - x1 > 0) & (y2 - y1 > 0) & (x2 > 0) & (y2 > 0) & (x1 < w) & (y1 < h)
    cx, cy = ((x1 + x2) / 2).clamp(0, w - 1e-3), ((y1 + y2) / 2).clamp(0, h - 1e-3)
    sig = (torch.sqrt((x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)) / 6).clamp(min=0.8)
    bi = batch["batch_idx"].to(dev).long()
    gy = torch.arange(h, device=dev).float() + 0.5
    gx = torch.arange(w, device=dev).float() + 0.5
    for b in range(B):
        idx = ((bi == b) & ok).nonzero().flatten()
        if idx.numel() == 0:
            continue
        g = torch.exp(-((gx[None, None, :] - cx[idx, None, None]) ** 2 + (gy[None, :, None] - cy[idx, None, None]) ** 2)
                      / (2 * sig[idx, None, None] ** 2))                                     # (n,h,w)
        T[b, 0] = g.amax(0)
        T[b, 0, cy[idx].long(), cx[idx].long()] = 1.0                                           # exact peaks
    return T


def heatmap_focal_loss(logit, target, alpha=2.0, beta=4.0):
    """Penalty-reduced focal loss of CenterNet, normalised by the number of heads."""
    p = logit.sigmoid().clamp(1e-4, 1 - 1e-4)
    pos = target.eq(1).float()
    pos_loss = torch.log(p) * (1 - p) ** alpha * pos
    neg_loss = torch.log(1 - p) * p ** alpha * (1 - target) ** beta * (1 - pos)
    n = pos.sum().clamp(min=1.0)
    return -(pos_loss.sum() + neg_loss.sum()) / n


class VisibleRegionAssigner(TaskAlignedAssigner):
    """Task-aligned assigner whose candidate anchors must lie inside the person's VISIBLE box (not the full box).
    The alignment metric and the regression target still use the full box. Set `cand_boxes` before each call."""
    cand_boxes = None

    def select_candidates_in_gts(self, xy_centers, gt_bboxes, mask_gt, eps=1e-9):
        boxes = self.cand_boxes if self.cand_boxes is not None and self.cand_boxes.shape == gt_bboxes.shape else gt_bboxes
        return super().select_candidates_in_gts(xy_centers, boxes, mask_gt, eps)


class VisWeightedDetLoss(v8DetectionLoss):
    """Option B: occlusion-aware loss weighting.

    Every anchor assigned to a ground-truth person gets the weight w = 1 + beta * (1 - v), where v is that person's
    visibility (0 = fully hidden, 1 = fully visible). The weights of the positive anchors in a batch are rescaled to
    mean 1, so the overall loss scale stays the same as the baseline and only the balance between visible and hidden
    people changes. Applied to the classification and the box losses. Nothing changes at test time."""

    def __init__(self, model, tal_topk=10, tal_topk2=None, beta=1.0, bins=20, norm=True, vra="none"):
        super().__init__(model, tal_topk, tal_topk2)
        self.beta, self.bins, self.norm = beta, bins, norm
        # visible-region assignment on this branch? (the one-to-one branch is the one built with tal_topk2 == 1)
        self.vra = vra == "all" or (vra == "o2o" and tal_topk2 == 1)
        if self.vra:
            self.assigner.__class__ = VisibleRegionAssigner

    def get_assigned_targets_and_loss(self, preds, batch):
        if "vis" not in batch or (self.beta <= 0 and not self.vra):
            return super().get_assigned_targets_and_loss(preds, batch)
        loss = torch.zeros(3, device=self.device)
        pred_distri = preds["boxes"].permute(0, 2, 1).contiguous()
        pred_scores = preds["scores"].permute(0, 2, 1).contiguous()
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)
        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        v = (batch["vis"].view(-1, 1).float() + 0.5) / self.bins                       # bin centre -> visibility
        kp = batch.get("keypoints")
        if kp is not None and kp.numel():                                              # visible box (normalised xyxy)
            kp = kp[:, :2, :2].float().to(v.device)          # first two keypoints = visible-box corners
            vb = torch.stack((kp[:, :, 0].min(1).values, kp[:, :, 1].min(1).values,
                              kp[:, :, 0].max(1).values, kp[:, :, 1].max(1).values), 1)
        else:
            vb = torch.zeros((v.shape[0], 4), device=v.device)
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"], v, vb), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes, gt_vis, gt_vbox = targets.split((1, 4, 1, 4), 2)
        if self.vra:
            vbox = gt_vbox * imgsz[[1, 0, 1, 0]]
            vbox = torch.cat((torch.maximum(vbox[..., :2], gt_bboxes[..., :2]), torch.minimum(vbox[..., 2:], gt_bboxes[..., 2:])), -1)
            bad = ((vbox[..., 2] - vbox[..., 0]) < 1) | ((vbox[..., 3] - vbox[..., 1]) < 1)
            self.assigner.cand_boxes = torch.where(bad.unsqueeze(-1), gt_bboxes, vbox)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)
        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)
        target_scores_sum = target_scores.sum().clamp_(min=1)

        # per-anchor weight from the visibility of the assigned person (positives only), mean 1 over positives
        v_anchor = gt_vis.squeeze(-1).gather(1, target_gt_idx.long()) if gt_vis.shape[1] else torch.ones_like(fg_mask, dtype=torch.float)
        w = 1.0 + self.beta * (1.0 - v_anchor.clamp(0, 1))
        fgf = fg_mask.float()
        if self.norm:
            w = w / ((w * fgf).sum() / fgf.sum().clamp(min=1))
        w = torch.where(fg_mask, w, torch.ones_like(w))

        bce_loss = self.bce(pred_scores, target_scores.to(dtype)) * w.unsqueeze(-1).to(dtype)
        loss[1] = bce_loss.sum() / target_scores_sum
        loss[0], loss[2] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes / stride_tensor,
                                          target_scores * w.unsqueeze(-1), target_scores_sum, fg_mask, imgsz,
                                          stride_tensor)
        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl
        return ((fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor), loss,
                dict(zip(self.loss_names, loss.detach())))


class VGDetectionModel(DetectionModel):
    """DetectionModel whose criterion includes the visibility loss."""

    def init_criterion(self):
        beta, vra = VG_CFG.get("vis_weight", 0.0), VG_CFG.get("vra", "none")
        if beta > 0 or vra != "none":
            fn = partial(VisWeightedDetLoss, beta=beta, bins=VG_CFG["vis_bins"], norm=VG_CFG.get("vis_weight_norm", True),
                         vra=vra)
            base = E2ELoss(self, loss_fn=fn) if getattr(self.model[-1], "one2one_cv2", None) is not None else fn(self)
        else:
            base = super().init_criterion()
        return VGLoss(self, base, VG_CFG)


# ----------------------------------------------------------------------------- targets
@torch.no_grad()
def visibility_target(batch, shape, cfg):
    """Dense target V* (Eq. 2) and weight map m (Eq. 3) at the resolution of `shape` (B,1,h,w).

    Uses the boxes *after* augmentation: full box from batch['bboxes'] (normalised xywh), visible box from the two
    keypoints (normalised corners), visibility from batch['vis'] (bin index)."""
    B, _, h, w = shape
    dev = batch["bboxes"].device
    V = torch.ones((B, 1, h, w), device=dev)
    M = torch.full((B, 1, h, w), float(cfg["lambda_bg"]), device=dev)
    if batch["bboxes"].numel() == 0:
        return V, M
    gy = (torch.arange(h, device=dev).float() + 0.5) / h
    gx = (torch.arange(w, device=dev).float() + 0.5) / w
    bi = batch["batch_idx"].long()
    xywh = batch["bboxes"].float()
    fx1, fy1 = xywh[:, 0] - xywh[:, 2] / 2, xywh[:, 1] - xywh[:, 3] / 2
    fx2, fy2 = xywh[:, 0] + xywh[:, 2] / 2, xywh[:, 1] + xywh[:, 3] / 2
    v = (batch["vis"].view(-1).float() + 0.5) / cfg["vis_bins"]
    if "keypoints" in batch and batch["keypoints"] is not None and batch["keypoints"].numel():
        kp = batch["keypoints"][:, :2, :2].float()          # first two keypoints = visible-box corners
        vx1, vx2 = kp[:, :, 0].min(1).values, kp[:, :, 0].max(1).values
        vy1, vy2 = kp[:, :, 1].min(1).values, kp[:, :, 1].max(1).values
        # keep the visible box inside the (augmented) full box
        vx1, vy1 = torch.maximum(vx1, fx1), torch.maximum(vy1, fy1)
        vx2, vy2 = torch.minimum(vx2, fx2), torch.minimum(vy2, fy2)
        bad = (vx2 - vx1 < 1e-4) | (vy2 - vy1 < 1e-4)
        vx1, vy1, vx2, vy2 = [torch.where(bad, a, b) for a, b in ((fx1, vx1), (fy1, vy1), (fx2, vx2), (fy2, vy2))]
    else:
        vx1, vy1, vx2, vy2 = fx1, fy1, fx2, fy2
    for b in range(B):
        idx = (bi == b).nonzero().flatten()
        if idx.numel() == 0:
            continue
        inx = lambda a1, a2, g: (g[None] >= a1[:, None]) & (g[None] < a2[:, None])  # noqa: E731
        vis_mask = inx(vy1[idx], vy2[idx], gy)[:, :, None] & inx(vx1[idx], vx2[idx], gx)[:, None, :]  # (n,h,w)
        full_mask = inx(fy1[idx], fy2[idx], gy)[:, :, None] & inx(fx1[idx], fx2[idx], gx)[:, None, :]
        vals = torch.where(vis_mask, v[idx].view(-1, 1, 1), torch.ones_like(vis_mask, dtype=torch.float32))
        V[b, 0] = vals.min(0).values                                          # Eq. (2): min over overlaps
        M[b, 0] = torch.where(full_mask.any(0), torch.ones_like(M[b, 0]), M[b, 0])  # Eq. (3)
    return V, M


def set_oracle(model, V_star):
    """Oracle experiment (variant G): feed ground-truth V* instead of the prediction (evaluation only)."""
    unwrap_model(model).model[-1].vg.oracle = V_star


# ----------------------------------------------------------------------------- model surgery
def attach_vg(model: DetectionModel, mode: str, cfg=VG_CFG):
    """Turn a built Ultralytics DetectionModel into VG-YOLO in place."""
    model.__class__ = VGDetectionModel
    det = model.model[-1]
    assert type(det) in {Detect, VGDetect}, f"expected a Detect head, got {type(det)}"
    det.__class__ = VGDetect
    if mode != "baseline":
        chs = [seq[0].conv.in_channels for seq in det.cv2]
        det.vg = VGNeck(chs, "vg" if mode in {"vg_oracle", "vg_sched"} else mode, cfg).to(next(model.parameters()).device)
    if cfg.get("crm") and getattr(det, "one2one_cv2", None) is not None:
        chs = [seq[0].conv.in_channels for seq in det.cv2]
        det.crm = nn.ModuleList(CrowdRelationModule(c, cfg["crm_dim"], cfg["crm_heads"], cfg["crm_win"]) for c in chs
                                ).to(next(model.parameters()).device)
    if cfg.get("vba"):
        chs = [seq[0].conv.in_channels for seq in det.cv2]
        det.vba = VisibleBodyAttention(chs, cfg["vba_r"]).to(next(model.parameters()).device)
    if cfg.get("hg") and getattr(det, "one2one_cv2", None) is not None:
        chs = [seq[0].conv.in_channels for seq in det.cv2]
        det.hg = HeadGuide(chs, cfg["hg_dim"]).to(next(model.parameters()).device)
    return model


# ----------------------------------------------------------------------------- data
def _vis_data(data, cfg):
    """Dataset view of the data dict: the class field holds visibility bins, so the dataset needs vis_bins names."""
    d = dict(data)
    d["names"] = {i: f"vis{i}" for i in range(cfg["vis_bins"])}
    d["nc"] = cfg["vis_bins"]
    d.setdefault("kpt_shape", [2, 3])
    d.setdefault("flip_idx", [0, 1])
    return d


def build_vg_dataset(args, img_path, batch, data, mode, stride, cfg=VG_CFG):
    a = copy(args)
    a.task = "pose"  # load the two visible-box corners as keypoints
    return YOLODataset(
        img_path=img_path, imgsz=a.imgsz, batch_size=batch, augment=mode == "train", hyp=a,
        rect=a.rect or mode == "val", cache=a.cache or None, single_cls=False, stride=stride,
        pad=0.0 if mode == "train" else 0.5, prefix=colorstr(f"{mode}: "), task="pose", classes=None,
        data=_vis_data(data, cfg), fraction=a.fraction if mode == "train" else 1.0,
    )


def split_vis(batch):
    """Move visibility bins out of the class field; class becomes 0 (single 'person' class)."""
    if "cls" in batch and "vis" not in batch:
        batch["vis"] = batch["cls"].clone()
        batch["cls"] = torch.zeros_like(batch["cls"])
    return batch


def sched_p(epoch, epochs, start=0.2, end=0.7):
    """Probability of feeding ground-truth V* at a given (0-based) epoch for mode vg_sched."""
    f = epoch / max(epochs, 1)
    return 1.0 if f < start else 0.0 if f >= end else 1.0 - (f - start) / (end - start)


class VGValidator(DetectionValidator):
    def build_dataset(self, img_path, mode="val", batch=None):
        return build_vg_dataset(self.args, img_path, batch, self.data, mode, stride=self.stride)

    def preprocess(self, batch):
        return split_vis(super().preprocess(batch))


class VGTrainer(DetectionTrainer):
    vg_cfg = VG_CFG

    def get_model(self, cfg=None, weights=None, verbose=True):
        name = str(cfg.get("yaml_file", "") if isinstance(cfg, dict) else cfg)
        if weights is not None and "-p2" in name:  # index-shifted transfer for P2 models (see load_p2_pretrained)
            model = super().get_model(cfg, None, verbose)
            fam = "yolov8" if "yolov8" in name else "yolo26" if "yolo26" in name else None
            if fam:
                n, tot = load_p2_pretrained(model, weights, fam)
                print(f"VG-YOLO: transferred {n}/{tot} items from pretrained weights ({fam} -> P2 mapping)")
            else:
                model.load(weights)
        else:
            model = super().get_model(cfg, weights, verbose)
        return attach_vg(model, self.vg_cfg["mode"], self.vg_cfg)

    def build_dataset(self, img_path, mode="train", batch=None):
        gs = max(int(unwrap_model(self.model).stride.max()), 32)
        return build_vg_dataset(self.args, img_path, batch, self.data, mode, stride=gs, cfg=self.vg_cfg)

    def preprocess_batch(self, batch):
        batch = split_vis(super().preprocess_batch(batch))
        mode = self.vg_cfg["mode"]
        if mode in {"vg_oracle", "vg_sched"}:  # feed ground-truth V* to the fusion for this training step
            neck = unwrap_model(self.model).model[-1].vg
            B, _, H, W = batch["img"].shape
            p = 1.0 if mode == "vg_oracle" else sched_p(self.epoch, self.epochs)
            if p <= 0:
                neck.oracle, neck.oracle_mask = None, None
            else:
                V, _ = visibility_target(batch, (B, 1, H // 4, W // 4), self.vg_cfg)
                neck.oracle = V
                neck.oracle_mask = None if p >= 1 else (torch.rand(B, 1, 1, 1, device=V.device) < p)
        return batch

    def get_validator(self):
        v = VGValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks)
        return v

    def plot_training_labels(self):  # class field holds visibility bins; skip the class histogram
        pass


# ----------------------------------------------------------------------------- pretrained weights for P2 models
_P2_LAYER_MAP = {"yolo26": {17: 23, 19: 25, 20: 26, 22: 28, 23: 29}, "yolov8": {16: 22, 18: 24, 19: 25, 21: 27, 22: 28}}


def load_p2_pretrained(model, weights, family="yolo26"):
    """Transfer COCO weights of a 3-level model (e.g. yolo26s.pt) into the 4-level P2 model.

    Ultralytics' default transfer matches layers by index, which breaks after the extra P2 layers are inserted
    (only the backbone and the first neck layers transfer). Here the shifted neck layers are re-indexed and the
    P3–P5 detection branches are moved to head positions 1–3; everything is then copied where shapes agree."""
    if isinstance(weights, (str, bytes)) or hasattr(weights, "__fspath__"):
        ck = torch.load(weights, map_location="cpu", weights_only=False)
        weights = ck.get("ema") or ck["model"]
    src = weights.float().state_dict() if hasattr(weights, "state_dict") else dict(weights)
    lmap = _P2_LAYER_MAP[family]
    det_src = max(lmap)
    out = {}
    for k, v in src.items():
        parts = k.split(".")
        i = int(parts[1])
        if i in lmap:
            parts[1] = str(lmap[i])
            if i == det_src and len(parts) > 3 and parts[2] in {"cv2", "cv3", "one2one_cv2", "one2one_cv3"}:
                parts[3] = str(int(parts[3]) + 1)  # P3,P4,P5 branches -> positions 1,2,3 (0 is the new P2 branch)
        elif i > min(lmap) - 1 and i not in lmap:
            continue  # layers without a counterpart
        out[".".join(parts)] = v
    dst = model.state_dict()
    ok = {k: v for k, v in out.items() if k in dst and dst[k].shape == v.shape}
    model.load_state_dict(ok, strict=False)
    return len(ok), len(dst)
