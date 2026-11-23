"""numbers5.py — compile every measured number used in Section 5 from the Kaggle summary files into numbers5.json.
Single source of truth for the tables (build5) and the figures (figs5_real)."""
import json, numpy as np
from pathlib import Path
C = Path(__file__).resolve().parent.parent / "results"
r4 = json.load(open(C / "summary_run4.json")); r5 = json.load(open(C / "summary_run5.json")); r6 = json.load(open(C / "summary_run6.json"))
r7 = json.load(open(C / "summary_run7.json")); r7b = json.load(open(C / "summary_run7b.json"))
HERE = Path(__file__).parent
size = json.load(open(HERE / "size_eval.json")); an = json.load(open(HERE / "analysis5.json"))

def e(d):  # entry -> dict of subset -> (AP, MR, JI) in %
    out = {}
    for g in ("all", "_light", "_heavy", "_severe"):
        m = d[g]; out[g.strip("_")] = [round(m["AP"] * 100, 2), round(m["MR"] * 100, 2), None if m.get("JI") is None else round(m["JI"] * 100, 2)]
    return out
# Run 3 (seed 0 of the head on C) was reported in the roadmap; all-subset values below are from that run's summary
SLR_C_S0 = {"all": [89.66, 48.98, 79.50], "light": [93.49, 31.43, None], "heavy": [85.95, 46.12, None], "severe": [79.49, 47.27, None]}

N = {
  "A": {"nms": e(r5["A_y26s__nms"]), "free": e(r5["A_y26s__nmsfree"]), "slr": e(r6["slr_on_A__slr"]),
        "params": r5["A_y26s__model"]["params_M"], "gflops": r5["A_y26s__model"]["gflops_640"], "train_h": r5["A_y26s__model"]["run_info"]["train_hours"]},
  "B": {"nms": e(r5["B_p2__nms"]), "free": e(r5["B_p2__nmsfree"]), "slr": e(r6["slr_on_B_noocc__slr"]),
        "slrv_seeds": [e(r5["slr_on_B__slr"]), e(r6["slr_on_B_s1__slr"])], "norel": e(r6["slr_on_B_norel__slr"]),
        "params": r5["B_p2__model"]["params_M"], "gflops": r5["B_p2__model"]["gflops_640"], "train_h": r5["B_p2__model"]["run_info"]["train_hours"]},
  "C": {"nms": e(r4["p2vw_full__nms"]), "free": e(r4["p2vw_full__nmsfree"]),
        "slrv_seeds": [SLR_C_S0, e(r4["slr_v4_s1__slr"]), e(r5["slr_v4_s2__slr"])]},
  "head": {"params": 0.97, "gflops": 3.1, "train_h": {"A": r6["slr_on_A__train_info"]["hours"], "B": r6["slr_on_B_noocc__train_info"]["hours"],
            "B_v_s0": r5["slr_on_B__train_info"]["hours"], "C_s2": r5["slr_v4_s2__train_info"]["hours"]}},
  "pilot": {   # 30 ep, 1/3 data (Run 4 re-evaluation): NMS and NMS-free
    "B_s0": {"nms": e(r4["baseline_s0__nms"]), "free": e(r4["baseline_s0__nmsfree"])},
    "B_s1": {"nms": e(r4["baseline_s1__nms"]), "free": e(r4["baseline_s1__nmsfree"])},
    "C_s0": {"nms": e(r4["vw_s0__nms"]), "free": e(r4["vw_s0__nmsfree"])},
    "C_s1": {"nms": e(r4["vw_s1__nms"]), "free": e(r4["vw_s1__nmsfree"])}},
  "VBA": {"nms": e(r4["vba_full__nms"]), "free": e(r4["vba_full__nmsfree"])},
  "cp_in": {k: r7b[k] for k in ("cp_B_in__nms", "cp_B_in__nmsfree", "cp_B_in__slr_noocc", "cp_B_in__slr_occ")},
  "cp_in_train_h": r7b["cp_B_in__run_info"]["train_hours"], "cp_head_h": r7b["cp_head_noocc__train_info"]["hours"],
  "cp_xd": {k: r7[k] for k in ("xd_B__nms", "xd_B__nmsfree", "xd_B__slr_noocc", "xd_B__slr_occ")},
  "size": {k: [round(v["AP"] * 100, 2), round(v["MR"] * 100, 2)] for k, v in size.items()},
  "errors": {k: an[k]["errors_1fppi"] for k in ("slr", "free", "nms")},
  "thr_1fppi": {k: an[k]["thr_1fppi"] for k in ("slr", "free", "nms")},
  "recall_1fppi": {k: an[k]["recall_1fppi"] for k in ("slr", "free", "nms")},
  "auc_tp_vs_dup": an["auc_tp_vs_dup"], "auc_tp_vs_bg": an["auc_tp_vs_bg"], "score_shift": {k: an["score_shift"][k]["median"] for k in an["score_shift"]},
  "score_shift_by_vis": {k: an["score_shift_by_vis"][k]["median"] for k in an["score_shift_by_vis"]},
  "density": an["density"],
}
def mean_sd(entries):
    out = {}
    for g in ("all", "light", "heavy", "severe"):
        ap = np.array([x[g][0] for x in entries]); mr = np.array([x[g][1] for x in entries]); ji = [x[g][2] for x in entries if x[g][2] is not None]
        out[g] = dict(ap=round(ap.mean(), 2), ap_sd=round(ap.std(ddof=1), 2), mr=round(mr.mean(), 2), mr_sd=round(mr.std(ddof=1), 2),
                      ji=round(float(np.mean(ji)), 2) if ji else None, n=len(entries))
    return out
N["B"]["slrv"] = mean_sd(N["B"]["slrv_seeds"]); N["C"]["slrv"] = mean_sd(N["C"]["slrv_seeds"])
N["pilot"]["B_mean"] = {m: mean_sd([N["pilot"]["B_s0"][m], N["pilot"]["B_s1"][m]]) for m in ("nms", "free")}
N["pilot"]["C_mean"] = {m: mean_sd([N["pilot"]["C_s0"][m], N["pilot"]["C_s1"][m]]) for m in ("nms", "free")}

# ---- Run 8 (latency) and Run 9 (head sensitivity)
lat = json.load(open(C / "latency_run8.json")); r9 = json.load(open(C / "summary_run9.json"))
ms = {k: np.array(v) for k, v in lat["ms"].items()}; pers = np.array(lat["persons"])
def stats(v): return dict(mean=round(float(v.mean()), 1), median=round(float(np.median(v)), 1), p95=round(float(np.percentile(v, 95)), 1), max=round(float(v.max()), 0), n_over_100=int((v > 100).sum()))
N["latency"] = {k: stats(v) for k, v in ms.items()}
N["latency"]["gpu"] = lat["gpu"]; N["latency"]["n_images"] = len(pers)
N["latency"]["by_persons"] = {k: {f"{lo}-{hi}": round(float(np.median(ms[k][(pers >= lo) & (pers < hi)])), 1) for lo, hi in ((0, 10), (10, 20), (20, 40), (40, 400))} for k in ("A_nms", "B_nms", "A_nmsfree", "B_nmsfree")}
N["latency"]["by_persons_n"] = {f"{lo}-{hi}": int(((pers >= lo) & (pers < hi)).sum()) for lo, hi in ((0, 10), (10, 20), (20, 40), (40, 400))}
slow = {k: set(np.where(ms[k] > 100)[0].tolist()) for k in ("A_nms", "B_nms")}
N["latency"]["slow_common"] = len(slow["A_nms"] & slow["B_nms"]); N["latency"]["slow_persons_max"] = int(pers[sorted(slow["A_nms"] | slow["B_nms"])].max())
N["latency"]["slow_range_ms"] = [round(float(min(ms["A_nms"][ms["A_nms"] > 100].min(), ms["B_nms"][ms["B_nms"] > 100].min())), 0), round(float(max(ms["A_nms"].max(), ms["B_nms"].max())), 0)]
def e9(tag, which):
    d = r9[f"{tag}__{which}"]; return {g: [round(d[k]["AP"] * 100, 2), round(d[k]["MR"] * 100, 2), None if d[k]["JI"] is None else round(d[k]["JI"] * 100, 2)] for g, k in (("all", "all"), ("light", "_light"), ("heavy", "_heavy"), ("severe", "_severe"))}
N["sens"] = {t: {"base": e9(t, "base"), "slr": e9(t, "slr")} for t in ("lr5e-4", "lr2e-3", "ep5", "final_k100", "final_k300")}
for t in ("lr5e-4", "lr2e-3", "ep5"): N["sens"][t]["hours"] = round(r9[f"{t}__train_info"]["hours"], 2); N["sens"][t]["occ"] = r9[f"{t}__train_info"]["occ"]
json.dump(N, open(HERE / "numbers5.json", "w"), indent=1)
print("B+SLR-V", N["B"]["slrv"]["all"], "C+SLR-V", N["C"]["slrv"]["all"])
print("pilot B free", N["pilot"]["B_mean"]["free"]["all"], N["pilot"]["B_mean"]["free"]["severe"]); print("pilot C free", N["pilot"]["C_mean"]["free"]["all"], N["pilot"]["C_mean"]["free"]["severe"])
