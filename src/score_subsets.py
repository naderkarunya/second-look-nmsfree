"""
score_subsets.py — re-score an existing dt.odgt (from export_eval.py, no --subset) on the visibility subsets
light / heavy / severe WITHOUT running the model again. Uses the image sizes stored in dt.odgt.

  python score_subsets.py --dt results/run/dt.odgt --gt annotation_val.odgt --out results/run --crowddet CrowdDet
Writes <out>_light/, <out>_heavy/, <out>_severe/ each with dt.odgt, gt_used.odgt, metrics.json.
"""
import argparse, json, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from export_eval import vis_ratio, in_subset  # noqa: E402


def score(out, crowddet, ji=True):
    """AP / MR^-2 (and JI unless ji=False: JI with 500 boxes per image is ~10x slower than AP+MR)."""
    lib = Path(crowddet) / "lib"
    code = ("import json,sys;sys.path.insert(0,'.');from evaluate import compute_JI, compute_APMR;"
            f"AP,MR=compute_APMR.compute_APMR(r'{(out/'dt.odgt').resolve()}', r'{(out/'gt_used.odgt').resolve()}', 'box');"
            + (f"_,JI=compute_JI.evaluation_all(r'{(out/'dt.odgt').resolve()}', 'box');" if ji else "JI=None;")
            + f"json.dump(dict(AP=AP,MR=MR,JI=JI),open(r'{(out/'metrics.json').resolve()}','w'),indent=2);"
            "print('AP %.4f  MR %.4f  JI %s'%(AP,MR,JI))")
    subprocess.run([sys.executable, "-c", code], cwd=lib, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dt", required=True); ap.add_argument("--gt", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--crowddet", required=True)
    ap.add_argument("--subsets", default="light,heavy,severe")
    ap.add_argument("--no-ji", action="store_true", help="skip the slow JI on the subsets (AP and MR only)")
    a = ap.parse_args()
    dts = {json.loads(l)["ID"]: json.loads(l) for l in open(a.dt)}
    gts = [json.loads(l) for l in open(a.gt)]
    gts = [r for r in gts if r["ID"] in dts]
    for subset in a.subsets.split(","):
        out = Path(f"{a.out}_{subset}"); out.mkdir(parents=True, exist_ok=True)
        dt_lines, gt_lines = [], []
        for r in gts:
            d = dts[r["ID"]]; W, H = d["width"], d["height"]
            gtb, gt_rec = [], dict(r)
            for g in r["gtboxes"]:
                ign = g["tag"] != "person" or g.get("extra", {}).get("ignore", 0) == 1
                if not ign:
                    ign = not in_subset(vis_ratio(g["fbox"], g["vbox"], W, H), subset)
                gtb.append({"box": g["fbox"], "tag": -1 if ign else 1})
                if ign and g["tag"] == "person":
                    g = dict(g); g["extra"] = dict(g.get("extra", {}), ignore=1)
                gt_rec.setdefault("_g", []).append(g)
            gt_rec["gtboxes"] = gt_rec.pop("_g", [])
            dt_lines.append(json.dumps({"ID": r["ID"], "height": H, "width": W, "dtboxes": d["dtboxes"], "gtboxes": gtb}))
            gt_lines.append(json.dumps(gt_rec))
        (out / "dt.odgt").write_text("\n".join(dt_lines) + "\n")
        (out / "gt_used.odgt").write_text("\n".join(gt_lines) + "\n")
        print(subset, end=": "); score(out, a.crowddet, ji=not a.no_ji)


if __name__ == "__main__":
    main()
