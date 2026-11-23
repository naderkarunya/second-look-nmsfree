"""Official AP / MR^-2 on pedestrian-size subsets (Small h<50, Medium 50<=h<150, Large h>=150; height of the full-body
box in the original image), persons outside the subset treated as ignore — same mechanism as the visibility subsets."""
import json, sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from score_subsets import score

UP = Path(os.environ.get("SLR_DATA_DIR", Path(__file__).resolve().parent.parent / "data"))   # see README: detections + annotation_val.odgt
CD = os.environ.get("CROWDDET_DIR", "CrowdDet")   # clone of https://github.com/Purkialo/CrowdDet (official evaluator)
GT = UP / "annotation_val.odgt"
DT = {"nms": UP / "dt.txt", "free": UP / "slr_on_B_noocc_basedt.txt", "slr": UP / "slr_on_B_nooccdt.txt"}
SIZES = {"small": lambda h: h < 50, "medium": lambda h: 50 <= h < 150, "large": lambda h: h >= 150}
OUTD = Path(__file__).parent / "size_eval"; OUTD.mkdir(exist_ok=True)
gts = [json.loads(l) for l in open(GT)]
res = {}
for k, p in DT.items():
    dts = {json.loads(l)["ID"]: json.loads(l) for l in open(p)}
    for name, rule in SIZES.items():
        out = OUTD / f"{k}_{name}"; out.mkdir(exist_ok=True)
        dl, gl = [], []
        for r in gts:
            d = dts[r["ID"]]; gtb = []; rec = dict(r); newg = []
            for g in r["gtboxes"]:
                ign = g["tag"] != "person" or g.get("extra", {}).get("ignore", 0) == 1
                if not ign: ign = not rule(g["fbox"][3])
                gtb.append({"box": g["fbox"], "tag": -1 if ign else 1})
                if ign and g["tag"] == "person": g = dict(g); g["extra"] = dict(g.get("extra", {}), ignore=1)
                newg.append(g)
            rec["gtboxes"] = newg
            dl.append(json.dumps({"ID": r["ID"], "height": d["height"], "width": d["width"], "dtboxes": d["dtboxes"], "gtboxes": gtb}))
            gl.append(json.dumps(rec))
        (out / "dt.odgt").write_text("\n".join(dl) + "\n"); (out / "gt_used.odgt").write_text("\n".join(gl) + "\n")
        print(k, name, end=": "); score(out, CD, ji=False)
        res[f"{k}_{name}"] = json.load(open(out / "metrics.json"))
json.dump(res, open(Path(__file__).parent / "size_eval.json", "w"), indent=2)
print(json.dumps(res, indent=1))
