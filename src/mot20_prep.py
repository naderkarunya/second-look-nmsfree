"""mot20_prep.py — MOT20 (MOTChallenge 2020) training sequences -> CrowdHuman-style evaluation set.

Purpose: a third, cross-dataset test of generalisation beyond CrowdHuman and CityPersons. MOT20 is a public, very crowded
pedestrian benchmark (62-227 people per frame). Only the four training sequences carry ground truth, and no model here is
trained on them, so the evaluation is strictly cross-dataset (CrowdHuman-trained detector and head, no retraining).

Protocol (fixed before any number was looked at):
  * frames: every STEP-th frame of each training sequence (consecutive frames at 25 fps are near-duplicates)
  * positives: class 1 (pedestrian) with the "consider" flag 1 and visibility > 0
  * ignore regions (a detection matching one is neither TP nor FP, exactly as CrowdHuman's ignore boxes):
      class 1 with visibility 0 (fully occluded), 2 person on vehicle, 7 static person, 8 distractor,
      12 reflection, 13 crowd/group
  * dropped (a detection there counts as a false positive, as it should): 3-6 vehicles, 9-11 occluders
  * visibility subsets as in the paper (light >= 0.65, heavy 0.30-0.65, severe < 0.30) use MOT20's own visibility
    value; we store it as a visible box of the same width and height x visibility so that export_eval.vis_ratio
    recovers it (the ratio of areas is what the subset rule needs)

gt.txt columns (MOT16/17/20 convention): frame, id, bb_left, bb_top, bb_width, bb_height, conf/flag, class, visibility.
Writes <out>/annotation_mot20.odgt and <out>/images/<ID>.jpg (symlinks), ID = <sequence>_<frame:06d>.
"""
import argparse, configparser, json, os
from pathlib import Path

IGNORE_CLASSES = {2, 7, 8, 12, 13}
DROP_CLASSES = {3, 4, 5, 6, 9, 10, 11}


def seq_info(seq_dir):
    cfg = configparser.ConfigParser(); cfg.read(Path(seq_dir) / "seqinfo.ini")
    s = cfg["Sequence"]; return int(s["imWidth"]), int(s["imHeight"]), int(s["seqLength"]), s.get("imExt", ".jpg")


def convert(root, out, step=5, seqs=None):
    root, out = Path(root), Path(out); (out / "images").mkdir(parents=True, exist_ok=True)
    seqs = seqs or sorted(p.name for p in root.iterdir() if (p / "gt" / "gt.txt").exists())
    records, stats = [], {}
    for seq in seqs:
        W, H, n_frames, ext = seq_info(root / seq)
        per_frame = {}
        for line in open(root / seq / "gt" / "gt.txt"):
            f = line.strip().split(",")
            if len(f) < 9: continue
            frame, cls, flag, vis = int(f[0]), int(f[7]), int(f[6]), float(f[8])
            x, y, w, h = (float(v) for v in f[2:6])
            if cls in DROP_CLASSES or w <= 0 or h <= 0: continue
            if cls == 1 and flag == 1 and vis > 0:
                ignore = 0
            elif cls == 1 or cls in IGNORE_CLASSES:
                ignore = 1
            else:
                continue
            # hbox (head box) is not annotated in MOT20; the CrowdDet scorer only requires the key to exist (unused in 'box' mode)
            per_frame.setdefault(frame, []).append({"tag": "person", "fbox": [x, y, w, h],
                                                    "vbox": [x, y, w, max(1.0, h * min(1.0, vis))], "hbox": [x, y, w, max(1.0, h * 0.2)],
                                                    "head_attr": {"ignore": ignore, "occ": 0, "unsure": 0},
                                                    "extra": {"ignore": ignore, "vis": vis, "cls": cls}})
        n_pos = n_ign = 0
        for frame in range(1, n_frames + 1, step):
            ID = f"{seq}_{frame:06d}"; src = root / seq / "img1" / f"{frame:06d}{ext}"
            if not src.exists(): continue
            dst = out / "images" / f"{ID}.jpg"
            if not dst.exists(): os.symlink(src.resolve(), dst)
            boxes = per_frame.get(frame, [])
            n_pos += sum(1 for b in boxes if b["extra"]["ignore"] == 0); n_ign += sum(1 for b in boxes if b["extra"]["ignore"] == 1)
            records.append({"ID": ID, "width": W, "height": H, "gtboxes": boxes})
        stats[seq] = {"frames_used": len([r for r in records if r["ID"].startswith(seq)]), "W": W, "H": H,
                      "positives": n_pos, "ignore": n_ign, "positives_per_frame": round(n_pos / max(1, len([r for r in records if r["ID"].startswith(seq)])), 1)}
        print(seq, stats[seq], flush=True)
    with open(out / "annotation_mot20.odgt", "w") as fh:
        for r in records: fh.write(json.dumps(r) + "\n")
    json.dump(stats, open(out / "mot20_stats.json", "w"), indent=2)
    print("total frames", len(records), "positives", sum(s["positives"] for s in stats.values()))
    return records, stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="MOT20/train"); ap.add_argument("--out", required=True)
    ap.add_argument("--step", type=int, default=5)
    a = ap.parse_args(); convert(a.root, a.out, a.step)
