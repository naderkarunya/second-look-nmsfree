"""
prepare_data.py — convert CrowdHuman and CityPersons to YOLO format for VG-YOLO.

What it does
  * writes one YOLO *pose-format* label file per image:
        <vis_bin> cx cy w h  vx1 vy1 2  vx2 vy2 2  hx1 hy1 f  hx2 hy2 f   (head box; f = 2 if labelled, else 0)
      - cx cy w h      : full-body box, clipped to the image, normalised
      - vx1 vy1, vx2 vy2: top-left and bottom-right corners of the visible-body box, stored as two
                          "keypoints" so that mosaic / flip / scale move them together with the box
      - vis_bin        : the person's occlusion visibility v quantised into NBINS bins.
        The custom trainer maps it back to (class 0, v) after augmentation.
  * optionally paints "ignore" regions (crowd / unlabelled areas) with the mean colour,
    without touching pixels that belong to a labelled person.
  * writes a dataset YAML and a small statistics report (persons/image, max persons,
    visibility groups) used for the dataset statistics and for the max_det setting.

Usage (examples)
  python prepare_data.py crowdhuman  --root /data/CrowdHuman  --out /work/ch_yolo  --paint-ignore
  python prepare_data.py citypersons --root /data/citypersons --out /work/cp_yolo  --paint-ignore

Expected input layout
  CrowdHuman : root/annotation_train.odgt, root/annotation_val.odgt, root/Images/, root/Images_val/
  CityPersons: root/leftImg8bit/{train,val}/<city>/*_leftImg8bit.png
               root/gtBboxCityPersons/{train,val}/<city>/*_gtBboxCityPersons.json
"""
import argparse, json, os, shutil, sys
from pathlib import Path
import numpy as np
from PIL import Image

NBINS = 20                      # visibility resolution 0.05
IMG_MEAN = (114, 114, 114)      # Ultralytics padding colour, used for ignore regions


def clip_box(b, W, H):
    x, y, w, h = b
    x1, y1, x2, y2 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    return (x1, y1, max(0, x2 - x1), max(0, y2 - y1))


def vis_ratio(fbox, vbox, W=None, H=None):
    """Eq. (1), occlusion-only: |B_vis ∩ B_full ∩ image| / |B_full ∩ image|.
    Clipping the full box to the image first means that a person cut off by the image border
    (truncation) is NOT counted as occluded — only parts hidden by other objects lower v."""
    if W is not None:
        fbox, vbox = clip_box(fbox, W, H), clip_box(vbox, W, H)
    fx, fy, fw, fh = fbox
    vx, vy, vw, vh = vbox
    iw = max(0.0, min(fx + fw, vx + vw) - max(fx, vx))
    ih = max(0.0, min(fy + fh, vy + vh) - max(fy, vy))
    return float(np.clip(iw * ih / max(fw * fh, 1e-6), 0.0, 1.0))


def vbin(v):
    return int(min(NBINS - 1, max(0, np.floor(v * NBINS))))


def yolo_line(b, vb, W, H, v, hb=None):
    """Pose-format line: full box (clipped, normalised) + 4 keypoints:
    visible-box corners (top-left, bottom-right) and head-box corners (top-left, bottom-right).
    A missing / ignored head gets zeros with visibility flag 0."""
    x1, y1, w, h = clip_box(b, W, H)
    if w < 2 or h < 2:
        return None
    vx1, vy1, vw, vh = clip_box(vb, W, H)
    if vw < 1 or vh < 1:                       # degenerate visible box: fall back to the full box
        vx1, vy1, vw, vh = x1, y1, w, h
    head = "0 0 0 0 0 0"
    if hb is not None:
        hx1, hy1, hw, hh = clip_box(hb, W, H)
        if hw >= 1 and hh >= 1:
            head = f"{hx1/W:.6f} {hy1/H:.6f} 2 {(hx1+hw)/W:.6f} {(hy1+hh)/H:.6f} 2"
    return (f"{vbin(v)} {(x1+w/2)/W:.6f} {(y1+h/2)/H:.6f} {w/W:.6f} {h/H:.6f} "
            f"{vx1/W:.6f} {vy1/H:.6f} 2 {(vx1+vw)/W:.6f} {(vy1+vh)/H:.6f} 2 {head}")


def paint_ignore(img_path, out_path, ignore_boxes, keep_boxes):
    im = np.array(Image.open(img_path).convert("RGB"))
    H, W = im.shape[:2]
    mask = np.zeros((H, W), bool)
    for x, y, w, h in ignore_boxes:
        mask[max(0, int(y)):min(H, int(y + h)), max(0, int(x)):min(W, int(x + w))] = True
    for x, y, w, h in keep_boxes:            # never paint over a labelled person
        mask[max(0, int(y)):min(H, int(y + h)), max(0, int(x)):min(W, int(x + w))] = False
    im[mask] = IMG_MEAN
    Image.fromarray(im).save(out_path, quality=95)
    return float(mask.mean())


def write_image(src, dst, ign, keep, paint):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if paint and ign:
        return paint_ignore(src, dst, ign, keep)
    if not dst.exists():
        try:
            os.symlink(os.path.abspath(src), dst)
        except OSError:
            shutil.copy(src, dst)
    return 0.0


def records_crowdhuman(root, split):
    ann = root / f"annotation_{split}.odgt"
    img_dir = root / ("Images" if split == "train" else "Images_val")
    for line in open(ann):
        r = json.loads(line)
        yield r["ID"], img_dir / f"{r['ID']}.jpg", r["gtboxes"]


def parse_crowdhuman(gtboxes):
    persons, ignore = [], []
    for b in gtboxes:
        if b["tag"] != "person" or b.get("extra", {}).get("ignore", 0) == 1:
            ignore.append(b["fbox"])
        else:
            hb = b.get("hbox")
            if b.get("head_attr", {}).get("ignore", 0) == 1:
                hb = None
            persons.append((b["fbox"], b["vbox"], hb))
    return persons, ignore


def records_citypersons(root, split):
    for jf in sorted((root / "gtBboxCityPersons" / split).glob("*/*_gtBboxCityPersons.json")):
        stem = jf.name.replace("_gtBboxCityPersons.json", "")
        img = root / "leftImg8bit" / split / jf.parent.name / f"{stem}_leftImg8bit.png"
        yield stem, img, json.load(open(jf))


def parse_citypersons(j):
    """Standard protocol: 'pedestrian' is positive; every other label is treated as ignore."""
    persons, ignore = [], []
    for o in j["objects"]:
        if o["label"] == "pedestrian":
            persons.append((o["bbox"], o["bboxVis"], None))   # CityPersons has no head boxes
        else:
            ignore.append(o["bbox"])
    return persons, ignore


def convert(dataset, root, out, paint, only_existing):
    root, out = Path(root), Path(out)
    stats = {}
    for split in ["train", "val"]:
        recs = records_crowdhuman(root, split) if dataset == "crowdhuman" else records_citypersons(root, split)
        n_img, counts, vis, painted, skipped, heads = 0, [], [], [], 0, []
        for stem, img, ann in recs:
            persons, ignore = parse_crowdhuman(ann) if dataset == "crowdhuman" else parse_citypersons(ann)
            counts.append(len(persons))
            if not img.exists():
                skipped += 1
                if only_existing:
                    continue
                raise FileNotFoundError(img)
            W, H = Image.open(img).size
            vs = [vis_ratio(b, vb, W, H) for b, vb, _ in persons]
            vis += vs
            heads += [hb is not None for _, _, hb in persons]
            lines = [l for l in (yolo_line(b, vb, W, H, v, hb) for (b, vb, hb), v in zip(persons, vs)) if l]
            (out / "labels" / split).mkdir(parents=True, exist_ok=True)
            (out / "labels" / split / f"{stem}.txt").write_text("\n".join(lines))
            ext = img.suffix if not paint else ".jpg"
            painted.append(write_image(img, out / "images" / split / f"{stem}{ext}", ignore,
                                       [b for b, _, _ in persons], paint))
            n_img += 1
        counts, vis = np.array(counts), np.array(vis)
        stats[split] = dict(
            annotated_images=int(len(counts)), converted_images=n_img, images_missing=skipped,
            persons=int(counts.sum()), persons_per_image=round(float(counts.mean()), 2),
            max_persons_in_one_image=int(counts.max()),
            images_over_300_persons=int((counts > 300).sum()),
            visibility_stats_over_converted_images_only=True,
            visibility_light_ge_0_65=round(float((vis >= .65).mean()), 3),
            visibility_heavy_0_3_to_0_65=round(float(((vis >= .3) & (vis < .65)).mean()), 3),
            visibility_severe_lt_0_3=round(float((vis < .3).mean()), 3),
            persons_with_head_box=round(float(np.mean(heads)) if heads else 0.0, 3),
            mean_fraction_of_image_painted=round(float(np.mean(painted)) if painted else 0.0, 4))
    maxp = max(s["max_persons_in_one_image"] for s in stats.values())
    stats["recommended_max_det"] = int(np.ceil(maxp * 1.3 / 100) * 100)
    yaml = (f"path: {out.resolve()}\ntrain: images/train\nval: images/val\n"
            f"# class field in label files = visibility bin (0..{NBINS-1}); trainer maps it to class 0\n"
            f"nc: 1\nnames: ['person']\nvis_bins: {NBINS}\n"
            f"kpt_shape: [4, 3]   # visible-box corners, then head-box corners (top-left, bottom-right)\n"
            f"flip_idx: [0, 1, 2, 3]\n")
    (out / f"{dataset}.yaml").write_text(yaml)
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["crowdhuman", "citypersons"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--paint-ignore", action="store_true", help="fill ignore regions with the mean colour")
    ap.add_argument("--only-existing", action="store_true", help="skip annotations whose image is missing (for samples)")
    a = ap.parse_args()
    print(json.dumps(convert(a.dataset, a.root, a.out, a.paint_ignore, a.only_existing), indent=2))
