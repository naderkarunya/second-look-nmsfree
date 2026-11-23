# A second look at NMS-free real-time pedestrian detection in crowds

Code, notebooks, raw results and trained weights for the paper

> *A second look at NMS-free real-time pedestrian detection in crowds: relation-aware re-scoring of the one-to-one output of YOLO26*
> N. Salam, T. J. Jebaseeli, S. Shirly, S. Sulaiman — manuscript under review at IEEE Access (revised version).

The paper measures the gap between the NMS-free (one-to-one) output and the NMS (one-to-many) output of a trained
YOLO26 network on crowded scenes, and proposes a **Second-Look re-scoring head (SLR)**: a small second-stage module
(0.97 M parameters, 3.1 GFLOPs, 4.7 ms on a T4) that re-scores the 500 one-to-one candidates of a frozen detector from
their pooled appearance, geometry and the relation to their strongest higher-scored overlapping neighbour. On CrowdHuman
it raises the NMS-free output of YOLO26s-P2 from 87.90 to 89.96 AP, lowers MR⁻² from 52.47 to 47.52 and raises JI from
75.90 to 79.49, above the NMS output of the same network and above Hard-NMS and Soft-NMS applied to the same
candidates. Without retraining, the head also closes most of the one-to-one deficit on CityPersons and on MOT20, where it
performs on par with the NMS output. Every table and figure of the paper can be regenerated from
the files in this repository; see [TABLES.md](TABLES.md) for the mapping.

## Repository layout

```
src/        the method and the evaluation tools
  vgyolo.py          YOLO26s / YOLO26s-P2 training with the (optional) visibility-weighted loss; label conversion helpers
  refine.py          the Second-Look re-scoring head: `train` (on a frozen detector) and `eval` (CrowdDet protocol)
  prepare_data.py    CrowdHuman / CityPersons -> YOLO label format (keeps visible-body boxes)
  export_eval.py     exports detections (NMS or one-to-one) and scores them with the official CrowdDet evaluator
  score_subsets.py   AP / MR⁻² / JI on the Light / Heavy / Severe visibility subsets
  cp_eval.py         CityPersons MR⁻² re-implementation (used for Partial / Bare; the paper uses the official code for the other subsets)
  mot20_prep.py      MOT20 training sequences -> CrowdHuman-style evaluation set (revision, cross-dataset test)
  rev_tools.py       Hard-NMS / Soft-NMS on the same candidates, paired bootstrap, latency timers (revision)
  train_run.py       one detector training run from the command line (used for the pilot experiments)
notebooks/main/      the Kaggle notebooks that produced the results of the paper (Runs 5-9, see TABLES.md)
notebooks/revision/  the Kaggle notebooks of the experiments added in the revision (Rev1-Rev4)
notebooks/pilot/     earlier notebooks: pilot protocol, seed study, feature-level visibility variants (Table 11)
results/             the raw summary files written by the notebooks (JSON), exactly as used for the tables
  revision/          numbers of the revision experiments (Tables 2, 8, 9, 10, 14, 15)
analysis/            scripts that turn detections into the analyses and figures of Section 5
figures/             the figures of Section 5 as generated
```

## Setup

```
pip install -r requirements.txt            # ultralytics==8.4.163, torch, torchvision, numpy, opencv-python, matplotlib, Pillow
git clone --depth 1 https://github.com/Purkialo/CrowdDet.git   # official CrowdHuman evaluator (AP, MR⁻², JI)
```

Datasets: [CrowdHuman](https://www.crowdhuman.org/) (`annotation_train.odgt`, `annotation_val.odgt`, `Images_train`, `Images_val`)
and [CityPersons](https://github.com/cvgroup-njust/CityPersons) on top of the Cityscapes `leftImg8bit` images.
Both are used under their own licences and are not redistributed here.

## Reproducing the main result (CrowdHuman, YOLO26s-P2 + SLR)

```
# 1. convert annotations (once)
python src/prepare_data.py crowdhuman --root /path/CrowdHuman --out data/ch_yolo --paint-ignore

# 2. stage 1: train the detector (16 epochs, one T4, ~2.5 h). Model B = yolo26s-p2; model A = yolo26s; model C = B with --vis-weight 1.0
python src/train_run.py --mode baseline --vis-weight 0.0 --base yolo26s-p2 --data data/ch_yolo/crowdhuman.yaml --epochs 16 \
       --fraction 1.0 --seed 0 --imgsz 640 --batch 16 --workers 2 --device 0 --project runs --name yolo26s-p2_baseline_s0

# 3. score both outputs of the trained network with the official evaluator
python src/export_eval.py --weights runs/yolo26s-p2_baseline_s0/weights/best.pt --images /path/CrowdHuman/Images_val \
       --gt /path/CrowdHuman/annotation_val.odgt --crowddet CrowdDet --out results/B_nms            # one-to-many + NMS
python src/export_eval.py ... --nms-free --out results/B_free                                      # one-to-one, NMS-free
python src/score_subsets.py --dt results/B_free/dt.odgt --gt /path/CrowdHuman/annotation_val.odgt --out results/B_free \
       --crowddet CrowdDet --no-ji                                                                  # Light / Heavy / Severe

# 4. stage 2: train the Second-Look head on the frozen detector (10 epochs, ~1.5 h) and evaluate (final model = --no-occ)
python src/refine.py train --weights runs/yolo26s-p2_baseline_s0/weights/best.pt --data data/ch_yolo/crowdhuman.yaml \
       --epochs 10 --batch 16 --imgsz 640 --k 500 --lr 1e-3 --score-only --rel --no-occ --seed 0 --out runs/slr_on_B
python src/refine.py eval --weights runs/yolo26s-p2_baseline_s0/weights/best.pt --head runs/slr_on_B/slr.pt \
       --images /path/CrowdHuman/Images_val --gt /path/CrowdHuman/annotation_val.odgt --crowddet CrowdDet \
       --imgsz 640 --k 500 --subsets --out results/B_slr
```

`refine.py train` flags used in the paper: `--score-only` (score correction only, box branch fixed), `--rel` (relation
feature of the strongest higher-scored neighbour), `--no-occ` (plain head = final model; omit it for the SLR-V variant
with the visibility branch). The notebooks in `notebooks/main/` run exactly these commands on Kaggle (two T4 GPUs) and
write the `results/summary_run*.json` files.

## Datasets added in the revision

[MOT20](https://motchallenge.net/data/MOT20/) training sequences (used only for evaluation, never for training):

```
python src/mot20_prep.py --root /path/MOT20/train --out data/mot20 --step 5   # 1,787 frames, 220,626 pedestrians
```

The CityPersons numbers of the revised paper are computed with the official evaluation code of
[cvgroup-njust/CityPersons](https://github.com/cvgroup-njust/CityPersons); `notebooks/revision/Rev3_*` shows how the
predictions are converted (`rev_tools.cp_to_coco`).

## Trained weights

The detectors (A, B, C on CrowdHuman; B on CityPersons) and all heads are published as a release asset of this
repository: see the **Releases** page (`weights.zip`). File names follow the notebooks (`yolo26s-p2_baseline_s0_e16_f1.0/weights/best.pt`, `slr_on_B_noocc/slr.pt`, ...).

## Analyses and figures of Section 5

With the exported detection files of step 3-4 (`dt.txt` of the NMS output, `*_basedt.txt` of the one-to-one output,
`*dt.txt` of the head) and `annotation_val.odgt` in a folder `data/`:

```
python analysis/analysis5.py        # operating points at 1 FPPI, error categories, score-shift statistics, MR-FPPI curves, density bins
python analysis/size_subsets.py     # Small / Medium / Large subsets with the official evaluator
python analysis/numbers5.py         # compiles results/*.json into results/numbers5.json (the single source of every table)
python analysis/figs5_real.py       # gap chart, subsets, head behaviour (Figure 3), curves/density (Figure 4)
python analysis/figs5_qual.py       # qualitative grid (Figure 5) and failure cases (Figure 6); --two for the two-row versions
```

## Licence

The code in this repository is released under the GNU Affero General Public License v3.0 (AGPL-3.0), the licence of the
Ultralytics framework it builds on. The datasets keep their own licences.

## Citation

See [CITATION.cff](CITATION.cff). Until the paper is published, please cite the repository.
