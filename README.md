# A second look at NMS-free real-time pedestrian detection in crowds

Code, notebooks, raw results and trained weights for the paper

> *A second look at NMS-free real-time pedestrian detection in crowds: relation-aware re-scoring of the one-to-one output of YOLO26*
> N. Salam, T. J. Jebaseeli, S. Shirly, S. Sulaiman — manuscript under review.

The paper measures the gap between the NMS-free (one-to-one) output and the NMS (one-to-many) output of a trained
YOLO26 network on crowded scenes, and proposes a **Second-Look re-scoring head (SLR)**: a small second-stage module
(0.97 M parameters, 3.1 GFLOPs, 4.6 ms on a T4) that re-scores the 500 one-to-one candidates of a frozen detector from
their pooled appearance, geometry and the relation to their strongest higher-scored overlapping neighbour. On CrowdHuman
it raises the NMS-free output of YOLO26s-P2 from 87.90 to 89.96 AP, lowers MR⁻² from 52.47 to 47.52 and raises JI from
75.90 to 79.49, above the NMS output of the same network. Every table and figure of the paper can be regenerated from
the files in this repository; see [TABLES.md](TABLES.md) for the mapping.

## Repository layout

```
src/        the method and the evaluation tools
  vgyolo.py          YOLO26s / YOLO26s-P2 training with the (optional) visibility-weighted loss; label conversion helpers
  refine.py          the Second-Look re-scoring head: `train` (on a frozen detector) and `eval` (CrowdDet protocol)
  prepare_data.py    CrowdHuman / CityPersons -> YOLO label format (keeps visible-body boxes)
  export_eval.py     exports detections (NMS or one-to-one) and scores them with the official CrowdDet evaluator
  score_subsets.py   AP / MR⁻² / JI on the Light / Heavy / Severe visibility subsets
  cp_eval.py         CityPersons MR⁻² evaluator (Reasonable / Small / Heavy / Partial / Bare / All)
  train_run.py       one detector training run from the command line (used for the pilot experiments)
notebooks/main/      the Kaggle notebooks that produced the results of the paper (Runs 5-9, see TABLES.md)
notebooks/pilot/     earlier notebooks: pilot protocol, seed study, feature-level visibility variants (the loss-level vs feature-level comparison of Section V-B)
results/             the raw summary files written by the notebooks (JSON), exactly as used for the tables
analysis/            scripts that turn detections into the analyses and figures of Section V
figures/             the figures of Section V as generated
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

## Trained weights

The detectors (A, B, C on CrowdHuman; B on CityPersons) and the heads listed below are published as release assets of this repository (Releases → v1.0):

| Archive | Contents |
|---|---|
| `detectors_crowdhuman.zip` | `A.pt` (YOLO26s) and `B.pt` (YOLO26s-P2), trained on CrowdHuman, 16 epochs, seed 0 — B is the base of the final model; `slr_on_C_s2.pt` (SLR-V head on C, seed 2) |
| `5.zip` | detector C (`ckpts/p2vw_full.pt`, YOLO26s-P2 with the visibility-weighted loss), the evaluation scripts, logs and the scored detections of A |
| `6.zip` | Second-Look heads on CrowdHuman (`runs/*/slr.pt` with `train_info.json`): `slr_on_B_noocc` (SLR, the final head), `slr_on_B_norel` (without relation features), `slr_on_B_s1` (SLR-V, seed 1), `slr_on_A` (head on A) |
| `7b.zip` | CityPersons: detector B trained on CityPersons (`runs/yolo26s-p2_baseline_s0_e60_cp1280/weights/best.pt`) and its heads `cp_head_noocc` (SLR) and `cp_head_occ` (SLR-V) |

The remaining head seeds of Tables 6, 7 and 9 (SLR-V on B, seed 0; SLR-V on C, seeds 0–1) can be retrained in about 1.5 h each with the commands above (step 4, without `--no-occ`); their summary files are in `results/`.

File names follow the notebooks (`yolo26s-p2_baseline_s0_e16_f1.0/weights/best.pt`, `slr_on_B_noocc/slr.pt`, ...).

## Analyses and figures of Section V

With the exported detection files of steps 3-4 (`dt.txt` of the NMS output, `*_basedt.txt` of the one-to-one output,
`*dt.txt` of the head) and `annotation_val.odgt` in a folder `data/`:

```
python analysis/analysis5.py        # operating points at 1 FPPI, error categories, score-shift statistics, MR-FPPI curves, density bins
python analysis/size_subsets.py     # Small / Medium / Large subsets with the official evaluator
python analysis/numbers5.py         # compiles results/*.json into results/numbers5.json (the single source of every table)
python analysis/figs5_real.py       # Figs. 3-4 of the paper (head behaviour; MR-FPPI curves and crowd-size bins) and the gap/subset charts
python analysis/figs5_qual.py       # Figs. 5-6 of the paper (three-output grid, failure cases); --two for the two-row versions
```

## Licence

The code in this repository is released under the GNU Affero General Public License v3.0 (AGPL-3.0), the licence of the
Ultralytics framework it builds on. The datasets keep their own licences.

## Citation

See [CITATION.cff](CITATION.cff). Until the paper is published, please cite the repository.
