# Where every table and figure comes from

Table and figure numbers refer to the 25-page manuscript; the numbers of the full-length version are given in
parentheses where they differ. All CrowdHuman numbers are AP / MR⁻² / JI (%) on the 4,370 validation images with the
official CrowdDet evaluator (max 500 detections, confidence ≥ 0.001); CityPersons numbers are MR⁻² (%) from `src/cp_eval.py`.

| Table / figure | Content | Notebook (notebooks/main) | Raw file (results/) |
|---|---|---|---|
| Table 3 (6), lower part | A, B, C: both outputs; A + SLR; C + SLR-V; **B + SLR** | Run5, Run6 | summary_run5.json, summary_run6.json |
| Table 4 (7), lower part | CityPersons in-domain: B both outputs, + SLR-V, + SLR | Run7b (detector, 60 ep) | summary_run7b.json |
| Table 5 (8) | NMS-free deficit of A, B, C | Run5 | summary_run5.json |
| Table 6 (9) | Component ablation (A, B, C, A + SLR, B + SLR) | Run5, Run6 | summary_run5.json, summary_run6.json |
| Table 7 (10) | Head design: no relation features, SLR-V (2 seeds), SLR; development rows on C | Run6, Run5, pilot notebooks | summary_run6.json, summary_run5.json, summary_run4.json |
| Table 8 (11) | Sensitivity: K = 100 / 300 at test time, lr 5e-4 / 2e-3, 5 epochs | Run9 (Run9b: plain-head rerun) | summary_run9.json |
| — (12) | Loss-level vs feature-level visibility (pilot protocol, F1, F2) | notebooks/pilot | summary_run4.json |
| Table 9 (13) | Results by occlusion subset | Run5, Run6 (`--subsets`) | summary_run5.json, summary_run6.json |
| — (14) | Size subsets Small / Medium / Large | `analysis/size_subsets.py` | size_eval.json |
| — (15) | Gain of the head on three base detectors | Run5, Run6 | as Table 6 |
| Table 10 (16) | Cross-dataset: CrowdHuman models on CityPersons without retraining | Run7 (GPU 1 part) | summary_run7.json |
| Table 11 (17) | Params, GFLOPs, latency, training time | Run8 (latency); params/GFLOPs from the framework profiler | latency_run8.json |
| Table 12 (18) | Error analysis at 1 FPPI | `analysis/analysis5.py` | analysis5.json |
| Fig. 3 (8) | Score change of the head (TP / duplicate / background; by visibility; FP composition) | `analysis/analysis5.py`, `analysis/figs5_real.py` | analysis5.json |
| Fig. 4 (9) | MR–FPPI curves (All, Severe) and recall at 1 FPPI by crowd density | same | analysis5.json |
| Fig. 5 (11) | Ground truth / NMS / one-to-one / + SLR on validation images | `analysis/figs5_qual.py` | detection files |
| Fig. 6 (12) | Failure cases | `analysis/figs5_qual.py` | detection files |
| — (6, 7, 10) | Gap chart, per-subset bars, three-column qualitative figure (full version only) | `analysis/figs5_real.py`, `figs5_qual.py` | analysis5.json |

`results/numbers5.json` is compiled from the files above by `analysis/numbers5.py` and is the single source of every
number printed in the tables.

## Notebook summary

| Notebook | What it trains / evaluates | GPU time |
|---|---|---|
| Run5_Baselines_A_B_and_SLR | A and B on the full training set (16 epochs); SLR-V on C (seed 2) and on B | ≈ 6 h (2 GPUs) |
| Run6_Head_Ablations_A_B | SLR on A; SLR-V on B (seed 1); **SLR on B (final model)**; B without relation features | ≈ 3.5 h |
| Run7_CityPersons_final_model | first CityPersons training (stopped at 38 epochs; superseded) + cross-dataset evaluations (Table 10) | ≈ 5 h |
| Run7b_CityPersons_full_schedule | CityPersons detector, full 60 epochs, + both heads (Table 4) | ≈ 6.5 h |
| Run8_latency | batch-1 FP16 latency on a T4 (Table 11) | ≈ 15 min |
| Run9_head_sensitivity | Table 8 (note: the three retrained heads use the visibility branch; Run9b reruns them without it) | ≈ 4 h |
| Run9b_head_sensitivity_noocc | optional plain-head rerun of the three retrained rows of Table 8 | ≈ 2.5 h |
