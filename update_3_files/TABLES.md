# Where every table and figure comes from

Table and figure numbers refer to the revised manuscript (IEEE Access resubmission). All CrowdHuman numbers are
AP / MR⁻² / JI (%) on the 4,370 validation images with the official CrowdDet evaluator (max 500 detections,
confidence ≥ 0.001). CityPersons numbers are MR⁻² (%) from the official CityPersons evaluation code
(Reasonable / Small / Heavy / All) and from `src/cp_eval.py` (Partial / Bare). MOT20 numbers use the CrowdDet
evaluator on the set built by `src/mot20_prep.py`.

| Table / figure | Content | Notebook | Raw file (results/) |
|---|---|---|---|
| Table 1 | Dataset statistics (CrowdHuman, CityPersons, MOT20) | `src/prepare_data.py`, `src/mot20_prep.py` | official annotation files |
| Table 2 | Official CityPersons evaluator vs. re-implementation, same predictions | revision/Rev3 | revision/rev3_citypersons_official_vs_reimpl.json |
| Table 3 | Implementation and training settings | — | — |
| Table 4, lower part | A, B, C; A + SLR-V; C + SLR-V; **B + SLR** (seed 0 and three seeds) | main/Run5, main/Run6, revision/Rev2 | summary_run5.json, summary_run6.json, revision/revision_numbers.json |
| Table 5, lower part | CityPersons in-domain: B both outputs, + SLR-V, + SLR | main/Run7b, revision/Rev3 | summary_run7b.json, revision/rev3_citypersons_official_vs_reimpl.json |
| Table 6 | NMS-free deficit of A, B, C and gain of the head | main/Run5, main/Run6 | summary_run5.json, summary_run6.json |
| Table 7 | Component ablation: (a) structure, (b) supervision | main/Run5, main/Run6, revision/Rev2 | as Table 4 |
| Table 8 | Every seed of the heads | main/Run5, main/Run6, revision/Rev2 | numbers5.json (`slrv_seeds`), revision/revision_numbers.json |
| Table 9 | Head structure: geometry only, geometry + relation, appearance + geometry, all three | main/Run6, revision/Rev2 | summary_run6.json, revision/revision_numbers.json |
| Table 10 | Re-scoring vs. Hard-NMS / Soft-NMS on the same candidates, latency, bootstrap intervals | revision/Rev1 | revision/revision_numbers.json |
| Table 11 | Sensitivity: K, learning rate, schedule | main/Run9 (Run9b: plain-head rerun) | summary_run9.json |
| Table 12 | Results by occlusion subset | main/Run5, main/Run6 (`--subsets`) | summary_run5.json, summary_run6.json |
| Table 13 | Cross-dataset: CrowdHuman models on CityPersons without retraining | main/Run7, revision/Rev3 | summary_run7.json, revision/rev3_citypersons_official_vs_reimpl.json |
| Table 14 | Cross-dataset: CrowdHuman models on MOT20 without retraining | revision/Rev4 | revision/revision_numbers.json |
| Table 15 | Params, GFLOPs, complete-pipeline latency, training time | revision/Rev3 (second session: revision/Rev1) | revision/revision_numbers.json |
| Table 16 | Error analysis at 1 FPPI | `analysis/analysis5.py` | analysis5.json |
| Figure 3 | Score change of the head (TP / duplicate / background; by visibility; FP composition) | `analysis/analysis5.py`, `analysis/figs5_real.py` | analysis5.json |
| Figure 4 | MR–FPPI curves (All, Severe) and recall at 1 FPPI by crowd density | same | analysis5.json |
| Figure 5 | Ground truth / NMS / one-to-one / + SLR on validation images | `analysis/figs5_qual.py` | detection files |
| Figure 6 | Failure cases | `analysis/figs5_qual.py` | detection files |

`results/numbers5.json` is compiled from the files of the original submission by `analysis/numbers5.py`;
`results/revision/revision_numbers.json` collects the numbers of the experiments added in the revision.

## Notebook summary

| Notebook | What it trains / evaluates | GPU time |
|---|---|---|
| Run5_Baselines_A_B_and_SLR | A and B on the full training set (16 epochs); SLR-V on C (seed 2) and on B | ≈ 6 h (2 GPUs) |
| Run6_Head_Ablations_A_B | SLR on A; SLR-V on B (seed 1); **SLR on B (final model)**; B without relation features | ≈ 3.5 h |
| Run7_CityPersons_final_model | first CityPersons training (stopped at 38 epochs; superseded) + cross-dataset evaluations (Table 13) | ≈ 5 h |
| Run7b_CityPersons_full_schedule | CityPersons detector, full 60 epochs, + both heads (Table 5) | ≈ 6.5 h |
| Run8_latency | batch-1 FP16 latency on a T4 (earlier latency run; superseded by revision/Rev3) | ≈ 15 min |
| Run9_head_sensitivity | Table 11 (note: the three retrained heads use the visibility branch; Run9b reruns them without it) | ≈ 4 h |
| Run9b_head_sensitivity_noocc | optional plain-head rerun of the three retrained rows of Table 11 | ≈ 2.5 h |

## Notebooks added in the revision (notebooks/revision)

| Notebook | What it evaluates | GPU time |
|---|---|---|
| Rev1_same_candidates_latency_citypersons | Hard-NMS / Soft-NMS on the same 500 candidates, complete-pipeline latency, paired bootstrap (Table 10) | ≈ 1.2 h |
| Rev2_head_variants_and_seeds | Seeds 1 and 2 of the final head; geometry-only and geometry + relation heads; head-only latency (Tables 4, 7, 8, 9) | ≈ 4.8 h |
| Rev3_citypersons_official_and_A_latency | Official CityPersons evaluator on the same predictions; latency for A and B (Tables 2, 5, 13, 15) | ≈ 0.5 h |
| Rev4_mot20_cross_dataset | MOT20 cross-dataset evaluation at 640 and 1,280 px with bootstrap (Table 14) | ≈ 2 h |
