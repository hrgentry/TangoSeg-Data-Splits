# Revision materials (v2.0.0)

This directory adds the materials behind the revised manuscript (CACAIE-D-26-01086,
first revision). Everything from v1.8.0 outside this directory is unchanged.
As before, **no dataset images or annotations are redistributed**, and no model
weights are included.

## What changed in the analysis

* Every matrix cell is retrained with two further seeds (42, 43, 44), and the
  primary score is **held-out F1**: the threshold is selected on the validation
  split and frozen; DeepCrack, whose validation and test lists coincide, uses
  two-fold cross-fitting on its test set (mean of 200 random splits).
  Test-selected pixel-ODS is kept as an upper bound.
* The partition controls are repeated with **fixed test images** (B2): a fixed
  parent-level Crack500 test set with equally sized leaking and clean training
  sets, and five-fold cross-validation on CrackMap folded by source photograph or
  at random.
* The training budget is split into **duration and schedule** (B3) with
  per-five-epoch checkpoints of the 50-epoch runs.
* The matched TANGO contrasts use **five seeds** and **crack-geometry metrics**
  (broken long cracks, thin-crack recall, centerline Dice, spurious components).
* The two pretraining sources share **one from-scratch reference** (B5).
* Six models are retrained at **native resolution** (B7).

## Contents

| Path | Contents | Manuscript |
|---|---|---|
| `results/matrix_heldout.json` | Held-out F1, pixel-ODS, AP per model, dataset and seed; nested bootstrap intervals (seed 42) | Table 4, Sections 2.5, 4.1, 4.5, Suppl. S8 |
| `results/partition_controls.json`, `partition_common.json`, `b2_partition_isolation.json` | Original and fixed-test partition comparisons | Section 3.1 |
| `results/budget_split.json` | Duration vs. schedule decomposition, short-budget spread across seeds, RINDNet micro-batch rerun | Section 4.2 |
| `results/b5_pretraining.json`, `b5_curves.json`, `b5_scalars_val.txt` | Pretraining contrasts against the shared reference; validation curves every five epochs | Section 4.3, Suppl. S6 |
| `results/matched_contrasts_rev1.json` | Five-seed matched contrasts, held-out F1 and geometry metrics, paired intervals, Holm | Table 6, Section 4.4 |
| `results/matrix_geometry.json` | Geometry metrics of the sixteen-model matrix (seed 42) | Suppl. S11 |
| `results/b7_native.json` | Native-resolution vs. 512 by 512, three seeds | Suppl. S10 |
| `results/final_vs_best.json` | Final-epoch vs. validation-selected checkpoint on untouched data | Section 3.2 |
| `results/implementation_provenance.json` | Code source, licence and loaded weights of every model row | Suppl. S8 |
| `per_image/histograms/` | For every evaluated checkpoint, per-image counts of foreground and background pixels in each of 1,001 score bins (`.npz`: `pos`, `neg`, `names`, `meta`). Every threshold rule, F1, AP and bootstrap in the paper is recomputable from these | all accuracy numbers |
| `per_image/geometry/` | Per-image geometry metrics at the held-out threshold | Table 6, Suppl. S10, S11 |
| `splits/crack500_fixed_test_g{0,1,2}/` | Fixed-test Crack500 splits (`train_leak`, `train_clean`, `test`, ...) for three grouped split seeds | Section 3.1 |
| `splits/crackmap_cv_{group,random}/fold{0..4}/` | CrackMap five-fold splits; the pooled test set of each arm is the same 120 images | Section 3.1 |
| `training_logs/` | Per-epoch validation scalars (`scalars.json`) and the framework's full dumped configuration (`config.py`) of every run, local and second machine. Seeds are in each config (`randomness.seed`) | checkpoint histories |
| `configs/` | The training configurations written for this study (mmsegmentation 1.2.2) | Section 2.3 |
| `scripts/` | Evaluation (`eval_rev1_bins.py`, `eval_rev1_geometry.py`, `eval_rev1_native.py`), statistics (`rev1_bins_stats.py`, `build_*.py`) and split generation (`make_rev1_isolation_splits.py`) | — |
| `supplementary_material.pdf` | The revised supplementary material | — |

## Not included

Dataset images and annotations (licensed by their original authors), model
weights, and third-party model source code. The configurations reference model
modules by their registered names; the third-party sources and their licences are
listed in `results/implementation_provenance.json`.

## Notes

Server paths in the second machine's dumped configurations are shortened to
`~/`, and one private network address is replaced by `<lan-host>`; no values
were changed. Split index files list one sample per line as
`<image path> <mask path>`, relative to the dataset root.
