# TangoSeg-Data-Splits

Derived metadata that make the partitions and measurements of the accompanying
crack-segmentation study reproducible.

**No dataset images are redistributed here.** The four benchmarks used in the
study are released by their original teams under their own licences; this
repository publishes only *derived* metadata — split indices, SHA-256 manifests,
machine-readable results, and the scripts needed to check a local download and
regenerate the partitions.

## What is in here

| Path | Contents |
|---|---|
| `splits/crack500_parent_disjoint/` | The parent-disjoint Crack500 partition used in the paper: 2,357 / 337 / 674 crops over 272 / 62 / 108 mutually exclusive parent photographs |
| `splits/crack500_distributed/` | The crop-level partition shipped with the archive, kept for reference: 2,355 / 334 / 675 crops |
| `splits/camcrack789/` | The CamCrack789 70/10/20 partition: 553 / 79 / 157 images |
| `splits/crackmap_source_disjoint/` | The source-disjoint CrackMap partition, used for every CrackMap number from v1.5.0 on: 84 / 12 / 24 crops over 40 / 4 / 10 mutually exclusive GoPro source photographs |
| `splits/crackmap_original/` | The image-level CrackMap partition used up to v1.4.0, kept so the partition control can be recomputed: the same 84 / 12 / 24 crop counts, but 15 of the 24 test crops share a source photograph with the training set |
| `manifests/*.sha256` | Per-file SHA-256 of the audited copy of each dataset (DeepCrack 1,078; Crack500 6,742; CamCrack789 1,581; CrackMap 243 files) |
| `benchmarks/*.json` | Machine-readable results: the full model × dataset matrix at the prespecified final epoch, the same runs at the checkpoint a validation-selection rule would have kept, the threshold-sweep and fixed-threshold variants, the seed-repeated budget arms, the matched TANGO arms, the two Crack500 partitions side by side, and parameter/FLOP counts |
| `build_budget_by_init.py` | Recomputes the initialization-stratified budget summaries from the archived budget matrix and counted operations |
| `verify_manifests.py` | Checks a local dataset copy against a manifest |
| `make_crack500_mother_split.py` | Regenerates the parent-disjoint Crack500 partition (vendored verbatim; see below) |
| `make_crackmap_parent_split.py` | Regenerates the source-disjoint CrackMap partition (vendored verbatim; same rule, grouping by GoPro source file) |

Split index files list one sample per line as `<image path> <mask path>`,
relative to the dataset root.

## Why the manifests matter

The study reports a split audit — which partitions leak across the parent
photograph, and what that leak is worth. **Such findings are properties of one
specific copy of a dataset, not of the dataset's name.** Archives circulate in
several repackagings that differ in file naming and in which crops they contain.
Before applying the indices published here, confirm your download is the copy
that was audited:

```bash
python3 verify_manifests.py verify /path/to/Crack500/dataset \
    --manifest manifests/crack500.sha256
```

A non-zero exit means your copy differs, and the indices here may not apply to
it. The command prints exactly which files are missing, extra, or changed.

## Reproducing the parent-disjoint Crack500 partition

```bash
python3 make_crack500_mother_split.py \
    --root /path/to/Crack500/dataset --out-dir splits_check
diff splits_check/train.txt splits/crack500_parent_disjoint/train.txt
diff splits_check/val.txt   splits/crack500_parent_disjoint/val.txt
diff splits_check/test.txt  splits/crack500_parent_disjoint/test.txt
```

The assignment is a deterministic longest-processing-time heuristic with no
random seed: the same input always yields the same output. All three files are
byte-identical on the audited copy.

`make_crack500_mother_split.py` is vendored **verbatim** rather than rewritten.
Its tie-breaking rule — when two splits have the same shortfall, prefer the one
with the larger target ratio (train > test > val) — determines the partition at
every tie. A rewrite that breaks ties differently, alphabetically for example,
produces a valid but *different* partition and would not reproduce the indices
published here.

### The four duplicate-marker crops

`splits/crack500_parent_disjoint/` covers **3,368** crops while
`splits/crack500_distributed/` covers **3,364**. The difference is four crops,
all from parent `20160328_154452`, whose filenames carry a `(2)` duplicate
marker and which do not appear anywhere in the distributed index files:

```
train_img/20160328_154452_1281_721_(2).jpg
train_img/20160328_154452_1921_721_(2).jpg
val_img/20160328_154452_1_361_(2).jpg
val_img/20160328_154452_641_361_(2).jpg
```

The parent-disjoint script enumerates the image directory and therefore includes
them; the distributed index files do not. The comparison between the two
partitions in the paper is accordingly a near-matched comparison rather than an
exact single-variable contrast, and is described as such there.

Note also that these four filenames use an underscore before the marker,
`_(2)`, not a space. The archive as distributed used a space, which breaks any
index file parsed on whitespace — a path is silently truncated at the space. The
audited copy was renamed to underscores before the manifests were computed, so a
copy that still contains spaces will not verify against `crack500.sha256`.

## Reproducing the source-disjoint CrackMap partition

```bash
python3 make_crackmap_parent_split.py \
    --root /path/to/CrackMap/dataset --out splits_check
diff splits_check/train.txt splits/crackmap_source_disjoint/train.txt
diff splits_check/val.txt   splits/crackmap_source_disjoint/val.txt
diff splits_check/test.txt  splits/crackmap_source_disjoint/test.txt
```

CrackMap ships 120 crops with no official partition. Crop names carry the GoPro
source file that produced them, `GOPR0315_(3).png` being the third crop of
`GOPR0315`, and the 120 crops come from 54 source photographs. The image-level
partition in `splits/crackmap_original/` splits 18 of those 54 photographs
across roles, so 15 of its 24 test crops have a crop of the same photograph in
the training set. A content hash cannot see this: the crops are distinct files
with distinct contents.

`splits/crackmap_source_disjoint/` assigns whole photographs instead, by the
same longest-processing-time rule and the same tie-break as the Crack500 script,
and lands on the same 84 / 12 / 24 crop counts. Unlike the Crack500 pair the two
CrackMap partitions cover exactly the same 120 crops, but their test sets share
only 3 images, so a per-model difference between them mixes the assignment unit
with test-set composition. `benchmarks/crackmap_parent_vs_random.json` holds
both columns, the per-model difference, and the leakage counts for each.

## CamCrack789 partition

`splits/camcrack789/` contains the index files **actually used for every
CamCrack789 number in the paper**, published as files rather than as a recipe.
They were produced at a 70/10/20 ratio by a seeded generator (`--seed 42`), but
the files are authoritative: directory enumeration order and library versions
can shift a seeded shuffle, and a regenerated index that differed from these
would no longer match the published results.

## Benchmark JSON

Each entry stores, per dataset and per model, the threshold-selected operating
point and the metrics computed at it, the same metrics at a fixed threshold of
0.5, the evaluated image count, and parameter and FLOP counts. `qualitative_selection.json`
records the four fixed samples used by the current qualitative figure, one per
dataset, and their per-model operating points. Its input-only descriptor rules
are explicit: crack width is foreground area divided by skeleton length at
native 256 x 256 resolution; contrast uses the common 512 x 512 resize and a
31 x 31 square dilation excluding foreground. The descriptor verification found
4.00390625 pixels for the thinnest CrackMap sample and 0.0561185181 for the
lowest-contrast DeepCrack sample, over 24 and 237 usable test images,
respectively. The samples themselves were not changed by that verification.
The `descriptor_verification.script` field names the companion paper's code
path; the measurement definitions are contained in this archive's JSON.

Operator counts for models with unsupported fused-attention or selective-scan
kernels are lower bounds, and are marked as such in the paper.

Every matrix cell (`public_ods_ALL.json`, `crack500_mother_vs_distributed.json`,
`public_ods_final_crack500_mother.json`) reports the last epoch of the fixed
50-epoch budget. The `public_ods_best_<dataset>.json` files score the *same*
runs at the checkpoint that selection by foreground IoU at a 0.5 threshold on
the validation split would have kept; the paper's Section 4.3 compares the two.
For DeepCrack that pair keeps its original names: `public_ods_fixed_deepcrack.json`
is the selected checkpoint and `public_ods_deepcrack_final.json` the final one.

The same fifteen configurations were also trained at one tenth of that budget,
and the paper reports the two arms against each other.
`budget_matrix_5_vs_50.json` holds the per-cell comparison: pixel-ODS at both
budgets, the between-model spread on each dataset, the rank changes, and the
Spearman correlation between the two orderings. `public_ods_ALL_5ep.json` is the
short arm's own matrix in the same layout as `public_ods_ALL.json`, so the
threshold statistics the paper quotes for the 5-epoch budget can be recomputed
from this release rather than only from the archived v1.3.0 record.

`fuse_order_contrast.json` records an implementation comparison between two
DeepCrack side-fusion orders with identical parameter counts. The counted
operation ratio is 2.717 (549.83 / 202.4 GFLOPs), but the accuracy and timing
comparison does not isolate fusion order. Fast-fuse used physical batch 8;
official-fuse used micro-batch 2 with four-step accumulation. Both applied an
extra division of mean-reduced BCE by physical batch size, so loss scaling also
differs. One seed per cell cannot establish accuracy equivalence; seed ranges
from other models cannot supply the uncertainty of this pair. All measured
scores, times and operation counts are retained.

## Initialization-stratified budget analysis

`benchmarks/budget_by_init.json` groups the same fifteen configurations by their
recorded initialization: six load pretrained weights and nine start from random
weights. It reports per-dataset and four-dataset mean gains, sensitivity to
excluding three short-budget collapse configurations, group gaps at each budget,
mean-rank changes and counted-operation summaries. No new model training is
introduced by this release.

The four-dataset mean gains are 1.4088 and 8.0324 pixel-ODS points for the two
groups, or 1.4088 and 3.1720 after excluding the three named configurations.
The pretrained-minus-scratch score gap is 8.5110 points at 5 epochs and 1.8874 at
50 epochs, a ratio of 4.51. Cross-group comparison counts (52/54 and 34/36)
reuse the same model runs and are not independent experimental pairs.

The standard median of the six pretrained models' counted GFLOPs is **135.1**;
176.5 was the upper middle order statistic, incorrectly described as the median
in the incoming manuscript analysis. The scratch-group median is 49.5. Counted
operations include lower-bound entries. The groups also differ in family and
scale, and U-Net alone cannot resolve those confounders. The stratification is
descriptive and does not estimate a causal pretraining effect or prove convergence.

Recompute using only Python's standard library and this archive:

```bash
python build_budget_by_init.py --budget-matrix benchmarks/budget_matrix_5_vs_50.json --params benchmarks/params_flops_512_merged.json --out budget_by_init_recomputed.json
```

The original per-budget input mode is retained for the training workspace;
its numerical output was checked against the portable archived-input mode.

Four files added in v1.5.0 record controls rather than matrix cells.
`crackmap_parent_vs_random.json` is the CrackMap partition control described
above. `budget_align.json` repeats two configurations under the other arm's
optimizer wrapper, separating "this model is stable" from "this model was
trained with gradient clipping". `shared_threshold_deepcrack.json` sweeps every
threshold that could be imposed on all fifteen models at once and records what
each one costs. `budget_by_init.json` stratifies the budget matrix by
initialization and by two capacity axes, so the short budget's apparent
initialization gap can be read against a capacity split of the same models.

`ops_eval.json` answers the two questions the ODS convention invites. Its
per-dataset entries hold, for every model, the score at a threshold selected on
data disjoint from the reported split, average precision over the same
1,000-bin sweep (which no threshold rule can move), and a 10,000-draw bootstrap of
the test set — source photographs rather than crops where the dataset is tiled —
giving a sampling interval per cell and a paired interval for every model pair.
`tango_priors_two_datasets.json` aligns the six matched TANGO contrasts across
the two datasets on which they were run, and
`threshold_curves_deepcrack.json` holds the pooled threshold sweeps the
threshold figure is drawn from.

## Versions

Each release is archived on Zenodo. The concept DOI
[10.5281/zenodo.22202977](https://doi.org/10.5281/zenodo.22202977) resolves to
the newest version. A fixed version DOI should be used when citing the exact
archive accompanying a manuscript submission.

Record metadata — authors, resource type, keywords, licence — is declared in
[`.zenodo.json`](.zenodo.json) rather than inferred by Zenodo from the GitHub
repository. **That file governs only releases archived after it was added;
Zenodo does not revise records that already exist.** The records for v1.0.0
through v1.3.0 keep what was inferred at the time — one creator taken from the
GitHub contributor list, no affiliations, no keywords, and `software` as the
resource type — and adding the file does not correct them. Each record's title
and description are still taken from its GitHub release, which is why
`.zenodo.json` declares neither.

The accompanying manuscript is not yet published and has no DOI, so no related
identifier points to it. One will be declared in `.zenodo.json` once that DOI
exists, and will appear on versions archived from that point onward.

- **v1.6.0 (2026-09-11)** — CrackMap repartitioned by source photograph and the
  whole column retrained, plus four control arms and a threshold-free ranking.
  The CrackMap indices shipped up to v1.4.0 were assigned per crop, which split
  18 of the 54 GoPro source photographs across roles and left 15 of the 24 test
  crops with a sibling crop in training. Reassigning whole photographs, at
  unchanged 84 / 12 / 24 counts, and retraining all fifteen models lowers the
  median model by 1.27 pixel-ODS points, changes 11 of the fifteen ranks, and
  contracts the column's between-model spread from 8.80 to 5.51 points — the
  first partition rebuild in this study that changes conclusions rather than
  decimals. Both partitions ship, and
  `benchmarks/crackmap_parent_vs_random.json` holds both columns side by side so
  the control can be recomputed. `benchmarks/public_ods_ALL.json` now carries the
  source-disjoint CrackMap column.

  The four TANGO crack-prior switches, previously run on DeepCrack only, were
  repeated at three seeds each on the parent-disjoint Crack500 partition, so all
  six matched contrasts now exist on two datasets
  (`benchmarks/tango_priors_two_datasets.json`,
  `benchmarks/tango_arms_ods_crack500_mother.json`). On Crack500 the reference
  arm varies by 0.18 points across seeds against 0.68 on DeepCrack, and against
  that smaller background removing orientation supervision costs 0.27 points and
  removing the centerline-Dice loss gains 0.17 while costing 1.35 points of
  centerline Dice; on DeepCrack none of the switches separates from seed
  variation. `benchmarks/budget_align.json` retrains TANGO and MixerCSeg
  faithful under the other arm's optimizer wrapper, which shows the stability
  reported for TANGO is not an artifact of its gradient clipping (84.51 against
  84.49 mean pixel-ODS at 50 epochs).

  `benchmarks/ops_eval.json` adds the measurement controls: thresholds selected
  on held-out data, average precision as a threshold-free ranking, and a
  bootstrap over the test sets. `benchmarks/shared_threshold_deepcrack.json`,
  `benchmarks/budget_by_init.json` and
  `benchmarks/threshold_curves_deepcrack.json` complete the set.

- **v1.5.0 (2026-09-10)** — add initialization-stratified budget analysis in
  `benchmarks/budget_by_init.json` and its standalone regeneration script.
  Correct the even-sized pretrained group's GFLOPs median to 135.1 and keep
  group comparisons descriptive. Existing benchmark scores, seed results,
  split indices, manifests and the v1.4.1 metadata corrections are unchanged.

- **v1.4.1 (2026-09-10)** — metadata corrections for the submission audit.
  `qualitative_selection.json` now records the four current figure samples and
  reproducible input-descriptor definitions; the earlier contrast value 0.057
  and usable count 230 are superseded by 0.0561185181 and 237 under the explicit
  resize/dilation rule. `fuse_order_contrast.json` now discloses physical batch,
  accumulation and loss-scaling differences and labels the result as an
  implementation comparison. The benchmark matrix, per-seed scores, operation
  counts, timing observations, partitions, manifests and scripts are unchanged.
  The README also corrects two interpretations in the prior version summary:
  1e-6 is a nonzero learning-rate floor at both budgets, and comparisons between
  different datasets do not isolate update count from image difficulty.

- **v1.4.0 (2026-09-09)** — the training budget of the whole matrix raised from
  5 epochs to 50, and the 5-epoch matrix kept as the short arm of a budget
  comparison. At both budgets the un-warmed polynomial schedule reached its
  nonzero floor of 1e-6 in the final epoch; that epoch occupies one fifth of
  the short budget and one fiftieth of the long budget. All
  fifteen models were retrained on the four datasets and on the parent-disjoint
  Crack500 rebuild — 75 runs on one machine under one recipe — and both the final
  and the selected checkpoint of every run were scored. Between-model differences
  shrink sharply on three datasets under the longer budget. Dataset size and
  image characteristics vary together, so these comparisons do not identify
  their separate contributions. The spread across
  the fifteen models falls from 58.47 to 8.80 pixel-ODS on CrackMap (about 550
  updates at 50 epochs) and from 20.42 to 4.22 on DeepCrack, while Crack500
  (14,700 updates) moves by 0.24 points on average. Changed files:
  `public_ods_ALL.json`, `public_ods_deepcrack_final.json`,
  `public_ods_fixed_deepcrack.json`, `public_ods_final_crack500_mother.json`,
  `public_ods_best_{crack500,camcrack789,crackmap,crack500_mother}.json`,
  `crack500_mother_vs_distributed.json`, and the thresholds in
  `qualitative_selection.json`. New files: `budget_matrix_5_vs_50.json`,
  `public_ods_ALL_5ep.json` and `fuse_order_contrast.json` (see *Benchmark JSON*
  above). One recipe difference is worth naming: RINDNet ran at micro-batches of
  4 with two accumulation steps in the 50-epoch arm and at an undivided batch of
  8 in the 5-epoch arm, so its two arms differ in more than epoch count.
  `params_flops_512_merged.json`, `budget_ods_deepcrack.json`, the two TANGO arm
  files, `mixercseg_official_parity.json`, the split indices, the manifests, and
  the scripts are unchanged.

- **v1.3.0 (2026-09-05)** — the whole model × dataset matrix retrained and
  re-scored under one checkpoint rule: every cell now reports the last epoch of
  the fixed 5-epoch budget. The earlier rule selected checkpoints by foreground
  IoU at a 0.5 threshold on the validation split (DeepCrack excepted). That
  criterion was constant across all five epochs for three model–dataset pairs
  (MixerCSeg faithful on Crack500 and CrackMap, DTrC-Net on CrackMap) and so
  silently returned the first epoch; the rule was replaced after this was found,
  and the paper says so. All fifteen models were retrained on the four datasets
  and on the parent-disjoint Crack500 rebuild — 75 runs on one machine under one
  recipe, which also brought ConvNeXt-XLarge to the same micro-batch schedule on
  every dataset — and both the final and the selected checkpoint of every run
  were scored. Changed files: `public_ods_ALL.json`,
  `public_ods_deepcrack_final.json`, `public_ods_fixed_deepcrack.json`,
  `crack500_mother_vs_distributed.json`, and the thresholds and checkpoint rule
  in `qualitative_selection.json`. New files:
  `public_ods_best_{crack500,camcrack789,crackmap,crack500_mother}.json` and
  `public_ods_final_crack500_mother.json` (see *Benchmark JSON* above).
  `params_flops_512_merged.json`, `budget_ods_deepcrack.json`, the two TANGO
  arm files, `mixercseg_official_parity.json`, the split indices, the manifests,
  and the scripts are unchanged.

- **v1.2.0 (2026-09-03)** — both MixerCSeg rows re-evaluated after a defect in
  our port of that architecture was corrected: every group normalization layer
  used eight groups, whereas the published implementation uses one group per
  eight channels. Loading the authors' released weights into both
  implementations gave a logit correlation of 0.40 before the correction and
  0.996 after, so the two arms were retrained on all five splits and on the
  DeepCrack budget seeds. Changed files: `public_ods_ALL.json`,
  `public_ods_deepcrack_final.json`, `public_ods_fixed_deepcrack.json`,
  `crack500_mother_vs_distributed.json`, `budget_ods_deepcrack.json`, and the
  MixerCSeg thresholds in `qualitative_selection.json`. New file:
  `mixercseg_official_parity.json`, the per-layer comparison and the released
  weights scored with this study's evaluator (DeepCrack 87.43, Crack500 77.56,
  CamCrack789 82.06, CrackMap 78.30 pixel-ODS). `params_flops_512_merged.json`
  is unchanged, since the group count changes neither parameters nor operators.
  Every other model's entry, the split indices, the manifests, and the scripts
  are unchanged.
- **v1.1.0 (2026-09-03)** — DTrC-Net entries re-evaluated after its Transformer
  branch was retrained from the public DeiT-tiny-distilled ImageNet-1k weights.
  The published DTrC-Net uses a 256-dimensional DeiT variant whose weights were
  never released, so the v1.0.0 entries had that branch randomly initialised;
  the manuscript now marks the row *after* Xiang et al. and describes the
  substitution. Changed files: `public_ods_ALL.json`,
  `public_ods_deepcrack_final.json`, `public_ods_fixed_deepcrack.json`,
  `crack500_mother_vs_distributed.json`, `params_flops_512_merged.json`, and
  the DTrC-Net thresholds in `qualitative_selection.json`. Every other model's
  entry, the split indices, the manifests, and the scripts are unchanged.
- **v1.0.0 (2026-08-31)** — initial release.

## Source datasets

The images and annotations must be obtained from their original releases:

- **DeepCrack** — Liu et al. (2019), *Neurocomputing* 338: 139–153
- **Crack500** — Yang et al. (2020), *IEEE T-ITS* 21 (4): 1525–1535
- **CamCrack789** — Zhu et al. (2024), *Computer-Aided Civil and Infrastructure Engineering* 39 (12): 1743–1765
- **CrackMap** — Katsamenis et al. (2023), *Advances in Visual Computing*, LNCS, 199–209

## Licence

The metadata, indices, and manifests in this repository are released under
CC BY 4.0. The Python scripts are released under the MIT licence. Neither
licence extends to the source datasets, which remain under the terms set by
their respective authors.
