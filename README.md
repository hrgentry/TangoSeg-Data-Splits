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
| `splits/deepcrack/` | The DeepCrack index files this study used: 300 / 237 / 237, plus the two halves of the test set used for the held-out-threshold control. The original release ships no validation split and no index files, so these are local artifacts---and `val.txt` and `test.txt` are byte-identical, which is the defect the study reports |
| `splits/crackmap_source_disjoint/` | The source-disjoint CrackMap partition, used for every CrackMap number from v1.5.0 on: 84 / 12 / 24 crops over 40 / 4 / 10 mutually exclusive GoPro source photographs |
| `splits/crackmap_original/` | The image-level CrackMap partition used up to v1.4.0, kept so the partition control can be recomputed: the same 84 / 12 / 24 crop counts, but 15 of the 24 test crops share a source photograph with the training set |
| `manifests/*.sha256` | Per-file SHA-256 of the audited copy of each dataset (DeepCrack 1,078; Crack500 6,742; CamCrack789 1,581; CrackMap 243 files) |
| `benchmarks/*.json` | Machine-readable results: the full model × dataset matrix at the prespecified final epoch, the same runs at the checkpoint a validation-selection rule would have kept, the threshold-sweep and fixed-threshold variants, the seed-repeated budget arms, the matched TANGO arms, the two Crack500 partitions side by side, parameter/FLOP counts, and the measured per-image latency and peak inference memory |
| `benchmarks/erf_arrays.npz`, `benchmarks/erf_arrays_crackprobe.npz` | The averaged gradient fields behind the effective-receptive-field figure, under the two probe placements |
| `build_budget_by_init.py` | Recomputes the initialization-stratified budget summaries from the archived budget matrix and counted operations |
| `erf_radii.py` | Recomputes the published receptive-field radii from those fields, and checks them against what the manuscript prints |
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

## DeepCrack indices

The DeepCrack release contains four directories---`train_img`, `train_lab`,
`test_img`, `test_lab`, holding 300 and 237 images---and no index files, no
validation split and no README describing one. Everything downstream of "which
images are validation" is therefore a local decision, and this study's decision
was a defective one: `splits/deepcrack/val.txt` and `splits/deepcrack/test.txt`
are the same 237 lines in the same order, SHA-256

```
0b8d4cd53e0d01ae3234450a622159bf54d24a86b6579e2552f262e74d372fb2
```

so any rule that selects a checkpoint on validation selected it on the test set.
The study reports what that is worth rather than repairing it, and publishes the
files so the claim can be checked rather than taken on trust.
`test_halfA.txt` and `test_halfB.txt` split the test set 119/118 for the control
that selects an operating point on one half and reports on the other.

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
0.5, the evaluated image count, parameter and FLOP counts, and the measured
per-image latency and peak inference memory. `qualitative_selection.json`
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
CrackMap has two partitions and therefore two selected-checkpoint files:
`public_ods_best_crackmap_parent.json` scores the source-disjoint rebuild, which
is the partition `public_ods_ALL.json` and the paper's matrix report, and
`public_ods_best_crackmap.json` scores the image-level split it shipped with
before that rebuild. Compare the matrix cell against the parent file; the
image-level one is kept because earlier records cite it, not because it pairs
with the current column.

The same sixteen configurations were also trained at one tenth of that budget,
and the paper reports the two arms against each other.
`budget_matrix_5_vs_50.json` holds the per-cell comparison: pixel-ODS at both
budgets, the between-model spread on each dataset, the rank changes, and the
Spearman correlation between the two orderings. `public_ods_ALL_5ep.json` is the
short arm's own matrix in the same layout as `public_ods_ALL.json`, so the
threshold statistics the paper quotes for the 5-epoch budget can be recomputed
from this release rather than only from the archived v1.3.0 record.

`latency_b8.json` holds the measured inference cost behind the last two
columns of the comparative table. Latency and peak memory are not properties
of a model the way ODS and FLOP counts are — they hold only on the machine
that produced them — so the file records the conditions next to the numbers:
the card and driver, the PyTorch and CUDA build, the batch size, whether cuDNN
autotuning and TF32 were enabled, and the warm-up and iteration counts. All
seventeen entries come from one card in one sweep, and measurements from a
second machine must not be merged into it.

Three fields are worth reading before the latencies themselves.
`reproducibility` gives the worst spread of any model over three independent
passes, 2.55 per cent. `batch1_control` repeats the sweep at batch one, where
that spread reaches 46.1 per cent by the median and 53.7 per cent by the
minimum, which is why the published column is measured at batch eight and
reported as the minimum rather than the mean. `mamba_layernorm_aliased`
records that the Mamba layer-norm module was aliased to the name its 2.x
release uses; without that alias SCSegamba silently substitutes a pure-PyTorch
scan and one forward pass takes about four seconds instead of the 56 ms that
`batch1_control` records for it, with nothing in any log to say it happened.
A latency sweep that does not assert its kernels can report an architecture as
slow when what it measured was a fallback.

`derived` holds the comparison between counted operations and measured time
that the manuscript quotes: across the sixteen configurations in the table the
two orderings disagree on 34 of the 120 pairs, at a rank correlation of 0.479,
and the largest inversion has RINDNet counted at 127 times the operations of
the MixerCSeg faithful port while running 1.07 times faster.

`fuse_order_contrast.json` records an implementation comparison between two
DeepCrack side-fusion orders with identical parameter counts. The counted
operation ratio is 2.717 (549.83 / 202.4 GFLOPs), but the accuracy and timing
comparison does not isolate fusion order. Fast-fuse used physical batch 8;
official-fuse used micro-batch 2 with four-step accumulation. Both applied an
extra division of mean-reduced BCE by physical batch size, so loss scaling also
differs. One seed per cell cannot establish accuracy equivalence; seed ranges
from other models cannot supply the uncertainty of this pair. All measured
scores, times and operation counts are retained.

`latency_b8.json` adds the one timing comparison of this pair that is like
for like: both arms were measured on the same card at batch eight, where
official-fuse takes 2.92 times as long per image as fast-fuse against the 2.72
ratio in counted operations. That pair of ratios is a comparison of fusion
orders at equal parameter count; it still says nothing about the accuracy
difference, which remains confounded as described above.

## Initialization-stratified budget analysis

`benchmarks/budget_by_init.json` groups the same sixteen configurations by their
recorded initialization: six load pretrained weights and ten start from random
weights. It reports per-dataset and four-dataset mean gains, sensitivity to
excluding three short-budget collapse configurations, group gaps at each budget,
mean-rank changes and counted-operation summaries. No new model training is
introduced by this release.

The four-dataset mean gains are 0.9967 and 7.9255 pixel-ODS points for the two
groups, or 0.9967 and 3.4349 after excluding the three named configurations.
The pretrained-minus-scratch score gap is 8.5409 points at 5 epochs and 1.6121 at
50 epochs, a ratio of 5.3. Cross-group comparison counts (58/60 and 40/42)
reuse the same model runs and are not independent experimental pairs. The four
cells averaged here are the parent-disjoint ones, listed in the file's own
`mean_over` field; the distributed Crack500 column is reported per dataset and
left out of the mean, as it is in the manuscript's matrix table.

The standard median of the six pretrained models' counted GFLOPs is **135.1**;
176.5 was the upper middle order statistic, incorrectly described as the median
in the incoming manuscript analysis. The scratch-group median is 36.7. Counted
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
threshold that could be imposed on all sixteen models at once and records what
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

## Effective receptive fields

`benchmarks/erf_arrays.npz` holds the averaged gradient fields behind the
receptive-field figure: for each model, the gradient of the crack logit at a
probe pixel with respect to the input, magnitude summed over input channels and
averaged over the same 50 DeepCrack test images, following the gradient
definition of Luo et al. (2016). Four fields are the figure's panels; three more
are TANGO ablation arms that the manuscript reports as numbers rather than
panels — the reference arm and the two switches that remove the propagation and
replace anisotropic steering with uniform weights, all at seed 42, so the three
form single-variable contrasts. The `meta` record carries the split, image
count, configuration and checkpoint path per model, and the probe statistics.

**The archived arrays are raw fields, and a radius read off a raw field is not
the number the manuscript prints.** Every field sits on a near-uniform floor —
normalization layers that aggregate over the spatial dimensions give each input
pixel a small gradient regardless of kernel or attention span — and the
contribution of that floor to the accumulated mass grows with the square of the
radius. For two of the four models the uncorrected r75 measures that floor
rather than the model: between 128 and 250 pixels the radial profile is flat for
DTrC-Net and rises for MixerCSeg, while it falls by a factor of 8.4 for
ConvNeXt-XLarge and 3.1 for TANGO. The published radii are therefore measured
after subtracting the floor, estimated per field as the median of its radial
profile between 220 and 250 pixels. Uncorrected, the four panel models summarize
as r75 = 41 / 191 / 151 / 165 pixels; corrected, 38 / 140 / 39 / 40.

Recompute and check, using only this archive (numpy is required to read an
`.npz`; nothing else is):

```bash
python erf_radii.py
```

The probe sits at the geometric centre of the image, which is a crack pixel in 5
of the 50 images, a median of 34 pixels from the nearest one, so the measurement
mostly describes the input dependence of a background prediction.
`benchmarks/erf_arrays_crackprobe.npz` repeats the four panel models with the
probe at the foreground pixel nearest the centre — 45 of the 50 images have one
within 128 pixels, and each gradient map is shifted back to the centre before
averaging — giving r75 = 38 / 143 / 39 / 30 and leaving the ordering unchanged.
Two limits apply to both files: the average runs over cracks at different
orientations, so it describes extent and not orientation structure, and DTrC-Net
runs its network at 256 x 256 and resamples its logits to 512, so its radii
describe that working domain.

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

- **v1.8.0 (2026-09-15)** — measured inference cost joins the comparative
  table. `benchmarks/latency_b8.json` is new and `public_ods_ALL.json` gains two
  fields per model, `latency_ms` and `peak_mem_GB`. No existing value in any
  file changed; every other file is byte-identical to v1.7.4.

  Until now the cost axis of the comparison was counted operations alone, which
  is a proxy. The measured columns are not a refinement of that proxy but a
  disagreement with it: over the sixteen configurations, counted operations and
  measured time rank 34 of the 120 pairs differently, at a rank correlation of
  0.479, and the largest inversion has RINDNet counted at 127 times the
  operations of the MixerCSeg faithful port while running 1.07 times faster.
  Two of the table’s columns previously could not be checked from this
  release at all.

  These two quantities travel worse than anything else here. ODS, parameter
  counts and FLOPs can be recomputed on any machine; a latency is a statement
  about one card under one driver at one batch size. `latency_b8.json`
  therefore ships the measurement conditions, a three-pass reproducibility
  spread, a batch-one control that explains the protocol rather than only
  asserting it, and a flag recording that the Mamba kernels were the real ones.
  Reproducing these numbers on different hardware should not be expected to
  agree, and the file says so rather than leaving it to be inferred.

- **v1.7.4 (2026-09-15)** — the effective-receptive-field arrays are released,
  together with the estimator that turns them into the published radii. Nothing
  in any existing file changed; every other file is byte-identical to v1.7.3.

  These arrays were listed as missing when v1.4.0 was audited and stayed missing
  through five releases, so the manuscript's receptive-field numbers were the one
  measurement a reader could not check. That gap mattered more than it looked:
  an adversarial review of the manuscript found that two of the four reported
  r75 values were measuring the near-uniform floor under every field rather than
  the field, and the correction moves them from 151 and 165 pixels to 39 and 40.
  Shipping the arrays alone would have handed a reader the uncorrected numbers,
  so `erf_radii.py` ships with them: it applies the floor subtraction described
  above, prints the corrected and uncorrected radii side by side, and fails if
  either stops matching the manuscript.

  Added in the same release: three TANGO ablation arms inside
  `erf_arrays.npz` — the reference arm and the two switches that remove the
  propagation and replace anisotropic steering with uniform weights, at seed 42
  — which give r75 = 141, 140 and 143, so the model's wide field does not come
  from the propagation it is named for; and
  `erf_arrays_crackprobe.npz`, the same four panel models measured with the probe
  on the crack instead of at the image centre, which leaves the radii and their
  ordering intact.

- **v1.7.3 (2026-09-14)** — `budget_by_init.json` joins the parent-disjoint
  convention, completing what v1.7.2 left half-done. No measurement was rerun.
  `build_budget_by_init.py` averaged its four-dataset gains over the distributed
  Crack500 column, while the manuscript's matrix table has taken its Mean column
  over the parent-disjoint rebuild since 2026-09-11. The manuscript quotes this
  file's group gap in the same section as that table, so the archive no longer
  reproduced the paper: recomputing from v1.7.2 gave 8.5144 and 1.5518 where the
  paper reports 8.54 and 1.61.

  The comment above `MEAN_DS` in the shipped script asserted that the
  distributed column was the one matching the table. That was true when it was
  written and stopped being true when the table changed under it, which is why
  the defect survived a partition rebuild — the script looked deliberate.

  Regenerated from the unchanged budget matrix: group gaps 8.5144 → 8.5409 at
  5 epochs and 1.5518 → 1.6121 at 50, ratio 5.49 → 5.3; group mean gains
  1.0279 → 0.9967 and 7.9905 → 7.9255, or 3.4253 → 3.4349 after excluding the
  three collapsed configurations; two ranks swap (TANGO and DTrC-Net trade
  places at both budgets); and the capacity split's light-side pair wins fall
  from 54 to 51 of 64 by parameters and from 42 to 40 by counted operations.
  The `per_dataset` block is identical value for value — only its key order
  follows the new roster — and every other file in the release is
  byte-identical to v1.7.2. The portable recompute command above still
  reproduces the shipped `budget_by_init.json` exactly.

  **This supersedes the closing note under v1.7.2**, which recorded that the
  archive still averaged `budget_by_init.json` over the distributed column.
  Every four-dataset mean in this archive is now taken over the parent-disjoint
  Crack500 cell. Each file still declares its own roster in its `mean_over`
  field, and reading that declaration is still the reliable check.

- **v1.7.2 (2026-09-14)** — one file changed, none added. The `mean_four_*`
  fields of `fuse_order_contrast.json` now average the parent-disjoint Crack500
  cell (`crack500_mother`) instead of the distributed one. Those are the four
  cells the paper's main matrix table takes its Mean column over, and the paper
  quotes this file's mean against that table's DeepCrack fast-fuse row, so two
  numbers meant to be read together were computed on different Crack500 test
  sets. Fast-fuse moves from 78.354 to 78.325, official-fuse from 78.02 to
  78.037, and the distance between the two fusion orders narrows from 0.334 to
  0.288 points. The file's `note` now names the four averaged cells and records
  that the distributed Crack500 cell is still reported per dataset but is left
  out of the mean.

  No measurement was rerun. Every `per_dataset` score and threshold, both
  operation counts, both wall-clock times and `max_abs_delta` are byte-identical
  to v1.7.1, as is every other file in this release. The three derived means and
  the `note` are the whole change.

  **This does not put every four-dataset mean in the archive on the
  parent-disjoint partition.** The gains in `budget_by_init.json` are still
  averaged over the distributed Crack500 column, which that file declares in its
  own `mean_over` field. Read that field rather than assuming one convention
  holds across files.

- **v1.7.1 (2026-09-13)** — one file added, none changed.
  `public_ods_best_crackmap_parent.json` scores the source-disjoint CrackMap
  rebuild at the checkpoint a validation-set rule would have selected, so the
  selected-versus-final comparison can be made on the partition the matrix
  actually reports. Until now CrackMap had only the image-level selected-checkpoint
  file, which pairs with a column the matrix stopped using in v1.6.0; the archive
  therefore invited a comparison across two different test sets. The image-level
  file is unchanged and stays: earlier records cite it, and rewriting what a
  published filename means is worse than adding the missing sibling, which is how
  Crack500 already ships both of its partitions. Every other file is byte-identical
  to v1.7.0.

  This release also carries the one commit pushed after the v1.7.0 tag:
  `build_budget_by_init.py` no longer restates the group sizes and pairing counts
  inside `pair_wins`'s explanation of why it reports pairs rather than an AUC.
  That text still said "n=6/9" and "54 of 52" after the sixteenth model arrived.
  It is a comment; the script's output is unchanged.

- **v1.7.0 (2026-09-13)** — a sixteenth model joins the comparative matrix and
  every whole-matrix artifact is regenerated. SCSegamba, a selective-scan
  (Mamba) crack segmentation network trained from random initialization at
  3.0 M parameters and 18.7 counted GFLOPs (a lower bound, its
  selective-scan kernel is uncounted), scores a 75.42 four-dataset mean
  pixel-ODS. No partition, manifest or source dataset changes, and no other
  model is retrained.

  **The addition reverses one published reading rather than extending it.** The
  CrackMap source-disjoint control of v1.6.0 contracted that column's
  between-model spread from 8.80 to 5.51 points over fifteen models; over
  sixteen it expands it to 17.99, entirely because this one model loses
  12.01 points on the rebuild and lands at 59.21 while the other
  15 stay within 5.51 points of one another. Readers comparing
  the two records should attribute the difference to the roster, not to a
  repartition.

  The initialization stratification moves with it: the pretrained-minus-scratch
  score gap at 50 epochs rises from 1.1634 to 1.5518 points and the
  short-to-long ratio falls from 7.44 to 5.49, because the new model is a
  from-scratch one that ends below the group it joins. Under the threshold rule
  14 of the sixteen DeepCrack ranks now change, against 13 of fifteen before.

  Regenerated: `public_ods_ALL.json`, `public_ods_ALL_5ep.json`,
  `public_ods_deepcrack_final.json`, `public_ods_fixed_deepcrack.json`,
  `public_ods_final_crack500_mother.json`,
  `public_ods_best_{crack500,camcrack789,crackmap,crack500_mother}.json`,
  `crack500_mother_vs_distributed.json`, `crackmap_parent_vs_random.json`,
  `budget_matrix_5_vs_50.json`, `budget_by_init.json`, `ops_eval.json`,
  `shared_threshold_deepcrack.json`, `threshold_curves_deepcrack.json` and
  `params_flops_512_merged.json`. `public_ods_ALL_5ep.json` gains three fields
  its previous build predated (`AP`, `F1_fixed`, `gflops_lower_bound`) and the
  four per-dataset provenance keys.

  **Rebuilding the short arm also corrects a partition mismatch that has been
  in the archive since v1.6.0.** That release moved the `crackmap` column of
  `public_ods_ALL.json` to the source-disjoint rebuild but left
  `public_ods_ALL_5ep.json` on the image-level split, so the two arms scored
  CrackMap on different test sets and the short arm disagreed with the
  `ODS_5` column of `budget_matrix_5_vs_50.json` for every model. Both arms
  now use `splits_parent/test.txt` and both reproduce their column of the
  budget matrix exactly. The DeepCrack, Crack500 and CamCrack789 values of
  the short arm are unchanged; every changed value is in its CrackMap column.

  `build_budget_by_init.py` is updated with it. The label table in the shipped
  script was fixed at fifteen entries, so the archive's own script would have
  refused the archive's own data with a roster-length error; its group sizes and
  pairing counts are now derived rather than written out. Running it in the
  portable mode documented above reproduces the shipped
  `benchmarks/budget_by_init.json` value for value.

- **v1.6.1 (2026-09-12)** — publish the DeepCrack index files. The original
  release ships no validation split and no index files, so the lists this study
  trained and scored on existed only locally; every DeepCrack number in the paper
  and two columns of its matrix were unreproducible without them. `val.txt` and
  `test.txt` are byte-identical, which is the checkpoint-selection defect the
  study reports, now checkable from the archive instead of asserted. The two test
  halves used for the held-out-threshold control ship with them. No results
  change.

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
