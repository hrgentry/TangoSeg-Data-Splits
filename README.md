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
| `manifests/*.sha256` | Per-file SHA-256 of the audited copy of each dataset (DeepCrack 1,078; Crack500 6,742; CamCrack789 1,581; CrackMap 243 files) |
| `benchmarks/*.json` | Machine-readable results: the full model × dataset matrix at the prespecified final epoch, the same runs at the checkpoint a validation-selection rule would have kept, the threshold-sweep and fixed-threshold variants, the seed-repeated budget arms, the matched TANGO arms, the two Crack500 partitions side by side, and parameter/FLOP counts |
| `verify_manifests.py` | Checks a local dataset copy against a manifest |
| `make_crack500_mother_split.py` | Regenerates the parent-disjoint Crack500 partition (vendored verbatim; see below) |

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
records the morphology-first rule by which the qualitative figure's samples were
fixed before any prediction was viewed.

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

`fuse_order_contrast.json` holds one single-variable comparison drawn from this
matrix. The two DeepCrack side-fusion orders have identical parameter counts,
tensor for tensor, and differ only in whether the fusion convolution runs before
or after the upsample. The file records both orders' pixel-ODS on all five
splits at the 50-epoch budget together with their counted operations, and is the
basis of the computation-versus-accuracy statement in the paper's capacity
section.

## Versions

Each release is archived on Zenodo. The concept DOI
[10.5281/zenodo.22202977](https://doi.org/10.5281/zenodo.22202977) resolves to
the newest version, and is the DOI cited in the accompanying study.

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

- **v1.4.0 (2026-09-09)** — the training budget of the whole matrix raised from
  5 epochs to 50, and the 5-epoch matrix kept as the short arm of a budget
  comparison. Under the 5-epoch schedule the polynomial decay reached its floor
  of 1e-6 in the final epoch, so "the last epoch" was both the end of the budget
  and the point at which the learning rate had gone to zero; at 50 epochs the
  last epoch means the budget is spent, not that learning has stopped. All
  fifteen models were retrained on the four datasets and on the parent-disjoint
  Crack500 rebuild — 75 runs on one machine under one recipe — and both the final
  and the selected checkpoint of every run were scored. Between-model differences
  shrink sharply with the longer budget, and by an amount that tracks the number
  of optimizer updates rather than any property of the imagery: the spread across
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
CC BY 4.0. The two Python scripts are released under the MIT licence. Neither
licence extends to the source datasets, which remain under the terms set by
their respective authors.
