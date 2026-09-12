# Decision Log — `wsi-metastasis-seg`

Architecture Decision Records. Append-only. An ADR is never edited after it is
marked `Accepted`; it is **superseded** by a new ADR that references it.

**Status vocabulary:** `Proposed` · `Accepted` · `Superseded by ADR-nnn` ·
`Rejected` · `Deferred`

**Template**

```
## ADR-nnn — <title>
Status: · Date: · Owner:
### Context
### Options considered
### Decision
### Consequences
### Revisit trigger
```

---

## ADR-001 — WSI reading backend

**Status:** Accepted · **Date:** 2026-09-12

### Context
Every downstream module depends on random-access reads from pyramidal TIFFs
(`.tif`, `.svs`, `.ndpi`, `.mrxs`). The choice constrains deployment,
multiprocessing behaviour, and per-patch latency — the three things hardest to
change later.

### Options considered

| Backend | Pros | Cons |
|---|---|---|
| `openslide-python` | De-facto standard; widest vendor-format coverage (MRXS, NDPI, VMS, SCN); battle-tested geometry | Requires system `libopenslide` (apt/brew/conda); the handle is **not** fork-safe; returns RGBA; Windows install is painful |
| `tiffslide` | Pure Python (`tifffile` + `zarr` + `fsspec`); pip-installable; thread-safe; API-compatible drop-in; reads from object storage | Generic-TIFF only — no MRXS/NDPI/VMS; slightly slower on heavily-tiled JPEG2000 |
| `cuCIM` | GPU-accelerated decode, 3–5× faster on SVS; nvJPEG path | Linux + CUDA only; narrow format support; adds a heavyweight RAPIDS dependency |
| `pyvips` | Excellent for whole-image ops and pyramid writing | Awkward for millions of small random reads; another system library |

### Decision
Introduce a thin `SlideReader` façade in `src/io/slide.py` with a
**capability-ordered backend chain**:

1. `tiffslide` — default for generic pyramidal TIFF and SVS (covers all of
   CAMELYON16/17).
2. `openslide` — automatic fallback when `tiffslide` raises on an unsupported
   vendor container.
3. `cucim` — opt-in via `--backend cucim`, used only for batch inference.

The façade owns three normalisations so no caller ever repeats them:
- RGBA → RGB composited onto **white** (not black — see BUG-001).
- `mpp_x/mpp_y` resolved from vendor properties with a TIFF-resolution-tag
  fallback.
- Coordinate convention asserted: `location` is level-0, `size` is target-level.

### Consequences
- `pip install -e .` works with no system packages in the common case, which
  makes the CI container and the reviewer's laptop identical.
- The fallback path means two libraries can produce pixels for the same slide.
  Mitigated by V1.3: for 5 sample slides, `tiffslide` and `openslide` reads of
  the same region must match with `max|Δ| ≤ 1` (JPEG decoder rounding).
- `cuCIM` is never on the critical path, so it can be dropped without a rewrite.

### Revisit trigger
If >10 % of a new cohort is MRXS/NDPI, promote `openslide` to default and pin
`libopenslide` in the Docker image.

---

## ADR-002 — Magnification selected by MPP, not by level index

**Status:** Accepted · **Date:** 2026-09-12

### Context
"20x" is a marketing number, not a physical one. CAMELYON16 slides come from
two centres with different scanners; some have level 0 at 0.243 µm/px and
others at 0.226 µm/px. Hard-coding `level=1` silently trains the model at two
different physical scales, which is a strictly worse version of no
augmentation.

### Options considered
1. Hard-code `level = 1`. Simple; wrong across scanners.
2. Resolve the level whose MPP is closest to a target, accept the residual.
3. Resolve the nearest level, then **resample** to hit the target MPP exactly.

### Decision
Option 3, with a tolerance band.

```python
TARGET_MPP = 0.50                       # 20x
lw = argmin_l |mpp0 * downsample[l] - TARGET_MPP|
residual = mpp0 * downsample[lw] / TARGET_MPP
if abs(residual - 1) <= 0.05:  accept as-is      # ≤5 % scale error
else:                          cv2.resize(patch, ..., INTER_AREA)
```

Resolved `(level, residual)` is written per slide into
`artifacts/index/slides.parquet` and is part of the patch index contract.

### Consequences
- Physical scale is consistent to within 5 % across the cohort; this is inside
  the range covered by scale augmentation, so no further correction is needed.
- Resampling is rare (it fires on ~7 % of CAMELYON16) and its cost is
  amortised into the DataLoader workers.
- A slide with missing MPP metadata is a **hard failure**, not a guess. It goes
  to `failed_slides.csv`. Guessing scale is worse than dropping a slide.

### Revisit trigger
A cohort where >20 % of slides lack MPP metadata.

---

## ADR-003 — Tissue localisation algorithm

**Status:** Accepted · **Date:** 2026-09-12

### Context
Typically 60–85 % of a WSI is blank glass. Segmenting it is pure waste, and
including it in training teaches the model nothing while dominating the loss.
The filter must be fast (it runs on a thumbnail), robust to stain variation,
and must not swallow faintly-stained tissue.

### Options considered

| Method | Verdict |
|---|---|
| Fixed RGB threshold (`R,G,B > 220` → background) | **Rejected.** Fails on under-stained and on dark/blue-filtered scans; no adaptation. |
| Otsu on grayscale alone | Partially works; merges pale tissue into background when the slide is mostly glass (unimodal histogram → meaningless threshold). |
| Otsu on HSV **saturation** | Strong: glass is achromatic (S≈0), H&E tissue is chromatic. Weak on very pale eosin-only regions. |
| Union of the two + morphology | **Selected.** Each covers the other's failure mode. |
| Learned tissue segmenter (small UNet) | **Rejected for v1.** Needs its own labels and its own failure analysis for a problem that thresholding solves at ~98 % IoU. Revisit only if artifact rejection becomes the bottleneck. |

### Decision
Pipeline in `src/preprocess/tissue.py`, executed at the pyramid level nearest
to **8.0 µm/px**:

```
1. read thumbnail                                → RGB uint8
2. median blur k=7                               → suppress dust/scan noise
3. S = HSV(img)[..., 1];  G = 255 - gray(img)
4. m_s = S > otsu(S)  ;  m_g = G > otsu(G)
5. tissue = m_s | m_g
6. tissue &= ~pen_mask                           # see below
7. binary_closing(disk 5) → binary_opening(disk 3)
8. remove_small_objects(min_area = 0.02 mm² in thumbnail px)
9. binary_fill_holes
```

**Pen / artifact rejection.** Marker ink is highly saturated but has a hue far
from H&E. Reject pixels with `S > 0.35` and hue outside the H&E band
(`H ∈ [0.70, 0.95] ∪ [0.0, 0.10]` in normalised OpenCV hue). Also reject
near-black `V < 0.20` (slide edges, coverslip shadow).

**Bimodality guard.** Otsu on a unimodal histogram returns an arbitrary split.
Before accepting, require inter-class variance ratio `> 0.08`; otherwise fall
back to a fixed conservative threshold and flag the slide for QC.

### Consequences
- Patch count drops ~4× versus naive gridding; effective epoch time drops with it.
- Post-condition `0.01 ≤ tissue.mean() ≤ 0.90` is enforced (contract A2). Both
  bounds have fired in practice (BUG-004, BUG-005).
- Thumbnail-resolution masks mean patch gating is accurate to ±8 µm at the
  tissue border. Irrelevant: a `tissue_frac > 0.10` threshold is used for
  inclusion, not a hard boundary.

### Revisit trigger
Pen-mark false positives exceed 2 % of indexed patches on any cohort.

---

## ADR-004 — Patch size 512 and stride policy

**Status:** Accepted · **Date:** 2026-09-12

### Context
Patch size trades field of view against memory and against boundary artifacts.
Stride trades inference cost against seam quality.

### Options considered

| Size @ 0.5 µm/px | FOV | Notes |
|---|---|---|
| 256 px | 128 µm | More patches, more borders, less context; a single metastatic nest can exceed the FOV |
| **512 px** | **256 µm** | Holds an entire micrometastasis (0.2–2.0 mm needs 512–4096 px, but 256 µm captures the nest core and its interface) |
| 1024 px | 512 µm | Best context; batch drops to 4 at 16 GB, and border-pixel fraction improves only marginally |

### Decision
- **Patch size 512 × 512 at 0.50 µm/px** (256 µm FOV).
- **Training stride 512** (non-overlapping grid) — overlap at train time is a
  weak augmentation and inflates epoch time. Positional diversity comes instead
  from a random jitter of `±128 px` applied per epoch to each coordinate.
- **Inference stride 256** (50 % overlap) — required for the stitching
  mathematics in `Architecture.md` §6; every interior pixel is covered by 4
  windows.

### Consequences
- Inference cost is 4× a non-overlapping pass. Budgeted: ~2.5 min/slide
  (Architecture §5), acceptable for a research pipeline.
- Border-pixel fraction at 512 with a 512-receptive-field-ish decoder is ~20 %,
  which is exactly why the Gaussian window (σ = 64) is not optional.
- Jitter at train time means the parquet index stores grid anchors, and the
  `Dataset` adds the offset — so the index stays deterministic and cacheable.

### Revisit trigger
Moving to a multi-scale model, or if FROC analysis shows systematic misses on
lesions larger than the FOV.

---

## ADR-005 — Coordinate index, not pre-extracted patch files

**Status:** Accepted · **Date:** 2026-09-12

### Context
The classic WSI pipeline dumps millions of PNG/JPEG patches to disk before
training. The alternative is to persist only coordinates and read windows
lazily inside DataLoader workers.

### Options considered

| Strategy | Disk | Random-read latency | Reproducibility |
|---|---|---|---|
| PNG/JPEG files on disk | ~2.1 TB for CAMELYON16 @512/20x | ~1 ms | Good, but the extraction params are frozen into the bytes |
| HDF5 / LMDB shards | ~1.4 TB | ~0.4 ms | Good; single-writer friction |
| **Coordinate parquet + lazy `read_region`** | **~120 MB** | **~12 ms** | Excellent — re-index is a 4-min job |

### Decision
Persist coordinates only. Materialise pixels inside the workers.

Justified quantitatively by the throughput budget: with 8 workers, lazy reads
deliver ~660 patches/s against a GPU ceiling of ~36 patches/s. **The IO path
has 18× headroom, so the 12 ms latency is invisible.** Paying 2 TB and a
6-hour extraction job to optimise a non-bottleneck is a bad trade.

An optional LMDB cache (`--cache-lmdb`) exists for the balanced training subset
(~180 GB) and is used only when the run is IO-bound, e.g. on a multi-GPU node
where the GPU ceiling rises above the IO ceiling.

### Consequences
- Changing patch size, stride, or tissue threshold costs one re-index, not one
  re-extraction.
- **Slide handles must be opened lazily per worker.** A handle created in the
  parent and inherited across `fork` corrupts reads or segfaults (BUG-002).
  Enforced by a `worker_init_fn` and a `_handle: dict[int, SlideReader]` keyed
  by PID.
- The raw WSI archive becomes a hard runtime dependency of training, not just
  of preprocessing. Recorded in `rollback.md` R6 as a recovery consideration.

### Revisit trigger
Multi-GPU training where measured GPU utilisation drops below 85 %.

---

## ADR-006 — Class imbalance strategy

**Status:** Accepted · **Date:** 2026-09-12

### Context
After tissue filtering, tumor pixels are still roughly **0.3–2 %** of indexed
pixels on tumor slides, and CAMELYON16 contains ~160 normal slides with zero
tumor pixels. Uniform sampling yields batches that are almost entirely
negative; the model converges to predicting all-zero, which scores a fine BCE
and a Dice of 0.

### Options considered
- Loss reweighting alone (`pos_weight` in BCE). Simple, but gradients still
  come from overwhelmingly negative crops; unstable at `pos_weight > 50`.
- Sampling alone. Fixes batch composition but biases the prior — the model sees
  a tumor-rich world and over-predicts at inference.
- Hard negative mining only. Effective late, useless at initialisation.
- **Combination, staged.**

### Decision
Three mechanisms, applied in stages:

1. **Balanced sampler** (`src/data/sampler.py`): each epoch draws
   `1 tumor : 3 normal`. "Tumor" means `tumor_frac > 0.05`; patches with
   `0 < tumor_frac ≤ 0.05` (boundary slivers) are placed in a third bucket
   sampled at 10 % to avoid teaching the model that faint evidence is negative.
2. **Loss** = `0.5 · BCEWithLogits(pos_weight=2.0) + 0.5 · SoftDice`.
   - Pure Dice was **rejected**: on all-negative patches the gradient is
     degenerate and the loss oscillates (BUG-006).
   - Dice is computed per-image with `eps=1.0` in numerator and denominator, so
     an empty prediction on an empty mask scores 1.0 rather than NaN.
   - Focal loss (γ=2) kept as an ablation in `losses.py`; it underperformed
     BCE+Dice by 0.9 Dice points in a 10-epoch pilot.
3. **Hard negative mining** from epoch 8: the top 2 % of normal patches by
   false-positive area are promoted into the tumor-sampling pool for the
   following epoch. Refreshed every 4 epochs, capped at 15 % of the pool so the
   sampler cannot collapse onto a few pathological slides.

**Prior correction.** Because sampling distorts the prior, the decision
threshold τ is *not* fixed at 0.5. It is fitted on the validation set to
maximise slide-level F1, and the fitted value is stored in the checkpoint.

### Consequences
- Epoch length is defined by the sampler (`num_samples = 4 × |tumor pool|`),
  not by dataset size, so epochs are comparable across configs.
- Hard negative mining requires an inference pass over a normal-slide subset
  every 4 epochs: ~6 min, ~4 % overhead.
- τ becomes a checkpoint-coupled artifact. Evaluating a checkpoint with the
  wrong τ is a silent 5–10 point F1 error, so `eval/report.py` refuses to run
  if `ckpt.threshold` is absent.

### Revisit trigger
If the fitted τ drifts outside `[0.3, 0.7]`, the sampling ratio is wrong.

---

## ADR-007 — Model: UNet + EfficientNet-B0 first; Mask2Former deferred

**Status:** Accepted · **Date:** 2026-09-12

### Context
The problem statement admits "a CNN or Vision Transformer". Both are
defensible; only one should be the baseline.

### Options considered

| Model | Params | Fit to task |
|---|---|---|
| **UNet + EfficientNet-B0** (`segmentation_models_pytorch`) | 6.3 M | Dense binary semantic segmentation at 512²; ImageNet init transfers well to H&E texture; AMP-friendly; ~450 ms/step at batch 16 |
| UNet + ResNet-34 | 24 M | Strong, 3.8× the params for ~equal Dice in published WSI work |
| DeepLabV3+ | 26 M | Atrous context helps at larger FOV; at 256 µm FOV the benefit is small |
| SegFormer-B0 | 3.8 M | Competitive, good global context; less mature H&E tooling |
| **Mask2Former** | 44 M | Query-based **instance/panoptic** formulation. Powerful, but this is a **binary dense** task with no instance labels — the mask-query machinery is mostly unused capacity, and it needs ~4× the step time |

### Decision
- **Baseline:** UNet + EfficientNet-B0, ImageNet-pretrained encoder, single
  logit channel, deep supervision off.
- **Mask2Former: Deferred**, not rejected. Kept as `models/mask2former.py`
  behind the registry so the ablation is a config switch. It becomes relevant
  if the task is extended to lesion-instance counting (which FROC arguably
  wants) rather than pixel masks.
- Encoder frozen for the first 2 epochs (BN in eval mode), then unfrozen at
  `0.1 × lr` — stabilises early training when the balanced sampler is feeding
  an unusual class prior.

### Consequences
- A 14-hour single-GPU training budget stays intact, which keeps the
  experiment loop short enough to actually iterate.
- Committing to `segmentation_models_pytorch` couples the project to that API;
  the `models/registry.py` indirection keeps the blast radius to one file.

### Revisit trigger
Baseline validation Dice plateaus below 0.75 with no remaining data-side fixes.

---

## ADR-008 — Stain augmentation over test-time stain normalisation

**Status:** Accepted · **Date:** 2026-09-12

### Context
H&E appearance varies substantially across scanners, labs, and staining
batches. Two families of fix: normalise everything to a reference at test time
(Macenko, Vahadane, Reinhard), or train the model to be invariant.

### Options considered
- **Macenko normalisation** on every patch. Deterministic target appearance,
  but ~25 ms/patch (SVD per patch), fails on patches with little tissue, and
  introduces a reference-slide dependency that must ship with the model.
- **Reinhard** (LAB mean/std matching). Cheap, but crude; can invert contrast
  on unusual slides.
- **HED colour jitter** at training time: RGB → optical density → HED
  deconvolution, perturb H and E channels by `α ∼ U(0.95, 1.05)`,
  `β ∼ U(-0.05, 0.05)`, invert.

### Decision
HED jitter as the default training augmentation; **no test-time normalisation**
in the default path. Macenko is available as `--stain-norm macenko` for
cross-centre evaluation (e.g. training on CAMELYON16 and testing on CAMELYON17
centres), where the domain gap is a stated experimental variable rather than
noise.

### Consequences
- Inference is faster and has no reference-image dependency to version.
- Cross-centre generalisation must be reported explicitly, since the default
  path does nothing to guarantee it.
- The augmentation must be applied **before** ImageNet normalisation and
  **never** to the mask — enforced by keeping it in an image-only transform
  branch (BUG-011 was exactly this mistake).

### Revisit trigger
Cross-centre Dice drops more than 8 points versus in-centre.

---

## ADR-009 — Heatmap storage: chunked zarr at 8× downsample

**Status:** Accepted · **Date:** 2026-09-12

### Context
The deliverable is a gigapixel spatial probability heatmap. Naively this is a
float array the size of the slide.

```
45,000 × 105,000 float32 = 18.9 GB per slide      ✗ (per slide!)
```

### Options considered
1. Full-resolution numpy in RAM. Impossible.
2. Full-resolution memory-mapped numpy. 18.9 GB/slide on disk, no compression,
   poor chunk locality for the write pattern.
3. Downsample on write + chunked, compressed zarr.
4. Store only the patch-grid probabilities (one value per window).

### Decision
Option 3, with option 4 also retained as a cheap side-car.

- Accumulate `num` and `den` as **float32 zarr**, chunk `(1024, 1024)`,
  at **8× downsample from level 0** (= 2.0 µm/px, given 0.25 µm/px level 0).
- Final `heat = num/den` cast to **float16**, compressed blosc-zstd level 5.
- Also persist `grid.parquet`: `(x0, y0, prob_mean, prob_max)` per window —
  ~20k rows, instant to load, sufficient for FROC and slide-level AUC without
  touching the raster.

Resulting size: `5,625 × 13,125 × 2 B ≈ 148 MB`, ~40 MB compressed.

### Consequences
- 8× downsample sets the spatial precision of the heatmap at 2 µm. Against a
  clinical threshold of 200 µm (isolated tumor cells vs micrometastasis) this
  is two orders of magnitude of margin.
- Chunked writes mean the sliding window should traverse in **row-major
  tiles**, not raster-by-column, or every chunk gets rewritten repeatedly.
  Implemented as a chunk-aligned traversal order in `sliding_window.py`.
- `den` must be retained until the whole slide is finished; it is deleted after
  normalisation. Peak disk per slide is ~2× the final heatmap in float32
  (~1.2 GB) before compression.

### Revisit trigger
A downstream task needing sub-2 µm heatmap precision (e.g. single-cell scoring).

---

## ADR-010 — Splits assigned per patient, never per patch

**Status:** Accepted · **Date:** 2026-09-12

### Context
CAMELYON17 has 5 slides per patient; CAMELYON16 is 1 slide per case but slides
from the same centre share scanner and staining characteristics. Patch-level
random splits leak aggressively: neighbouring patches from the same slide are
near-duplicates, which inflates validation Dice by 15–25 points.

### Decision
- Split key is **patient_id** (falling back to slide_id where patients are
  unavailable), assigned by a seeded hash: `md5(patient_id) % 100`.
- `< 70` → train, `70–84` → val, `≥ 85` → test.
- Centre is recorded as a column and the centre distribution of each split is
  asserted to be within ±10 % of the cohort distribution (V4.4).
- The split column is baked into `patches.parquet`. Re-deriving splits at
  training time is forbidden — it makes runs irreproducible across code
  versions.

### Consequences
- Validation numbers are pessimistic relative to patch-split literature. That
  is the point.
- Adding slides changes nobody's split (hash is stable), so the index can grow
  incrementally.

### Amendment (2026-09-12, after BUG-020)
Patient derivation is **explicit per cohort**, never heuristic:
`splits.patient_from` is `slide_id` for CAMELYON16 (one slide per case) and a
`patient_\d+` regex for CAMELYON17. An unmatched rule raises. A prefix-based
guess mapped all 110 CAMELYON16 tumor slides to a single "patient" and all 160
normals to another, which would have emptied validation and left the test set
with no tumor. Gate V4.6 (`assert_splits_usable`) now blocks any index whose
splits are empty or tumor-free.

### Revisit trigger
Cohort change that introduces repeated patients across sources, or any new
cohort (which requires choosing `patient_from` deliberately).

---

## ADR-011 — Optimisation hyperparameters

**Status:** Accepted · **Date:** 2026-09-12

### Context
Baseline needs a defensible, non-arbitrary starting configuration on one 16 GB
GPU.

### Decision

| Hyperparameter | Value | Justification |
|---|---|---|
| Optimiser | AdamW, `wd=1e-4` | Standard for pretrained encoders; decoupled decay avoids over-penalising BN/bias (both excluded from decay) |
| LR | `3e-4` decoder, `3e-5` encoder | 10:1 discriminative LR is standard for a frozen-then-unfrozen ImageNet encoder |
| Schedule | Cosine to `1e-6`, 3-epoch linear warmup | Warmup matters because the balanced sampler makes early batches unrepresentative |
| Batch | 16 | Fits in 5.8/16 GB with headroom (Architecture §4) |
| Precision | bf16 AMP | bf16 over fp16: no loss-scaler, and Dice sums over 262k pixels are numerically safer |
| Grad clip | 1.0 (L2) | Dice gradients spike on near-empty masks |
| Epochs | 40, early stop patience 8 on val Dice | Pilot runs plateaued at ~28–34 |
| Seed | 1337, `torch.use_deterministic_algorithms(warn_only=True)` | Full determinism costs ~20 % throughput; warn-only is the compromise, and the seed + git SHA are recorded in the checkpoint |
| EMA | decay 0.999, evaluated alongside raw weights | Consistently +0.5–1.0 Dice at no training cost |

Batch-size/LR coupling: if `batch` changes, LR scales linearly
(`lr = 3e-4 × batch/16`). Encoded in `configs/base.yaml` as an expression so it
cannot drift.

### Consequences
Single documented baseline; every experiment is a diff against it. Any run that
deviates must record the deviation in its config, which is hashed into the run
directory name.

### Revisit trigger
Any change to model family or batch size beyond 2×.

---

## ADR-012 — Post-processing threshold and minimum lesion size in physical units

**Status:** Accepted · **Date:** 2026-09-12

### Context
Raw thresholded heatmaps contain speckle: isolated few-pixel activations on
stain artifacts, macrophages, and folds. These destroy FROC (each is a false
positive) while being clinically meaningless.

### Decision
Post-processing in `src/infer/postproc.py`, expressed in **µm², never pixels**:

```
1. binary = heat > τ                        # τ from checkpoint (ADR-006)
2. binary_fill_holes
3. remove connected components < 0.02 mm²   # ~141 µm across
4. binary_opening with a 10 µm disk
```

`0.02 mm²` sits well below the isolated-tumor-cell threshold (0.2 mm largest
dimension) so genuinely reportable findings are never removed, while speckle
below cellular-cluster scale is.

Pixel radii are computed at runtime from the heatmap's `.zattrs["mpp"]`.
Hard-coded pixel constants are a bug class of their own: they silently change
meaning whenever ADR-002 or ADR-009 changes.

### Consequences
- FROC improves substantially (pilot: 0.61 → 0.74 average sensitivity) without
  touching the model.
- Post-processing parameters are part of the reported result and are written
  into `artifacts/reports/{run_id}/config_snapshot.yaml`.

### Revisit trigger
Any clinical requirement to report isolated tumor cells.

---

## ADR-013 — Evaluation protocol

**Status:** Accepted · **Date:** 2026-09-12

### Context
Pixel Dice on patches is easy to report and easy to game — it is dominated by
large lesions and says little about whether a micrometastasis was found.

### Decision
Report three tiers, always together:

1. **Patch-level:** Dice, IoU on the held-out patch set. Development signal only.
2. **Slide-level detection (primary):** **FROC** — sensitivity at
   `{0.25, 0.5, 1, 2, 4, 8}` average false positives per slide, following the
   CAMELYON16 protocol. Lesion hits are computed on connected components after
   ADR-012 post-processing; a detection counts if its centroid falls inside an
   annotated lesion.
3. **Slide-level classification:** AUC using `max(heat)` over tissue as the
   slide score, plus a specificity check on normal slides.

Bootstrapped 95 % CIs (1000 resamples over slides) accompany every number. With
~50 test slides, a 3-point Dice difference is inside the noise; reporting a
point estimate alone invites over-reading.

### Consequences
- FROC requires lesion-level ground truth geometry, which the annotation parser
  must preserve — reinforcing the vector-geometry choice in contract A3.
- Evaluation is slide-bound and slow (~2.5 min/slide); it runs as a separate
  gated stage (V9), not inside the training loop.

### Revisit trigger
Extension to N-stage classification (CAMELYON17), which needs a patient-level
quadratic-weighted-kappa metric on top.

---

## ADR-014 — Configuration and experiment tracking

**Status:** Accepted · **Date:** 2026-09-12

### Decision
- YAML configs composed by inheritance (`base` ← `data` ← `model` ← overrides),
  materialised into a single frozen dict at startup.
- The **resolved** config is hashed (`sha1[:8]`); run directory is
  `artifacts/ckpt/{date}_{git_sha[:7]}_{config_hash}/`.
- The checkpoint embeds: resolved config, git SHA, dirty-tree flag, package
  versions (`pip freeze`), fitted threshold τ, seed, and the parquet index hash.
- A run started from a dirty working tree is tagged `-dirty` and is barred from
  producing a reported number (`eval/report.py` refuses unless `--allow-dirty`).

### Consequences
Any heatmap can be traced to the exact code, data index, and config that
produced it — which is what makes the invalidation procedure in `rollback.md`
R5 mechanically possible.

---

## ADR-015 — Gaussian window carries a floor; coverage means "written", not "well-weighted"

**Status:** Accepted · **Date:** 2026-09-12 · **Amends:** ADR-009

### Context
Found while implementing `infer/stitch.py` and testing against a fixture with
deliberately non-stride-aligned dimensions. Two defects in the ADR-009 design
as originally written:

1. With `σ = S/8`, the raw Gaussian at a patch corner is ~1.4e-7. Nothing can
   extend past the slide edge, so the outermost ~40 heatmap pixels are covered
   by exactly one patch at that weight. The original coverage assert
   (`den < 0.05` → error) fired on **15.4 %** of a perfectly healthy slide.
2. The heatmap is sized `ceil(H0/d)`, but a patch's rounded destination tops
   out at `floor((H0 − extent)/d) + tile`. The final row and column were never
   written — `den == 0`, reading as a confident negative. Measured: 1501
   pixels on the test fixture, exactly `876 + 626 − 1`.

### Decision
- Window is `w = exp(...) + 1e-3`. Measured effect: dynamic range 7e6:1 →
  1000:1; interior `den` minimum 0.0735 → 0.0775; seam suppression unchanged
  (border still vetoed 1000-fold).
- Coverage assert triggers on `den < 1e-6` — "never written" — not on low
  weight. Low-weight fraction is logged, not raised.
- Patches flush against the slide edge are anchored to the heatmap's far edge,
  shifting them by at most 1 heatmap pixel (2 µm) and only at the border.

### Consequences
Gate V8.4 now reports `0.00 %` uncovered on a 5003 × 7001 slide with stride
256, and V8.3 reconstructs a constant 0.37 field to within 1.2e-7.

### Revisit trigger
Any change to `sigma_divisor` or `stride`; the interior `den` minimum should
be re-measured, since a sparser stride lowers it toward the hole threshold.

---

## ADR-016 — Artifact rejection runs before Otsu, and thresholds have a physical floor

**Status:** Accepted · **Date:** 2026-09-12 · **Amends:** ADR-003

### Context
Two ordering/robustness defects found by unit tests on synthetic slides.

1. **Ordering.** ADR-003 computed the Otsu thresholds first and subtracted the
   pen/artifact mask afterwards. A black border sits at saturation 0, which
   manufactures a spurious bimodality: Otsu then splits black-vs-glass instead
   of glass-vs-tissue and returns ~0, so every glass pixel with colour noise
   becomes tissue. Subtracting the artifact afterwards removes the black
   region but leaves the poisoned threshold. Measured tissue fraction on a
   blank slide with a black border: **0.67**.
2. **The bimodality guard is not sufficient on its own.** Otsu can split a
   pure-noise channel at 0, putting 5 % of pixels below and 95 % above, which
   scores as strongly bimodal (ratio 0.30, well over the 0.08 threshold) while
   meaning nothing. Blank slide measured at **0.95** tissue.

### Decision
- Compute the pen/artifact mask **first**; threshold only over valid pixels.
  Raise if under 1 % of the thumbnail is non-artifact.
- The former `fallback_*` constants become **floors**: `t = max(otsu, floor)`.
  H&E tissue has saturation above ~20 and inverted-grey above ~40; a threshold
  below that is physically implausible whatever the histogram says. The
  bimodality ratio is retained, but now only sets the QC flag.

### Consequences
Blank-with-border and blank-with-ink slides both resolve to a tissue fraction
of 0.00 and are correctly quarantined by the contract-A2 lower bound. Real
tissue at 0.25 coverage is still detected at 0.25.

### Revisit trigger
A stain protocol whose tissue genuinely falls below the saturation floor
(e.g. IHC with a very pale counterstain) — the floors are H&E-specific.

---

## Open Questions

| # | Question | Blocking? | Owner |
|---|---|---|---|
| Q1 | Does 256 µm FOV under-detect large lesion interiors (uniform texture)? Multi-scale ablation needed. | No | — |
| Q2 | Is D4 TTA worth 4× inference cost, or does it mostly smooth? Measure Δ FROC. | No | — |
| Q3 | Should normal slides contribute to the epoch proportionally to their tissue area, or uniformly per slide? Current: per slide. | No | — |
| Q4 | Threshold τ fitted on val may not transfer across centres. Per-centre τ, or a calibrated probability? | For cross-centre claims | — |
| Q5 | Interior `den` minimum is 0.078 at stride S/2. At stride 3S/4 it would fall near the old 0.05 threshold. Should σ scale with stride rather than with patch size? | No | — |
