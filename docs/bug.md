# Debugging Journal — `wsi-metastasis-seg`

**How this file is used.** Every non-trivial defect gets an entry before it is
fixed, so the hypothesis is recorded *before* the outcome is known. Entries are
append-only; a wrong hypothesis stays in the file with `✗` next to it, because
the list of things that turned out not to be the cause is the most useful part
of a debugging log.

**Honesty convention.** Entries are tagged by provenance:

- `[OBSERVED]` — actually hit in this repo; has a commit SHA and a regression test.
- `[PRE-SEEDED]` — a known WSI-pipeline failure mode that has **not** occurred
  here, entered in advance because a guard or invariant was written to prevent
  it. These carry a `Guard:` line pointing at the code and the verification
  gate instead of a fix SHA.

Pre-seeded entries exist because most of these defects are silent — they
degrade a metric rather than raise — and the cost of discovering them at
evaluation time is a full retrain.

---

## Entry Template

```markdown
## BUG-nnn — <one-line symptom>
Status:  Open | Investigating | Fixed | Won't fix | Guard-in-place
Severity: S1 wrong-results-silently | S2 crash | S3 perf | S4 cosmetic
Provenance: [OBSERVED commit abc1234] | [PRE-SEEDED]
Surfaced in: <stage / module / gate>

### Symptom
What was seen, with the exact command and the exact output.

### Hypotheses
H1 ...  ✗ ruled out by <test>
H2 ...  ✓ confirmed by <test>

### Root cause
### Fix
### Validation
Command + expected output proving it's fixed.
### Regression guard
tests/... or verification_checklist gate V-n.n
```

---

## BUG-001 — Patches from `.svs` have black borders; tissue mask includes them

**Status:** Fixed · **Severity:** S1 · **Provenance:** `[OBSERVED]`
**Surfaced in:** stage 01, `verification_checklist` V2.2 (tissue fraction 0.94, expected <0.90)

### Symptom
```
$ python scripts/01_build_tissue_masks.py --slides tumor_001
TissueDetectionError: tissue fraction 0.9412 exceeds upper bound 0.90
```
Dumping the thumbnail showed a large black L-shaped region along the right and
bottom edges of the slide, classified as tissue by the inverted-grayscale Otsu
branch.

### Hypotheses
- **H1** Scanner wrote a genuinely dark region (coverslip shadow). ✗ — the
  region is exactly rectangular and exactly aligned to `level_dims[lt]`,
  which physical shadows are not.
- **H2** Otsu threshold collapsed. ✗ — inter-class variance ratio was 0.31,
  healthy.
- **H3** `read_region` returns RGBA, and converting via `arr[..., :3]` leaves
  pixels where `A = 0` as `(0,0,0)`. ✓ — confirmed by
  `np.unique(arr[...,3])` returning `{0, 255}` and the zero-alpha mask being
  pixel-identical to the black region.

### Root cause
OpenSlide/tiffslide return **RGBA**, and regions outside the scanned area have
`A = 0` with undefined RGB (usually zero). Slicing off the alpha channel turns
"nothing here" into "pure black", which the inverted-grayscale branch reads as
maximally dense tissue.

### Fix
`src/io/slide.py::_rgba_to_rgb_on_white` — composite onto **white**, which is
what glass looks like:
```python
a = rgba[..., 3:4].astype(np.float32) / 255.0
rgb = (rgba[..., :3] * a + 255.0 * (1.0 - a)).round().astype(np.uint8)
```
Applied inside `SlideReader.read_region` so no caller can forget it.

### Validation
```
$ python -m scripts.01_build_tissue_masks --slides tumor_001
tumor_001  tissue_frac=0.2137  OK
```

### Regression guard
`tests/unit/test_slide_rgba.py::test_zero_alpha_becomes_white` — synthetic RGBA
tile with an alpha=0 quadrant; asserts the composited output is `255` there.

---

## BUG-002 — DataLoader with `num_workers>0` returns corrupted or duplicated patches

**Status:** Fixed · **Severity:** S1 · **Provenance:** `[OBSERVED]`
**Surfaced in:** stage 03, first multi-worker run

### Symptom
Training loss looked normal, but a visual dump of one batch showed 3 of 16
patches were visually identical despite distinct coordinates, and 2 contained
torn image data (top half from one location, bottom half from another). With
`num_workers=0` the same indices produced correct patches.

### Hypotheses
- **H1** Bug in the balanced sampler emitting duplicate indices. ✗ —
  `len(set(indices)) == len(indices)` over a full epoch.
- **H2** Coordinate jitter RNG shared across workers giving identical offsets.
  ✗ — offsets differed; and it would not explain torn images.
- **H3** A single `SlideReader` handle created in the parent process and
  inherited across `fork()`, so N workers share one file descriptor and one
  internal seek/cache state. ✓ — reproduced deterministically with
  `num_workers=2` and two concurrent reads from the same slide; disappeared
  with `multiprocessing_context="spawn"`.

### Root cause
`libtiff`/`openslide` handles carry mutable state and are not fork-safe. The
`PatchDataset.__init__` was pre-opening every slide into `self.readers` for
speed. After `fork`, all workers share those descriptors and interleave seeks.

`spawn` hid the symptom (each worker re-imports and re-opens) but at a large
startup cost and while silently pickling handles — not a fix.

### Fix
Handles are opened **lazily, per process**, and the object refuses to be
pickled with a live handle:

```python
class PatchDataset(Dataset):
    def __init__(...):
        self._readers: dict[tuple[int, str], SlideReader] = {}   # (pid, slide_id)

    def _reader(self, slide_id):
        key = (os.getpid(), slide_id)
        if key not in self._readers:
            self._readers[key] = open_slide(self.paths[slide_id])
        return self._readers[key]

    def __getstate__(self):
        s = self.__dict__.copy(); s["_readers"] = {}; return s

def worker_init(worker_id):
    info = torch.utils.data.get_worker_info()
    info.dataset._readers = {}
    np.random.seed(BASE_SEED + worker_id)
```

An LRU cap of 64 open handles per worker prevents fd exhaustion on large cohorts.

### Validation
```
$ pytest tests/integration/test_loader_consistency.py -q
# reads the same 256 indices with num_workers=0 and num_workers=8
# asserts byte-identical tensors
256 patches matched (max |Δ| = 0)
```

### Regression guard
`tests/integration/test_loader_consistency.py`; gate V5.2.

---

## BUG-003 — Masks misaligned with images by a growing offset toward the slide edge

**Status:** Fixed · **Severity:** S1 · **Provenance:** `[OBSERVED]`
**Surfaced in:** stage 02/03, gate V3.2 overlay QC

### Symptom
Overlaying the rasterised tumor mask on its patch looked correct near the
top-left of the slide and progressively wrong toward the bottom-right — by
~180 px at the far corner. Training still converged, to a mediocre Dice of 0.58.

### Hypotheses
- **H1** Annotation XML uses a different origin convention. ✗ — the offset is
  proportional to position, not constant, so it is a scale error not a shift.
- **H2** Mask rasterised at level `lw` but polygons in level-0 coordinates
  without scaling. ✗ — that would give a ~2× error, not 0.4 %.
- **H3** `level_downsamples[1]` was assumed to be exactly `2.0`, but the file
  reports `1.9992175...`. Over 105,000 px that accumulates to ~205 px. ✓ —
  confirmed by printing `reader.level_downsamples` and recomputing.

### Root cause
Hard-coded `scale = 2 ** level` in `geometry.l0_to_level`. Scanners produce
pyramid levels whose dimensions are `ceil(w / 2)` per level, so the effective
downsample is *not* a clean power of two and the discrepancy compounds.

### Fix
`src/utils/geometry.py` now takes the factor from the file, never from the
level index:
```python
def l0_to_level(xy, downsamples, level):
    s = downsamples[level]          # float, from the slide
    return xy[0] / s, xy[1] / s
```
and `read_region`'s contract is asserted at the boundary:
```python
assert_level0_coords(x0, y0)   # documented + checked, see flow.md §1
```

### Validation
```
$ python scripts/qc_overlay.py --slide tumor_009 --corners
top-left     IoU(mask, annotation-rendered-at-L0) = 0.998
bottom-right IoU = 0.997        # was 0.61
```

### Regression guard
`tests/unit/test_geometry.py::test_non_power_of_two_downsample` uses a
synthetic pyramid with `downsamples = [1.0, 1.9992, 3.9981]`; gate V3.2.

---

## BUG-004 — Otsu classifies noise as tissue on near-empty slides

**Status:** Guard-in-place · **Severity:** S1 · **Provenance:** `[PRE-SEEDED]`

### Symptom (expected form)
A slide containing a very small tissue fragment has a near-unimodal intensity
histogram. Otsu still returns a threshold — an arbitrary one splitting the
glass distribution in half — so ~50 % of blank glass is indexed as tissue and
the patch index fills with white patches labelled negative.

### Why it is silent
Nothing crashes. Training simply gets a large, easy negative class, the loss
drops, and validation Dice on real tissue quietly degrades.

### Guard
`src/preprocess/tissue.py::_bimodality_guard` — before accepting an Otsu
threshold, require
```
inter_class_variance / total_variance > 0.08
```
On failure, fall back to a conservative fixed threshold and set
`slides.parquet.tissue_qc_flag = "unimodal_fallback"`. Contract A2's lower
bound (`tissue.mean() ≥ 0.01`) catches the complementary case.

### Verification gate
V2.3 — synthetic all-glass thumbnail must produce `tissue.mean() < 0.01` and
raise `TissueDetectionError`, not a half-full mask.

---

## BUG-005 — Pen marks and slide labels indexed as tissue

**Status:** Guard-in-place · **Severity:** S2 (wasted compute) / S1 (FP source)
**Provenance:** `[PRE-SEEDED]`

### Symptom (expected form)
Pathologist ink annotations are highly saturated and pass the HSV-saturation
Otsu branch. Patches of green/blue marker enter the index. At inference they
become high-confidence false positives, which is expensive under FROC (every
FP moves the operating point).

### Guard
`_pen_and_artifact_mask` (ADR-003): reject `S > 0.35` outside the H&E hue band,
and `V < 0.20`. The rejected fraction is logged per slide; a slide with
`pen_frac > 0.05` is flagged for manual QC rather than silently cleaned.

### Verification gate
V2.4 — on the 3 CAMELYON16 slides with known marker ink, indexed patches
intersecting the ink region must be < 2 % of that region's area.

---

## BUG-006 — Pure Dice loss oscillates and stalls at zero prediction

**Status:** Fixed (design) · **Severity:** S1 · **Provenance:** `[OBSERVED]`
**Surfaced in:** stage 03 pilot, epochs 1–6

### Symptom
With `loss = SoftDice` alone, training loss oscillated between ~0.97 and ~1.00
and never descended. Predictions were uniformly ≈0. Validation Dice 0.000.

### Hypotheses
- **H1** LR too high. ✗ — reducing to 3e-5 changed the oscillation amplitude,
  not the behaviour.
- **H2** Encoder frozen and decoder unable to learn. ✗ — same with everything
  unfrozen.
- **H3** Dice's gradient is degenerate on all-negative patches: with
  `y = 0` everywhere, `dice = 2Σpy / (Σp + Σy)` → `0/Σp`, and the gradient
  pushes `p → 0` with magnitude independent of how wrong `p` is elsewhere.
  Combined with the sampler's negative-heavy composition early on, the model
  collapses to the trivial solution. ✓ — confirmed by restricting a pilot to
  tumor-only patches, where pure Dice trained fine.

### Root cause
Dice is a set-overlap measure and is undefined/degenerate when the ground-truth
set is empty. Empty masks are ~75 % of batches under the 1:3 sampler.

### Fix
- `loss = 0.5·BCEWithLogits(pos_weight=2.0) + 0.5·SoftDice` (ADR-006). BCE
  supplies a well-behaved per-pixel gradient on empty masks.
- Smoothed Dice: `(2Σpy + 1) / (Σp + Σy + 1)` so an empty prediction on an
  empty mask scores 1.0 rather than NaN.
- Dice computed **per image, then averaged**, not over the flattened batch —
  batch-flattened Dice lets one large lesion dominate 15 empty patches.

### Validation
```
$ python scripts/03_train.py --set train.max_steps=500
step 500  loss=0.412  train_dice=0.514   # was 0.98 / 0.000
```

### Regression guard
`tests/unit/test_losses.py::test_dice_empty_pair_is_one`,
`::test_bcedice_gradient_nonzero_on_empty_mask`.

---

## BUG-007 — Validation Dice of 0.93 that collapses to 0.61 on the test set

**Status:** Fixed · **Severity:** S1 · **Provenance:** `[OBSERVED]`
**Surfaced in:** stage 05, first end-to-end evaluation

### Symptom
Validation Dice climbed to 0.93 within 6 epochs — implausibly high and
implausibly fast for this task. Test Dice was 0.61.

### Hypotheses
- **H1** Test slides are harder (different centre). ✗ — centre distribution
  was balanced across splits.
- **H2** Overfitting. Partially true but cannot explain a 32-point gap at
  epoch 6 with a 6 M-parameter model.
- **H3** Split leakage: the split was assigned per **patch** rather than per
  **slide/patient**, so adjacent overlapping-context patches from the same
  slide appeared in both train and val. ✓ — confirmed:
  `set(train.slide_id) & set(val.slide_id)` had 312 elements.

### Root cause
Splits were being derived at training time with
`train_test_split(patches, ...)` rather than read from the index column
produced in stage 01.

### Fix
- Splits assigned once in stage 01 by seeded hash of `patient_id` (ADR-010) and
  persisted into the index.
- Training-time split derivation deleted. `data.index.load` raises if the
  `split` column is missing.
- Startup assertion in `scripts/03_train.py`:
```python
assert not (set(tr.slide_id) & set(va.slide_id)), "slide-level split leakage"
```

### Validation
```
$ python scripts/03_train.py --set train.max_epochs=6
epoch 6  val_dice=0.681   # plausible; gap to test now 2.1 points
```

### Regression guard
`tests/unit/test_splits.py::test_no_slide_in_two_splits`; gate V4.2.

---

## BUG-008 — Tumor masks include regions annotated as non-tumor exclusions

**Status:** Fixed · **Severity:** S1 · **Provenance:** `[OBSERVED]`
**Surfaced in:** stage 02, QC overlay on `tumor_026`

### Symptom
An annotated region containing an obvious fat/stroma island inside a metastasis
was fully labelled tumor. The model learned to predict tumor on adipose tissue.

### Root cause
CAMELYON16 XML uses annotation **groups**: `_0` and `_1` are tumor regions,
`_2` marks areas inside them that are *not* tumor. Parsing all `<Annotation>`
elements uniformly unions the exclusions into the positive class.

### Fix
`src/io/annotations.py`:
```python
tumor = unary_union([p for p, g in polys if g in ("_0", "_1")])
excl  = unary_union([p for p, g in polys if g == "_2"])
geom  = tumor.difference(excl).buffer(0)      # buffer(0) repairs self-intersections
```
`buffer(0)` is required because several CAMELYON polygons are self-intersecting
and `difference` on an invalid geometry raises `TopologicalError`.

### Validation
```
$ python scripts/qc_annotations.py --slide tumor_026
groups: _0=3  _1=0  _2=2
tumor area before exclusion: 4.81 mm²
tumor area after  exclusion: 4.12 mm²   (-14.3%)
geometry valid: True
```

### Regression guard
`tests/unit/test_annotations.py::test_group_2_is_subtracted`; gate V3.3 asserts
total tumor area per slide matches a stored reference within 0.5 %.

---

## BUG-009 — Heatmap saturates at 1.0 across large tissue regions

**Status:** Guard-in-place · **Severity:** S1 · **Provenance:** `[PRE-SEEDED]`

### Symptom (expected form)
Accumulating `num` and `den` in `float16` to save disk. `float16` has ~3
decimal digits and a max of 65504, but the real problem is precision: adding
`w ≈ 0.0003` to a running total near 4.0 is a no-op in float16
(`4.0 + 0.0003 == 4.0`). Border contributions vanish, `den` under-counts, and
`num/den` inflates toward 1.0 in overlap regions.

### Guard
Accumulators are `float32` (ADR-009); only the final quotient is cast to
`float16`. Asserted at construction:
```python
assert num.dtype == np.float32 and den.dtype == np.float32
```
plus a range check on the result: `0.0 ≤ heat ≤ 1.0 + 1e-3`.

### Verification gate
V8.3 — stitch a synthetic slide where every patch predicts a constant 0.37; the
reconstructed heatmap must be `0.37 ± 1e-3` everywhere inside coverage,
including at patch seams.

---

## BUG-010 — Visible grid seams in the reconstructed heatmap

**Status:** Fixed (design) · **Severity:** S2 · **Provenance:** `[OBSERVED]`
**Surfaced in:** stage 04, first overlay render

### Symptom
The overlay showed a regular 512-px checkerboard: lesion boundaries were
discontinuous exactly at patch borders, and thin bright lines ran along the
grid.

### Hypotheses
- **H1** Off-by-one in the stitch destination slice. ✗ — coordinates were
  exact; the artifact is a value discontinuity, not a spatial shift.
- **H2** Normalisation statistics differ per patch. ✗ — fixed ImageNet stats.
- **H3** Border pixels of each patch have a truncated receptive field, so the
  model is genuinely least accurate there; non-overlapping tiling makes those
  worst predictions the *only* prediction for those pixels. ✓ — confirmed by
  plotting mean absolute error vs distance from patch border: error at the
  border was 2.4× the patch-centre error.

### Root cause
Not a coding bug — an inference-design bug. Stride == patch size.

### Fix
Stride 256 (50 % overlap) plus Gaussian weighting with `σ = S/8 = 64`
(ADR-004, Architecture §6), so border predictions are weighted ~0.0003 wherever
a neighbouring patch's centre covers the same pixel.

### Validation
```
$ python scripts/qc_seams.py --slide test_012
seam contrast metric (|Δ| across grid lines / |Δ| elsewhere):
  stride 512, no window : 3.81
  stride 256, gaussian  : 1.02      # 1.00 == indistinguishable from background
```

### Regression guard
`tests/integration/test_stitch_seams.py` — a synthetic model returning a smooth
radial gradient must reconstruct with max seam-line gradient < 1.05× the
interior gradient.

---

## BUG-011 — Stain augmentation applied to the mask, destroying labels

**Status:** Fixed · **Severity:** S1 · **Provenance:** `[OBSERVED]`
**Surfaced in:** stage 03, after adding HED jitter

### Symptom
Adding HED colour jitter dropped validation Dice from 0.68 to 0.31 in one
epoch. Masks dumped from the DataLoader contained non-binary values in
`[0, 1.07]`.

### Root cause
The augmentation was appended to a single Albumentations `Compose` applied to
`{"image": img, "mask": mask}`. Albumentations routes unknown custom transforms
to *all* targets unless they declare `targets = {"image": ...}`. The custom
`HEDJitter` did not declare its targets, so it colour-transformed the mask.

### Fix
Split into two pipelines with an explicit contract in
`src/data/transforms.py`:
```python
geometric = A.Compose([...flips, rot90...])           # image + mask
photometric = A.Compose([HEDJitter(), Blur(), ...])   # image only
```
and a post-condition in `__getitem__`:
```python
assert set(np.unique(mask)) <= {0, 1}, "mask is no longer binary"
```

### Validation
```
$ python scripts/dump_batch.py --n 64
mask unique values: [0. 1.]      image mean/std: 0.412 / 0.271
```

### Regression guard
`tests/unit/test_transforms.py::test_mask_binary_after_full_pipeline`; gate V5.3.

---

## BUG-012 — Inference produces a heatmap with a blank rectangular hole

**Status:** Guard-in-place · **Severity:** S1 · **Provenance:** `[PRE-SEEDED]`

### Symptom (expected form)
A batch is dropped (OOM retry, an exception swallowed in the prefetch thread,
or a planner off-by-one at the right/bottom margin where `x0 + 512 > width`).
`den` stays 0 in that region, `num/den` → 0, and the region reads as a
confident negative. Under FROC this is indistinguishable from a correct
negative, so the metric never reveals it.

### Guard
`GaussianStitcher.finalize` asserts coverage before writing:
```python
uncovered = (den < 0.05) & tissue_at_heatmap_res
if uncovered.mean() > 1e-4:
    raise CoverageError(f"{uncovered.mean():.2%} of tissue uncovered")
```
Margin positions are clamped (`x0 = min(x0, W - 512)`) rather than skipped, so
the right/bottom edges are always covered — with slightly more overlap, which
the `num/den` normalisation handles for free.

### Verification gate
V8.4 — inference on a slide whose width is not a multiple of the stride must
report `uncovered = 0.00%`.

---

## BUG-013 — `cv2.resize` on the mask introduces interpolated label values

**Status:** Guard-in-place · **Severity:** S1 · **Provenance:** `[PRE-SEEDED]`

### Symptom (expected form)
When ADR-002's residual scale correction fires (~7 % of slides), both image and
mask are resampled. Using `INTER_AREA` or the default `INTER_LINEAR` on the
mask produces fractional labels at every lesion boundary, which softens the
supervision along exactly the pixels the metric cares about.

### Guard
Image and mask are resized by different code paths, and the mask path is
pinned:
```python
img  = cv2.resize(img,  dsize, interpolation=cv2.INTER_AREA)
mask = cv2.resize(mask, dsize, interpolation=cv2.INTER_NEAREST)
```
Followed by the same binary post-condition as BUG-011.

### Verification gate
V5.3 (shared with BUG-011) covers every path out of `__getitem__`.

---

## BUG-014 — Throughput collapses after ~2 hours of training

**Status:** Guard-in-place · **Severity:** S3 · **Provenance:** `[PRE-SEEDED]`

### Symptom (expected form)
Steps/s degrades steadily. Cause candidates, in the order they should be
checked:
1. **fd exhaustion** — each worker opens a handle per slide and never closes
   them; at 400 slides × 8 workers the process hits `ulimit -n`.
2. **Page-cache thrash** — random access across 400 multi-GB files evicts
   everything; `iostat` shows read throughput rising while steps/s falls.
3. **Zarr/tifffile internal caches** growing unbounded.

### Guard
- LRU cap of 64 open `SlideReader`s per worker (BUG-002 fix).
- A throughput watchdog in `train/loop.py`: if the 100-step moving average
  drops below 60 % of the first-500-step baseline, log a warning with
  `len(self._readers)`, RSS, and open-fd count. Diagnose, don't guess.
- `persistent_workers=True` so worker startup cost is paid once.

### Verification gate
V7.4 — a 2000-step run must show final-500-step throughput within 15 % of the
first-500-step throughput.

---

## Watchlist (no entry opened yet)

| ID | Concern | Detection idea |
|---|---|---|
| W1 | Slides with MPP metadata present but *wrong* (scanner misconfiguration) — undetectable by the ADR-002 check | Cross-check nuclei diameter distribution against a reference; flag outliers |
| W2 | `bf16` autocast reducing Dice-sum precision over 262k pixels | Compute the loss reduction in fp32 explicitly; A/B the final metric |
| W3 | EMA weights and raw weights fitting different optimal τ | Fit and store τ separately for each; currently one τ is stored |
| W4 | Zarr write contention if slide inference is parallelised across processes | One zarr store per slide, so currently safe; revisit if a shared store is introduced |
| W6 | `thumbnail_level` picks the nearest level with no tolerance; a shallow pyramid silently changes the effective morphology scale (BUG-019) | Warn when the chosen level is >2x from the 8 um/px target |
| W5 | Annotation polygons in a coordinate space other than level 0 for non-CAMELYON cohorts | Assert annotation bbox ⊆ level-0 dims at parse time |

---

## BUG-015 — Coverage assert fires on 15% of a healthy slide

**Status:** Fixed · **Severity:** S2 (false alarm) · **Provenance:** `[OBSERVED]`
**Surfaced in:** first run of gate V8.4 on `synth_edge`

### Symptom
```
CoverageError: 15.39% of tissue uncovered by the sliding window
(threshold 0.01%)
```
on a slide where every grid position had in fact been processed.

### Hypotheses
- **H1** The planner drops margin positions. ✗ — `summary["kept"] == grid`, and
  the far-edge positions were present in the coordinate list.
- **H2** The stitcher writes to the wrong destination. ✗ — a constant-field
  reconstruction was exact wherever `den` was non-trivial.
- **H3** The threshold conflates "never written" with "written at low
  weight". ✓ — measured `den` over a full grid: interior minimum 0.0735,
  border minimum 1.44e-07. The assert was set at 0.05, so the entire outer
  ~40-pixel frame tripped it.

### Root cause
Design error, not a coding error. At `σ = S/8` the window is ~1.4e-7 at a
patch corner, and nothing can extend past the slide edge, so border pixels are
legitimately covered by a single very small weight. `num/den` is still exactly
right there; only the guard was wrong.

### Fix
Two changes (ADR-015): a window floor of 1e-3, and a hole threshold of 1e-6
with low-weight area reported rather than raised.

### Validation
```
uncovered tissue: 0.0000%
recon 0.370000..0.370000   max|err| 1.19e-07
den range 0.050 .. 1.038
```

### Regression guard
`tests/integration/test_stitch_constant.py::test_no_uncovered_tissue_on_non_aligned_dims`

---

## BUG-016 — Final heatmap row and column never written

**Status:** Fixed · **Severity:** S1 · **Provenance:** `[OBSERVED]`
**Surfaced in:** gate V8.4, after BUG-015 was fixed

### Symptom
After the window floor fix, uncovered tissue dropped from 15.39% to 0.27% —
but not to zero. Locating the residual:
```
shape (876, 626)   holes 1501
hole rows: [0 1 2 3 4] ... [871 872 873 874 875]
hole cols: [0 1 2 3 4] ... [621 622 623 624 625]
```
`1501 == 876 + 626 - 1`: exactly the last row plus the last column.

### Root cause
The heatmap is sized `ceil(H0 / d)` so its final row represents a partial
level-0 row. A patch's extent is `512 * 1.9996 / 8 = 127.97` heatmap pixels,
not 128, and its destination origin is rounded down. The last patch therefore
reaches heat column 624 while the array has 626 columns.

The `min(hx, W - t)` clamp did **not** help: `hx = 497` and `W - t = 498`, so
the clamp was a no-op precisely where it was needed.

### Fix
Anchor edge-flush patches to the far edge explicitly, rather than relying on a
clamp that only fires on overshoot:
```python
if x0_l0 + self.extent_l0 >= W0 - self.level_downsample:
    hx = W - t
```
Shifts an edge patch by at most 1 heatmap pixel (2 µm at the defaults), and
only for patches already against the border.

### Validation
`uncovered tissue: 0.0000%` on 5003 × 7001 with stride 256.

### Regression guard
Same test as BUG-015; the fixture's dimensions are non-aligned on purpose.

---

## BUG-017 — Blank slide with a black border indexes at 0.67 tissue

**Status:** Fixed · **Severity:** S1 · **Provenance:** `[OBSERVED]`
**Surfaced in:** `tests/unit/test_tissue.py::test_black_border_does_not_become_tissue`

### Symptom
```
assert 0.6699375 < 0.1
sat_threshold=0  gray_threshold=19  pen_frac=0.3  qc_flag='high_pen'
```
A thumbnail of pure glass with a black strip reported two thirds tissue.

### Hypotheses
- **H1** The pen/artifact mask fails to catch the black region. ✗ —
  `pen_frac = 0.30` exactly matches the strip, so it was caught.
- **H2** Morphological closing dilates the surviving noise. ✗ — the fraction
  was already ~0.67 before morphology.
- **H3** The artifact is subtracted *after* thresholding, so it still
  participates in the Otsu histogram. ✓ — `sat_threshold == 0` is the tell:
  black sits at saturation 0 and manufactures a bimodal split of
  black-vs-glass, so Otsu separates the artifact instead of the tissue.

### Root cause
Ordering. ADR-003 as written computed thresholds on the full thumbnail and
masked artifacts afterwards, which removes the artifact pixels but keeps the
threshold they poisoned.

### Fix
Compute the artifact mask first and threshold only over valid pixels
(ADR-016).

### Validation
Blank-with-border now resolves to `tissue_frac = 0.00` and is correctly
quarantined by the contract-A2 lower bound.

### Regression guard
`tests/unit/test_tissue.py::test_black_border_does_not_become_tissue`

---

## BUG-018 — Bimodality guard passes on a degenerate split of pure noise

**Status:** Fixed · **Severity:** S1 · **Provenance:** `[OBSERVED]`
**Surfaced in:** `tests/unit/test_tissue.py`, immediately after BUG-017's fix

### Symptom
With artifacts correctly excluded, an all-glass thumbnail still reported
`tissue_frac = 0.95` — and now tripped the contract-A2 *upper* bound instead
of the lower one.

### Root cause
The BUG-004 guard checks inter-class variance ratio > 0.08. On a pure-noise
saturation channel (values 0–15), Otsu splits at 0: roughly 5 % of pixels
below, 95 % above, mean separation ~5 against a variance of ~4. The ratio
computes to ~0.30 — comfortably "bimodal" — while describing nothing but
sensor noise. The guard was necessary but not sufficient.

### Fix
Treat the fallback constants as **floors** rather than fallbacks:
`t = max(otsu, floor)`, with `floor_sat = 20` and `floor_gray = 40`. H&E
tissue has saturation above ~20 and inverted-grey above ~40; a threshold below
that is physically implausible regardless of what the histogram reports. The
bimodality ratio is kept, but now only sets the QC flag.

Note the general lesson: a statistical guard answers "is this split
well-separated?", which is not the question. The question is "is this split
physically plausible?", and only a domain constant answers it.

### Validation
```
all-glass thumbnail -> TissueDetectionError (fraction 0.0000 below lower bound)
real tissue at 0.25 coverage -> tissue_frac 0.25, detected
```

### Regression guard
`tests/unit/test_tissue.py` (4 cases: all-glass, black border, ink, real tissue)

---

## BUG-019 — Physical morphology radii behave differently on shallow pyramids

**Status:** Fixed (fixtures) · **Severity:** S3 · **Provenance:** `[OBSERVED]`
**Surfaced in:** stage 01 on the synthetic fixtures

### Symptom
`synth_normal`, generated with 0.30 tissue coverage, reported
`tissue_frac = 0.8266`. `synth_tumor` reported 0.9028 and was quarantined by
the contract-A2 upper bound.

### Root cause
Not a defect in `tissue.py`. The fixtures had only 3 pyramid levels, so
`thumbnail_level(8.0)` resolved to 1.0 µm/px instead of ~8 µm/px. The closing
radius is specified as 40 µm (ADR-003), which is 5 px at 8 µm/px but **40 px**
at 1.0 µm/px — a structuring element large enough to merge every tissue blob.

This is the physical-units design working correctly: the algorithm did the
same thing in microns at both resolutions. The fixture supplied conditions a
real slide never would.

### Fix
Fixtures now build 6 pyramid levels (real WSIs have 6–9), so the thumbnail
resolves near 8 µm/px.

### Validation
```
synth_normal  tissue_frac=0.1618  qc=ok
synth_tumor   tissue_frac=0.2065  qc=ok
```

### Note for the watchlist
A real cohort with an unusually shallow pyramid would hit the same behaviour.
`thumbnail_level` currently picks the nearest level with **no tolerance**;
consider warning when the chosen level is more than 2x from the 8 µm/px target.

---

## BUG-020 — Patient rule collapsed 270 CAMELYON16 slides into two "patients"

**Status:** Fixed · **Severity:** S1 · **Provenance:** `[OBSERVED]`
**Surfaced in:** review of stage 01 before the first real data run

### Symptom
None yet — and that is the point. The fixtures all passed. The defect would
have surfaced only after a 14-hour training run, as a meaningless evaluation.

### Root cause
Stage 01 derived `patient_id` by stripping the last underscore segment of the
filename, a heuristic that fits CAMELYON17 (`patient_017_node_2`) and is
catastrophic on CAMELYON16:

```
tumor_001, tumor_002, ... tumor_110   ->  patient_id "tumor"   -> all one split
normal_001, ... normal_160            ->  patient_id "normal"  -> all one split
```

Every tumor slide would have landed in `train`, every normal slide in `test`.
Validation would have been empty. The test set would have contained no tumor
at all, making FROC and slide-level AUC undefined.

The three synthetic fixtures hid it completely: `synth_tumor`, `synth_normal`
and `synth_edge` all reduce to patient `synth`, so the collapse looked like
normal behaviour for a three-slide cohort.

### Fix
1. Patient derivation is now explicit per cohort, configured as
   `splits.patient_from`. CAMELYON16 uses `slide_id` (one slide per case);
   CAMELYON17 uses a regex capturing `patient_\d+`. An unmatched rule raises
   rather than falling back.
2. New blocking gate `assert_splits_usable` (V4.6) in stage 02: every split
   must be non-empty, train and val must each contain tumor patches, and test
   must contain tumor patches. An `--allow-single-split` escape hatch exists
   for synthetic fixtures only.

### Validation
```
CAMELYON16 with patient_from=slide_id:
  tumor    train  82   val 14   test 14
  normal   train 116   val 19   test 25

CAMELYON17 with patient_from=camelyon17:
  patient_017_node_{0,2,4} -> patient_017 -> all in the same split
```

### Regression guard
`tests/unit/test_patient_rules.py` (7 cases, including a direct reconstruction
of the original failure).

### Lesson
A heuristic that reads plausibly on one dataset can be silently destructive on
another. The deeper error was not the regex — it was having no check that the
resulting splits were usable. The gate matters more than the fix, because the
gate catches the next variant of this too.
