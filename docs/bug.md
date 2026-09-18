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

---

## BUG-021 — Half the download was label masks, not slides

**Status:** Fixed · **Severity:** S1 · **Provenance:** `[OBSERVED]`
**Surfaced in:** gate V0.7 on the real cohort

### Symptom
382 slides downloaded, 239 GB. 131 failed to open; 251 opened cleanly but
reported implausible microns-per-pixel — six clusters, three of them at
exactly 8× the other three.

### The decisive observation
The 8× was exact in every pair, never approximate:
```
1.9447 / 0.2431 = 8.000      1.8106 / 0.2263 = 8.000
1.8153 / 0.2269 = 8.000
```
Exactness rules out scanner variation and points at a structural cause. The
raw dump then showed `axes=YX`, `photometric=1` — single-channel images. These
were not photographs at all.

### Root cause
The bucket holds four parallel trees:
```
CAMELYON16/images/             399 files  700.8 GB   the WSIs
CAMELYON16/masks/              399 files    8.8 GB   rasterised labels
CAMELYON16/background_tissue/  400 files    0.4 GB   tissue masks, 8x downsampled
CAMELYON16/annotations/        160 files    0.1 GB   lesion polygons
```
`masks/` and `background_tissue/` are also `.tif` and also named `normal_*` /
`tumor_*`. The fetch planner classified by filename alone, so it happily
selected them. The three coarse MPP clusters were `background_tissue`, which
is stored at 8× downsample — hence the exact ratio.

Arithmetic confirms it: 144 mask + 107 tissue = 251, exactly the "readable"
set; `(239.34 − 9.2) GB ÷ 1.76 GB/slide = 131`, exactly the failure count.

### Fix
Selection is by bucket prefix, not filename: slides only from
`CAMELYON16/images/`, annotations only from `CAMELYON16/annotations/`, and
`*_mask` / `*_tissue` stems excluded outright. The planner now prints what it
skipped and why, so the exclusion is visible rather than silent.

### Regression guard
`tests/unit/test_fetch_planner.py` — four cases built on the real bucket layout.

### Lesson
A filename convention is not a type. Two directories used the same naming
scheme for entirely different content, and the only reliable discriminator was
the path. The planner should have been reading the structure it was given
rather than the structure it assumed.

---

## BUG-022 — Every real slide rejected for missing microns-per-pixel

**Status:** Fixed · **Severity:** S1 (total blocker) · **Provenance:** `[OBSERVED]`
**Surfaced in:** gate V0.7, once BUG-021 revealed that the 131 "failures" were
the only real slides in the cohort

### Symptom
```
SlideReadError: no microns-per-pixel metadata found
  (checked openslide.mpp-*, aperio.MPP, tiff.*Resolution);
  available keys: ['tiff.ImageDescription', 'tiffslide.background-color', ...]
```
Not one CAMELYON16 image would open. 100% failure, not a long tail.

### Root cause
CAMELYON16 images are Philips scanner exports — `Software = "Philips DP v1.0"`.
They carry **no** `XResolution` tag whatsoever. The pixel spacing is inside an
XML document stored in `ImageDescription`:

```xml
<DataObject ObjectType="DPUfsImport">
  <Attribute Name="PIM_DP_SCANNED_IMAGES" ...>
    <Array>
      <DataObject ObjectType="DPScannedImage">
        <Attribute Name="PIM_DP_IMAGE_TYPE">WSI</Attribute>
        <Attribute Name="DICOM_PIXEL_SPACING">"0.000243" "0.000243"</Attribute>
```

ADR-002 deliberately refuses to guess a missing scale, so the reader did
exactly what it was designed to do — for a format it had never been told about.

### Fix
`_philips_mpp()` parses the Philips block. Three details matter:

1. The file describes **several** scanned images — the WSI plus a macro
   photograph and a label photograph — each with its own spacing. Taking the
   first match would return the macro image's ~0.022 mm, roughly 90× too
   coarse. The parser selects the entry whose `PIM_DP_IMAGE_TYPE` is `WSI`,
   falling back to the finest spacing present.
2. Spacing is in millimetres, not microns.
3. A value outside 0.01–10 µm/px is rejected rather than used.

Cross-check: the parser returns 0.243 µm/px, and `normal_004_mask.tif` from
the same scanner independently reports 0.2431 via its TIFF resolution tag.

`tiffslide.mpp-x` was also added to the lookup chain ahead of the raw tags.

### Alternative worth knowing
OpenSlide has a native Philips TIFF driver. Installing it makes the ADR-001
fallback chain resolve this without any parsing:
`conda install -c conda-forge openslide-python`. Both paths now work; the
parser means the project is not *dependent* on a system library.

### Regression guard
`tests/unit/test_slide_rgba.py` — four cases, including the macro-vs-WSI
selection and the implausible-value rejection.

### Lesson
ADR-002's refuse-rather-than-guess rule behaved correctly and turned a silent
scale error into a loud, diagnosable blocker. The cost was a hard stop; the
alternative was 100% of slides training at an invented scale. That trade was
worth it, and it is the clearest vindication of the loud-failure principle so
far.

---

## BUG-023 — Tissue guard rejected real lymph node slides

**Status:** Fixed · **Severity:** S2 · **Provenance:** `[OBSERVED]`
**Surfaced in:** stage 01 on the first real cohort — 4 of 131 slides rejected

### Symptom
```
tissue fraction 0.0089 below lower bound 0.01; slide is empty or
thresholding is too aggressive
```
Meanwhile the surviving 127 slides ran from 0.0114 to 0.3077, with a median
of 0.0896 — far below the 0.15–0.45 band the design had assumed.

### Root cause
Not a detection failure. A design assumption that did not survive contact
with the data.

ADR-003 set the contract-A2 lower bound as a **fraction of the slide**, on the
generic intuition that a whole-slide image is mostly tissue. CAMELYON16 is
sentinel lymph node sections, and a node is small relative to the glass:

```
3D Histech slide : 23.4 x 53.1 mm = 1244 mm^2 of glass
lowest observed  : tissue_frac 0.0114 -> 14.2 mm^2 -> ~3.8 mm across
```

A 3.8 mm lymph node is entirely normal. The floor was rejecting healthy
slides, and the four that failed were most likely the smallest nodes in the
cohort — not the worst masks.

### Fix
Express the lower bound in **physical area**, matching ADR-012's treatment of
lesion size: `min_tissue_mm2 = 1.0`. A node smaller than about 1 mm across is
implausible; a blank slide is 0.00.

The upper bound stays fractional at 0.90, because "most of the slide is
tissue" really is impossible and that framing is correct for it.

Measured behaviour after the change:
```
lowest observed real slide   frac 0.01112   13.82 mm^2   ACCEPTED
median real slide            frac 0.08868  110.28 mm^2   ACCEPTED
genuinely near-blank         frac 0.00072    0.90 mm^2   REJECTED
```

### Consequence for the fixtures
The synthetic slides are about 2 x 2 mm — roughly 1/300th of a real slide —
so a guard calibrated in mm² rejects them all. That is the guard being right
and the fixtures being miniature. `configs/data_fixtures.yaml` now relaxes
exactly the physical guards and nothing else, so fixture runs exercise the
same code paths at toy scale.

### Regression guard
`tests/unit/test_tissue.py` — four cases built on real CAMELYON16 thumbnail
geometry, including one asserting the guard gives the same answer at two
thumbnail resolutions.

### Lesson
This is the third defect in this project traceable to a threshold expressed
in the wrong units, after ADR-012 and BUG-019. The pattern is consistent
enough to state as a rule: **if a threshold describes something physical,
express it in physical units, even when a fraction or a pixel count is more
convenient to compute.** Gate V0.5 greps for hard-coded pixel constants; it
should arguably also flag fractional thresholds on physical quantities.

---

## BUG-024 — Stale failure list blocked a clean run

**Status:** Fixed · **Severity:** S3 · **Provenance:** `[OBSERVED]`
**Surfaced in:** stage 02, immediately after stage 01 first reported
`ok=131 failed=0`

### Symptom
```
ERROR stage02  artifacts/index/failed_slides.csv is present: 4 slides failed
               stage 01. Acknowledge with --allow-failed, or fix them.
```
Stage 01 had just succeeded on every slide. The four failures were from the
*previous* run, before BUG-023 was fixed.

### Root cause
Stage 01 wrote `failed_slides.csv` when there were failures but never removed
it when there were none. Stage 02's guard therefore tested a file that
recorded history, not current state.

### Why this is worth an entry despite being minor
The guard exists to stop someone silently training on a subset. Firing on a
run that fixed every failure teaches exactly the wrong lesson: the apparent
fix is `--allow-failed`, and once that flag is in the command line it stays
there and the guard is permanently disabled. **A guard that cries wolf is
worse than no guard**, because it converts a safety check into a habit of
overriding safety checks.

### Fix
1. Stage 01 deletes the file when a run has no failures, and stamps a UTC
   timestamp into it when it does.
2. Stage 02 compares the file's mtime against `slides.parquet`. A failure list
   older than the index it accompanies is reported as stale and ignored,
   rather than blocking.

### Regression check
Fixture run with a planted stale `failed_slides.csv`: stage 01 logs
`all slides succeeded; removed stale failed_slides.csv`, and stage 02 proceeds
without `--allow-failed`.

---

## BUG-025 — Every annotation polygon treated as a lesion that must be found

**Status:** Fixed · **Severity:** S1 (invalidates the primary metric)
**Provenance:** `[OBSERVED]` · **Surfaced in:** gate V4.7 on the real index

### Symptom
```
lesions total 1,586
  p50     0.00334 mm^2
below 0.02 mm^2 : 1,208 / 1,586 (76.2%) -- UNFINDABLE by construction
FROC sensitivity is therefore capped at 23.8%
```
A 23.8% ceiling would make the project's primary metric meaningless.

### The observation that reframed it
The sizes are not small lesions. They are cells.
```
p50  0.00334 mm^2  ~=  58 um across
p25  0.00097 mm^2  ~=  31 um across
p5   0.00028 mm^2  ~=  17 um across
```
A tumor cell nucleus is 10-20 um. A 17 um "lesion" is one cell; a 31 um one
is two or three. These are isolated tumor cells, and CAMELYON16 annotates
them because they are real, not because a detector is expected to report each
one as a metastasis.

### Root cause
ADR-013 specified FROC as "sensitivity at N false positives per slide" without
specifying **which ground-truth objects are in the denominator**, so the
implementation used all of them. That is not the CAMELYON16 convention and it
is not clinically meaningful.

Clinically (AJCC), deposits are staged by **largest dimension**, not area:

| category | major axis | stages node positive? |
|---|---|---|
| macrometastasis | >= 2.0 mm | yes (pN1+) |
| micrometastasis | 0.2 - 2.0 mm | yes (pN1mi) |
| isolated tumor cells | < 0.2 mm | **no** (pN0(i+)) |

The official CAMELYON16 evaluation drops lesions with major axis below 275 um
from the ground truth entirely: they are not in the denominator, **and** a
prediction landing on one is not charged as a false positive.

The second half matters as much as the first, and my original scheme got it
doubly wrong: a model that correctly flagged a cluster of tumor cells was
penalised twice -- once for the ITC it could not "hit", and again as a false
positive for having found it.

### Fix
`src/eval/froc.py` now implements the convention:

* `major_axis_um()` measures largest dimension via the **minimum rotated
  rectangle**, not an axis-aligned bounding box. A 100 um lesion lying
  diagonally would otherwise measure 141 um.
* `classify()` returns macro / micro / itc against the clinical boundaries.
* `match()` returns `(detections, n_evaluable, n_itc)`. ITC hits are dropped;
  duplicate hits on one lesion are dropped; only genuine misses are FPs.
* `qc_index.py` reports the distribution by major axis and by clinical
  category.

### Consequence
The 23.8% ceiling does not exist. ITCs leave the denominator, so a sensitivity
of 1.0 stays reachable. Reported numbers also become comparable with the
published CAMELYON16 literature, which was the point of using this benchmark.

### Regression guard
`tests/unit/test_patient_rules.py` -- six cases, including rotation invariance
of the major axis and the ITC-hit-is-not-an-FP rule.

### Lesson
A metric name is not a metric definition. "FROC" sounded specified; the
denominator was not. The defect was invisible in code review and only appeared
when real annotations produced a number too bad to be plausible -- which is
its own useful signal. **An implausibly bad result deserves the same suspicion
as an implausibly good one.**

---

## BUG-026 — Detection filtered by area, evaluation scored by major axis

**Status:** Fixed · **Severity:** S2 · **Provenance:** `[OBSERVED]`
**Surfaced in:** reviewing gate V4.7 output after the BUG-025 fix

### Symptom
None yet — found by comparing two thresholds that had never been compared,
because until BUG-025 they were expressed in the same (wrong) units.

### The inconsistency
After BUG-025, evaluation keeps a ground-truth lesion when its **major axis**
is at least 275 um. Post-processing (ADR-012) deleted a predicted component
when its **area** was below 0.02 mm². Those are different shapes of filter:

```
275 x 275 um   axis 275 um (evaluable)   area 0.0756 mm^2   kept
275 x  50 um   axis 275 um (evaluable)   area 0.0138 mm^2   DELETED
400 x  40 um   axis 400 um (evaluable)   area 0.0160 mm^2   DELETED
```

An elongated deposit — which is a perfectly ordinary shape for tumor tracking
along a sinus — would be found by the model, deleted by the pipeline, and
scored by FROC as a miss. The model would be blamed for a plumbing decision.

### Fix
Post-processing now filters by major axis, measured the same way evaluation
measures it (minimum rotated rectangle, `_major_axes`), at a default of
100 um. That is well below the 275 um evaluability boundary, so **no lesion
FROC can score is ever removed**, while sub-cellular speckle still goes.

The area filter remains available as an optional secondary (`min_lesion_mm2`,
default 0 = disabled).

`Lesion` now carries `major_axis_um`, so the size a component was judged on is
visible in the output rather than recomputed downstream.

### Verification
```
axis filter (new)  : axis 398.0 um  area 0.01584 mm^2   KEPT
area filter (old)  : deleted
10 x 10 um speckle : removed by both
```

### Regression guard
`tests/unit/test_postproc.py` — six cases, including one asserting the filter
threshold stays below the evaluability boundary, and one asserting a predicted
square and a ground-truth square of equal size measure equal.

### Lesson
Two thresholds that gate the same pipeline must be expressed in the same
quantity, or there is a band where one accepts and the other rejects. This is
the fourth units defect in the project (ADR-012, BUG-019, BUG-023, and now
this one) — but the first where both units were individually reasonable and
only the *pairing* was wrong. Gate V0.5 greps for hard-coded pixel constants;
it cannot catch this. A checklist item for reviewing threshold pairs would.

---

## BUG-027 — Appended model config silently overrode the caller's settings

**Status:** Fixed · **Severity:** S2 · **Provenance:** `[OBSERVED]`
**Surfaced in:** first CPU run, as `Killed` with no traceback

### Symptom
```
$ python scripts/03_train.py --config base data_camelyon16 \
      model_unet_effb0 cpu_smoke --dry-run
dry run: device=cpu batch=16 source=slides
Killed
```
`cpu_smoke.yaml` sets `batch_size: 4`. The run used 16 and was killed by the
kernel's OOM reaper, which produces no Python traceback at all -- just the
word `Killed`.

### Root cause
Stage 03 did this:
```python
cfg = load(a.config + ["configs/model_unet_effb0.yaml"], a.set)
```
Config files merge left to right, so **anything appended overrides everything
the caller passed**. The caller's trailing `cpu_smoke.yaml` was clobbered by a
file the script added after it.

The convenience was real -- callers did not have to name the model config --
but it silently inverted the precedence the whole config system is built on.

### Fix
Insert the default **first**, and only when the caller did not name a model
config themselves:
```python
paths = list(a.config)
if not any("model" in Path(p).stem for p in paths):
    paths.insert(0, "configs/model_unet_effb0.yaml")
```
Caller configs now always win, and the common case still needs no model file.

Stage 03 also now logs the resolved config list and the effective
device / batch / workers / epochs at startup. A config that silently did not
apply is indistinguishable from one that did, right up until the run dies.

### Also fixed in the same pass
* `float(loss)` on a tensor with `requires_grad` warned on every log line;
  now `float(loss.detach())`.
* A CPU run now estimates its activation budget against physical RAM and warns
  with a concrete smaller batch size, because `Killed` carries no diagnostic
  information whatsoever.
* `cpu_smoke.yaml` drops workers from 4 to 2.

### Regression guard
`tests/unit/test_patient_rules.py` -- four cases covering trailing-config
precedence, default insertion, the patchset path, and `--set` overriding
every file.

### Lesson
A default that is applied *after* user input is not a default; it is an
override wearing a default's name. Defaults belong at the start of a merge
chain, never the end.

---

## BUG-028 — The overfit gate was never implemented

**Status:** Fixed · **Severity:** S1 (a gate that reported nothing)
**Provenance:** `[OBSERVED]` · **Surfaced in:** first real CPU run of stage 03

### Symptom
```
e00 s00000 loss=0.8671
e00 s00050 loss=0.8521
e00 s00100 loss=0.6548
e00 s00150 loss=0.5775
e00 s00190 loss=0.8579
```
Expected `loss <= 0.05` by step 200. It bounced between 0.57 and 0.92 instead.

### Root cause
`train.overfit_batches` appeared in `configs/cpu_smoke.yaml`, in
`docs/verification_checklist.md`, and in every instruction that referenced
gate V6.2 — but **nothing in `scripts/03_train.py` read it**. The loop pulled
a fresh batch from the DataLoader every step, so the run was ordinary training
on 200 different batches. A loss wandering between 0.57 and 0.92 over 200
unrelated batches is unremarkable. It was neither a pass nor a failure; the
test simply did not exist.

This was the check described throughout the documentation as the most
informative in the project: "a model that cannot memorise one batch has a
wiring defect no amount of data will fix". It had never run.

### How it survived so long
Every earlier reference to V6.2 was a *plan* to run it, and the config key
looked implemented because it was present and documented. The failure mode is
specific to unimplemented options: an unknown key in a permissive config
system is silently ignored, so the command succeeds, produces plausible
output, and answers a different question than the one asked.

### Fix
* `overfit_batches` pins N batches and reuses them for every step.
* Validation in overfit mode scores those same pinned batches — "can it
  memorise these" is only answerable on those, not on a held-out set.
* The run now prints an explicit verdict, `GATE V6.2 ... PASS/FAIL`, and exits
  non-zero on failure. A gate that does not state its own verdict invites
  exactly this kind of silent no-op.
* The pinned batch's positive fraction is logged, with a warning if it is
  almost pure background — Dice on an empty target is degenerate (BUG-006)
  and would make the gate unreadable for a different reason.

### Also fixed in the same pass
`max_val_batches`. After `max_steps` the loop validated the full split:
39,809 patches at batch 4 is ~10,000 forward passes, hours on CPU, printing
nothing. Indistinguishable from a hang, and reported as one. Validation now
logs progress and can be capped.

### Lesson
A config key is not a feature. In a permissive config system an unimplemented
option is silently accepted, and the resulting run looks like a successful
test of something it never tested. Any option that gates a decision should
state its verdict explicitly and fail loudly when unmet — as this one now
does.

---

## BUG-029 — Overfit gate passed at epoch 0, then ran for another hour

**Status:** Fixed · **Severity:** S3 · **Provenance:** `[OBSERVED]`
**Surfaced in:** the first successful run of gate V6.2

### Symptom
```
e00 val dice=0.9988 iou=0.9977 tau=0.35     <- gate criterion (>=0.97) already met
...
e07 s00150 loss=0.2561                      <- still running 56 minutes later
```
The gate's verdict was only printed after the epoch loop finished, and
`max_epochs` was 40. So a question answered in nine minutes took over three
hours to report.

### Fix
Overfit mode now breaks out the moment validation Dice reaches the threshold,
logs `overfit criterion met at epoch N; stopping early`, and returns.
`cpu_smoke.yaml` also caps `max_epochs` at 8 as a backstop.

### Lesson
A gate answers a yes/no question. It should stop as soon as it can answer it,
and say so at that moment -- not accumulate evidence and report at the end.

---

## OBS-001 — Excellent overlap, negligible confidence

**Status:** Observed, not a defect · **Provenance:** `[OBSERVED]`
**Surfaced in:** the same run

### The observation
```
e06 val dice=0.9999 iou=0.9999 tau=0.80     train loss plateaued at ~0.375
```
Dice of 0.9999 with a loss stuck at 0.375 is contradictory only until the loss
is decomposed:

```
loss   = 0.5*BCE(pos_weight=2) + 0.5*SoftDice
dice   = 0.9999  ->  Dice term ~ 0.0001
so      0.5*BCE ~ 0.375  ->  BCE ~ 0.75
```

A BCE of 0.75 on this class balance corresponds to probabilities of roughly
**0.52** on the correct side. The model has learned the shape essentially
perfectly and has almost no confidence margin.

The fitted threshold corroborates it, wandering 0.35 -> 0.55 -> 0.80 -> 0.60
-> 0.60 -> 0.75 -> 0.80 across epochs. When both classes cluster near 0.5,
many thresholds separate them equally well and tau is underdetermined.

### Why it matters
Dice is threshold-swept, so it reports the best achievable overlap and is
blind to margin. At inference the pipeline uses **one** fitted tau (ADR-006).
A model with a 0.02 margin will have an unstable tau and will be fragile to
any distribution shift, while looking solved by the headline metric.

### Response
Validation now reports mean predicted probability per class and the margin
between them, and warns when Dice exceeds 0.9 with a margin under 0.10.

This is expected during a short low-LR run -- the loss was still descending
(0.375 -> 0.256 by epoch 7, consistent with p ~ 0.80) -- so it is recorded as
something to watch on the real GPU run, not a defect. If the margin is still
under ~0.2 after real training converges, tau is not trustworthy and
calibration needs attention before any FROC number is reported.

---

## BUG-030 — Wrong conda environment surfaced as an unrelated AttributeError

**Status:** Fixed · **Severity:** S3 · **Provenance:** `[OBSERVED]`
**Surfaced in:** first patchset export

### Symptom
```
src.io.slide.SlideReadError: could not open normal_004.tif with any backend:
  tiffslide: AttributeError: module 'numpy.dtypes' has no attribute 'StringDType'
```
wrapped in a `SlideReadError`, wrapped again in a `ProcessPoolExecutor`
traceback.

### Root cause
The script was run from Anaconda's `base` environment rather than `wsi`. Base
carries numpy 1.26.4 -- deliberately restored there after BUG-021's fallout --
and `numpy.dtypes.StringDType` only exists in numpy 2, which tiffflile
requires.

So the chain was: wrong environment -> old numpy -> tiffslide import fails ->
backend chain exhausted -> SlideReadError -> pool traceback. Four layers, none
of which mention an environment.

### Fix
`src/utils/envcheck.py`, called at the top of every stage script. It checks
numpy's major version and that tiffslide, shapely and cv2 actually import,
then raises with the interpreter path, the numpy version, the exact symptom
the user would otherwise have seen, and -- when the prefix looks like a conda
base install -- the literal command to fix it.

`scripts/version.py` now also prints the interpreter and numpy version, so
"which environment am I in" is answered by the command already used to check
everything else.

### Regression guard
`tests/unit/test_envcheck.py` -- four cases, including one asserting the
message names `StringDType`, because the whole point is that the user can
match the error they saw against the explanation.

### Lesson
An error raised four layers below the mistake will describe the layer, not the
mistake. Where a whole class of failures has one cause -- here, the wrong
interpreter -- check for that cause up front rather than letting each
downstream component report its own symptom.

---

## BUG-030 — worker_init assumed the slide-backed dataset

**Status:** Fixed · **Severity:** S1 (blocked all patchset training)
**Provenance:** `[OBSERVED]` · **Surfaced in:** first Colab dry run, gate V6.1

### Symptom
```
AttributeError: Caught AttributeError in DataLoader worker process 0.
  File "src/data/dataset.py", line 116, in worker_init
    ds._readers.clear(); ds._geoms.clear(); ds._pid = os.getpid()
AttributeError: 'PatchSetDataset' object has no attribute '_readers'
```
Every worker died immediately. No patchset training was possible.

### Root cause
`worker_init` exists to stop slide handles being inherited across `fork`
(BUG-002), so it reaches straight for `_readers` and `_geoms`. When the
patchset backend was added (0.7.0) it reused the same `worker_init` -- the
training loop passes it unconditionally -- but `PatchSetDataset` reads loose
files and has neither attribute.

The two backends were written to share an output contract (`image`, `mask`,
`meta`) and that contract was honoured. What was missed is that they also
share a *lifecycle* hook, and nothing stated what a dataset had to provide for
it.

### Fix
`worker_init` now resets only what is present:

* `_readers` / `_geoms` / `_pid` -- slide-backed only.
* `rng` and `tf.rng` -- **both** backends, and this is the part that matters
  for correctness rather than just not crashing. Without the reseed every
  worker inherits the same seed and emits an identical augmentation sequence,
  which quietly removes most of the augmentation's value. `PatchSetDataset`
  now carries an `rng` so it participates.

### Regression guard
`tests/unit/test_worker_init.py` -- four cases: slide-like, patchset-like, a
dataset with neither attribute, and an assertion that two workers draw
different augmentation streams. The tests stub `torch` in `sys.modules` so
they run without it installed.

### Lesson
A shared hook is an interface. Two implementations of a dataset agreed on
their output contract and were assumed interchangeable, but nothing recorded
what the *framework* would call on them. The gate caught it in seconds --
which is the argument for V6.1 existing at all.

---

## BUG-031 — bfloat16 autocast on a pre-Ampere GPU ran ~6x slow

**Status:** Fixed · **Severity:** S2 (throughput) · **Provenance:** `[OBSERVED]`
**Surfaced in:** gate V6.2 on Colab

### Symptom
```
13:25:09  e00 s00000 loss=0.9453
13:27:28  e00 s00050 loss=0.5757      -> 2.66 s/step
```
Against ~0.45 s/step expected for UNet-EffNetB0 at 512 px, batch 16, on a T4.

The measurement was unusually clean: in overfit mode the batch is pinned and
reused, so no data loading happens at all. The 2.66 s was pure GPU compute,
which rules out the DataLoader, the 2-core VM, and JPEG decode in one stroke.

### Root cause
ADR-011 specified `amp_dtype: bfloat16`, justified on two grounds: no loss
scaler needed, and better numerical headroom for Dice sums over 262k pixels.
Both hold -- **on Ampere**.

```
GPU    arch     sm    native bf16    fp16 tensor cores
T4     Turing   75    NO             yes
V100   Volta    70    NO             yes
A100   Ampere   80    yes            yes
```

bfloat16 requires sm_80. On sm_75 autocast still runs, but down a fallback
path with no tensor-core acceleration. Measured 5.9x slower, which matches the
gap almost exactly.

The config was hardware-dependent and did not say so, and the default was set
against the hardware the design assumed rather than the hardware it would meet.

### Fix
The dtype is now chosen from `torch.cuda.get_device_capability()` at startup:
bfloat16 on sm_80+, float16 below. float16 needs a `GradScaler` -- its
exponent range is narrow enough that gradients underflow to zero without loss
scaling -- so one is constructed and enabled only on that path.

`scaler.unscale_(opt)` runs before `clip_grad_norm_`, or the clip threshold
would be applied to scaled gradients and mean nothing.

The GPU name and compute capability are now logged at startup, so this class
of mismatch is visible in every run's first lines.

### Lesson
A config default encodes an assumption about hardware. Stating the preference
(`bfloat16`) without stating the requirement (`sm_80+`) meant the system
silently degraded instead of adapting. Anything in a config that only works on
some hardware should be resolved at runtime against the hardware actually
present -- and logged.

---

## OBS-002 — Data pipeline became the bottleneck once the GPU was fixed

**Status:** Partially addressed · **Provenance:** `[OBSERVED]`
**Surfaced in:** first real training run on Colab, after BUG-031

### The measurement
```
pinned batch, no data loading (gate V6.2)   0.30 s/step
real training, augment=True, 2 workers      0.98 s/step
                                            ------------
data pipeline                               ~0.68 s/step
```
Fixing the autocast dtype moved the bottleneck from the GPU to the CPU, which
on a 2-core VM is where it now sits.

### Profile, per patch
```
JPEG decode                3.22 ms
geometric D4               1.21 ms
HED stain jitter          12.70 ms   <- 62% of the cost
GaussianBlur               0.80 ms
JPEG re-encode (aug)       3.31 ms   <- redundant, see below
normalize -> CHW float32   5.62 ms
```

### What was fixed
1. **Stain jitter 12.70 -> 8.65 ms.** The forward transform begins with
   `-log(x/255)` over 786k pixels, but the input is uint8, so there are only
   256 possible results: a 256-entry lookup replaces the log entirely. The two
   3x3 matmuls also collapse into one, since the whole operation is a single
   affine map in optical-density space. Verified: at zero jitter the round
   trip differs from the input by at most 1 grey level.
2. **JPEG re-encode augmentation disabled by default.** Simulating compression
   artifacts makes sense on lossless source data. An exported patch set is
   already JPEG q90 -- re-encoding models nothing the network has not seen and
   cost 3.3 ms on the critical path.
3. **Probabilities are now config** (`data.p_stain`, `p_blur`, `p_jpeg`), so
   augmentation can be traded against throughput without editing code.

Net effect roughly 20%: ~21.4 -> ~17.2 ms per patch.

### What was NOT fixed, and why it is probably the larger cost
The dataset returns **float32 CHW**, so one batch of 16 is
`16 x 3 x 512 x 512 x 4 B = 50 MB` shipped through DataLoader worker IPC every
step. As uint8 it would be 12.6 MB. Normalising on the GPU instead would cut
both the 5.62 ms CPU cost and 75% of the IPC volume.

Not done here because it changes the dataset output contract and would touch
both backends plus the inference path, and a run was in progress. Recorded as
the first thing to try if throughput matters again.

### Note on the profile numbers
They were measured on a development container whose cores are considerably
faster than the Colab VM's. The 0.68 s/step observed there is ~6x the ~115 ms
these figures predict, so treat the profile as a guide to *proportions*, not
as absolute timings.

---

## BUG-032 — Capped validation took the head of an ordered split

**Status:** Fixed · **Severity:** S1 (metric silently meaningless)
**Provenance:** `[OBSERVED]` · **Surfaced in:** first real Kaggle training run,
end of epoch 0

### Symptom
```
validated on 150 batches of 707
e00 val dice=0.0000 iou=0.0000 tau=0.10 | p(pos)=0.000 p(neg)=0.032 margin=-0.032
```
After a full epoch -- three hours -- validation Dice was exactly zero. It looks
like the model collapsed to predicting all-negative (BUG-006's signature).

### What identified it instead
Two values pin the cause precisely:

* `p(pos)` is **exactly** 0.000. The accumulator computes
  `sum(prob over positive pixels) / max(n_pos, 1)`, so an exact zero means
  `n_pos == 0` -- there were no positive pixels to average over.
* `tau` is **exactly** 0.10, the lowest value in the sweep. `best()` takes
  `argmax` over per-threshold Dice; with every Dice at zero, argmax returns
  index 0.

Neither is what a collapsed model produces -- a collapsed model still has
positive pixels in its ground truth. Both say the same thing: the validation
subset contained no tumour at all.

### Root cause
`train.max_val_batches` was implemented as `itertools.islice(val_dl, n)` --
the **first** n batches of a loader with `shuffle=False`.

The exporter writes patches grouped by slide, so the manifest is ordered by
`slide_id`. The first 2,400 of 11,310 validation patches therefore came from
the first few slides in that order, and those were normal slides. Zero
positives.

```
head of an ordered split : 0.0000 tumour   <- what happened
random sample            : 0.2329 tumour   <- what was intended
```

The cap was added to stop validation dominating epoch time (BUG-028's second
half). It fixed that and quietly broke the metric.

### Why it mattered beyond the log line
Three things read validation Dice: early stopping, `best.pt` selection, and
the fitted threshold stored in the checkpoint. With Dice pinned at zero,
`best.pt` would never improve after epoch 0, early stopping would fire after
its 8-epoch patience, and the checkpoint would carry tau = 0.10 -- which
inference uses directly.

Training itself was unaffected: the sampler shuffles, so the model was
learning normally the whole time.

### Fix
The subset is now **sampled** across the split with a fixed seed, drawn once
before the epoch loop so the numbers stay comparable between epochs. The
proportion carrying tumour is logged, and a subset with none raises rather
than reporting a number. A separate guard refuses to record any validation
result computed over zero positive pixels.

### Lesson
`shuffle=False` plus a head-slice is a sampling method, and a bad one whenever
the underlying order carries structure. The order here was meaningful --
grouped by slide -- which is exactly the case where taking a prefix stops
being a sample and becomes a selection. If a cap exists to save time, it still
has to be representative, or it is not measuring what its name says.

---

## BUG-033 — Resume searched only one of the two places a checkpoint can be

**Status:** Fixed · **Severity:** S2 · **Provenance:** `[OBSERVED]`
**Surfaced in:** Kaggle, after training stopped at epoch 6

### Symptom
Re-running the notebook restarted from epoch 0 with no warning, discarding six
epochs of work.

### Root cause
The resume cell globbed `/kaggle/input/*/ckpt/*/last.pt` only. Checkpoints can
be in two places:

```
/kaggle/working/ckpt/<run>/last.pt      this session (a re-run)        <- missed
/kaggle/input/<ver>/ckpt/<run>/last.pt  a previous version, attached   <- found
```

Worse, it said nothing when it found neither -- it printed a single line about
starting from scratch and proceeded. Kaggle wipes `/kaggle/working` between
sessions, so unless the previous version is explicitly attached under *Input*
there is genuinely nothing to resume from, and that is a user action the
notebook cannot perform for itself.

### Fix
Search both locations, report which one was used, and when neither exists say
plainly what has to be done (`Input -> Add Input -> Notebook Output`) rather
than quietly beginning again.

### Lesson
A silent fallback to a reasonable default is the wrong behaviour when the
default discards hours of work. Resuming is an intent; failing to resume
should be loud.

---

## OBS-003 — Drive rate-limits gdown on a folder of large files

**Status:** Mitigated · **Provenance:** `[OBSERVED]`

### Symptom
```
100%|##########| 1.01G/1.01G [00:10<00:00, 94.9MB/s]   shard_0000.tar
Failed to retrieve file url:
    Cannot retrieve the public link of the file. You may need to change
    the permission to 'Anyone with the link', or have had many accesses.
```
One shard at 95 MB/s, then refusal. The permission was correct; the limit is
on how often a public link may be resolved.

### Why it happens
`gdown` does not use the Drive API. It scrapes the public download page per
file, and Drive throttles that path. A folder of eleven 1 GB files trips it
quickly, and gdown aborts the whole folder rather than the one file.

### Mitigation
1. **Retry.** gdown skips files already present, so each attempt fetches only
   what is missing. Up to six attempts, stopping after two with no progress.
2. **rclone fallback.** rclone uses the Drive API with your own OAuth token
   and backs off properly. Enabled by putting `rclone.conf` in a Kaggle Secret
   named `RCLONE_CONF`. Keep the notebook private -- that file holds a
   refresh token.
3. Once any complete copy exists, publish it as a Kaggle Dataset. Later
   sessions attach it and Drive is never touched again.

The 95 MB/s on the first shard is the useful number: the route is sound, only
the link-resolution step is limited.
