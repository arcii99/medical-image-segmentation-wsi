# Architecture — Gigapixel WSI Tumor Segmentation

**Project:** `wsi-metastasis-seg`
**Domain:** H&E-stained sentinel lymph node sections (CAMELYON16/17 format)
**Task:** Binary pixel-wise segmentation of metastatic tissue, reconstructed as a
slide-level spatial probability heatmap.

---

## 0. The Core Architectural Constraint

A single WSI at 40x is a pyramidal TIFF whose base level can reach
`100,000 x 100,000 x 3` bytes = **30 GB uncompressed**. At the working
magnification of 20x it is still ~7.5 GB. No GPU holds this, and no
`np.ndarray` should ever hold this.

Every architectural decision below follows from one rule:

> **Nothing at level-0 resolution is ever materialised in full.**
> The slide is touched only through (a) low-resolution pyramid levels for
> global reasoning, and (b) bounded random-access windows for local reasoning.

The pipeline is therefore a **coordinate-first** architecture. The unit of data
that moves between modules is not an image — it is a `(slide_id, x0, y0, level)`
tuple. Pixels are materialised as late as possible and discarded immediately.

---

## 1. Top-Level Block Diagram

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                            STAGE A — INGEST (CPU, offline)                   │
│                                                                              │
│   data/raw/*.tif ──┐                                                         │
│   (pyramidal WSI)  │                                                         │
│                    ▼                                                         │
│            ┌───────────────┐        ┌────────────────────┐                   │
│            │ SlideReader   │        │ AnnotationParser   │  data/raw/*.xml   │
│            │ (tiffslide /  │        │ XML → shapely →    │◄──────────────    │
│            │  openslide)   │        │ raster mask        │                   │
│            └───────┬───────┘        └─────────┬──────────┘                   │
│                    │ level_k thumbnail                  │                    │
│                    ▼ (~1.25x, 3000x3000)                │                    │
│            ┌───────────────────────┐                    │                    │
│            │ TissueLocalizer       │                    │                    │
│            │ HSV-S Otsu ∪ Gray Otsu│                    │                    │
│            │ − pen/artifact filter │                    │                    │
│            │ + morphological close │                    │                    │
│            └───────┬───────────────┘                    │                    │
│                    │ tissue_mask (bool, level_k)        │                    │
│                    ▼                                    ▼                    │
│            ┌──────────────────────────────────────────────┐                  │
│            │ PatchIndexer                                 │                  │
│            │ grid × tissue_mask × tumor_mask → coords     │                  │
│            │ label each coord: tumor_frac ∈ [0,1]         │                  │
│            └───────────────────┬──────────────────────────┘                  │
│                                ▼                                             │
│                  artifacts/index/patches.parquet                             │
│                  (slide_id, x0, y0, level, mpp, tumor_frac, split)           │
└──────────────────────────────────────────────────────────────────────────────┘
                                 │
                                 │  coordinates only  (~10 MB for 400 slides)
                                 ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                          STAGE B — TRAIN (GPU, online)                       │
│                                                                              │
│   patches.parquet ──► BalancedSampler ──► PatchDataset ──► DataLoader        │
│                       (1 tumor : 3 normal)      │           (8 workers)      │
│                                                 │ lazy read_region           │
│                                                 ▼                            │
│                                     ┌───────────────────────┐                │
│                                     │ uint8 512×512×3 patch │                │
│                                     │ uint8 512×512  mask   │                │
│                                     └──────────┬────────────┘                │
│                                                ▼                             │
│                                     ┌───────────────────────┐                │
│                                     │ Augment: flip/rot90,  │                │
│                                     │ HED stain jitter,     │                │
│                                     │ blur, JPEG artifact   │                │
│                                     └──────────┬────────────┘                │
│                                                ▼                             │
│                              x: [B,3,512,512] float32, ImageNet-normalised   │
│                                                ▼                             │
│                    ┌───────────────────────────────────────────┐             │
│                    │  SegModel: UNet(EfficientNet-B0)          │             │
│                    │  encoder 5 stages → decoder 5 stages      │             │
│                    │  logits [B,1,512,512]                     │             │
│                    └───────────────────┬───────────────────────┘             │
│                                        ▼                                     │
│                        Loss = 0.5·BCEWithLogits + 0.5·SoftDice               │
│                        AdamW + cosine, AMP (bf16), grad-clip 1.0             │
│                                        ▼                                     │
│                        artifacts/ckpt/{run_id}/best.pt                       │
└──────────────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                    STAGE C — INFER & RECONSTRUCT (GPU, per slide)            │
│                                                                              │
│   unseen WSI ──► TissueLocalizer ──► SlidingWindowPlanner                    │
│                                       stride 256 (50% overlap)               │
│                                       tissue-gated positions only            │
│                                              │                               │
│                                              ▼                               │
│                                    batched read_region (prefetch thread)     │
│                                              ▼                               │
│                                       model.forward + sigmoid                │
│                                       + 4x D4 TTA (optional)                 │
│                                              ▼                               │
│                            ┌─────────────────────────────────────┐           │
│                            │ GaussianStitcher                    │           │
│                            │ num  += prob · w   (zarr float32)   │           │
│                            │ den  += w          (zarr float32)   │           │
│                            │ written at 1/4 of 20x = 2.0 µm/px   │           │
│                            └──────────────┬──────────────────────┘           │
│                                           ▼                                  │
│                                  heatmap = num / (den + ε)                   │
│                                           ▼                                  │
│              ┌────────────────────────────┴────────────────────┐             │
│              ▼                                                 ▼             │
│   artifacts/heatmaps/{slide}.zarr            artifacts/overlays/{slide}.tif  │
│   (float16, chunked 1024²)                   (pyramidal RGB overlay, OME)    │
│              │                                                               │
│              ▼                                                               │
│   Post-proc: threshold τ, remove CC < 0.02 mm², fill holes                   │
│              ▼                                                               │
│   Slide metrics: Dice/IoU · connected-component FROC · slide-level AUC       │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. The Resolution Ladder

Three distinct resolutions are in play simultaneously. Confusing them is the
single largest source of geometric bugs in WSI code (see `bug.md`, BUG-003).

| Name | Symbol | Typical MPP | Typical size | Used for |
|---|---|---|---|---|
| **Level 0 (native)** | `L0` | 0.25 µm/px (40x) | 90k × 210k | Coordinate space **only** |
| **Working level** | `Lw` | 0.50 µm/px (20x) | 45k × 105k | Patch pixels, model input |
| **Thumbnail level** | `Lt` | 8.00 µm/px (1.25x) | 2.8k × 6.6k | Tissue mask, planning, QC |

**Invariant (enforced in `SlideReader.read_region`):**

```
read_region(location, level, size)
        ↑                    ↑
   ALWAYS level-0        ALWAYS level-`level` pixels
```

All coordinates persisted anywhere in the project — parquet index, logs,
heatmap metadata, annotations — are **level-0 coordinates**. Conversion happens
at exactly one place:

```python
scale = slide.level_downsamples[lw]          # e.g. 2.0
x0_L0 = x0_Lw * scale                        # index → read_region
xt    = x0_L0 / slide.level_downsamples[lt]  # index → thumbnail/mask
```

`level_downsamples` is **not** guaranteed to be exactly `2**k` (scanners emit
1.9997…). Always read it from the file; never assume.

---

## 3. Data Contracts Between Modules

Each arrow in the block diagram is a typed contract. Breaking one should fail
loudly at the boundary, not silently three stages later.

### A1 — `SlideReader` → everything
```python
@dataclass(frozen=True)
class SlideMeta:
    slide_id: str
    path: Path
    level_count: int
    level_dims: list[tuple[int, int]]     # (w, h) per level
    level_downsamples: list[float]
    mpp_x: float                          # microns per pixel at level 0
    mpp_y: float
    vendor: str
    backend: Literal["tiffslide", "openslide", "cucim"]

read_region(x0_L0: int, y0_L0: int, level: int, w: int, h: int) -> np.uint8[h, w, 3]
# Post-condition: RGB, alpha already composited onto white, C-contiguous.
```

### A2 — `TissueLocalizer` → `PatchIndexer`
```python
TissueMask = np.bool_[H_t, W_t]     # at thumbnail level `lt`
# Post-condition: 0.01 <= mask.mean() <= 0.90, else raise TissueDetectionError
```

### A3 — `AnnotationParser` → `PatchIndexer` / `PatchDataset`
```python
TumorGeometry = shapely.MultiPolygon   # level-0 coords, exclusions subtracted
rasterize(geom, x0_L0, y0_L0, level, w, h) -> np.uint8[h, w]   # {0, 1}
```
Polygons are kept as **vector geometry** until the moment a patch is read.
Rasterising a 45k × 105k mask per slide would cost 4.7 GB; rasterising a
512 × 512 window costs 256 KB.

### A4 — `PatchIndexer` → training (the pipeline's narrow waist)
`artifacts/index/patches.parquet`

| column | dtype | meaning |
|---|---|---|
| `slide_id` | `str` (dict-encoded) | join key to `slides.parquet` |
| `x0`, `y0` | `int32` | top-left, **level-0** coords |
| `level` | `int8` | working level resolved from MPP |
| `size` | `int16` | 512 |
| `tissue_frac` | `float32` | from thumbnail mask |
| `tumor_frac` | `float32` | from rasterised geometry, 0.0 for normal slides |
| `split` | `str` | `train` / `val` / `test`, assigned **per patient** |

Roughly 40 bytes/row; a 400-slide CAMELYON16 index is ~3 M rows ≈ 120 MB —
trivially loadable, sortable, and re-samplable without touching a pixel.

### A5 — `SegModel` → `GaussianStitcher`
```python
probs: torch.float32[B, 1, 512, 512]   # post-sigmoid, in [0, 1]
coords: list[tuple[int, int]]          # level-0 top-left per batch element
```

### A6 — `GaussianStitcher` → evaluation
```
artifacts/heatmaps/{slide_id}.zarr
  ├── .zattrs      {mpp: 2.0, level0_shape: [H0, W0], downsample: 4.0,
  │                 model_run_id: ..., stride: 256, tta: false}
  └── heat         float16, shape (H0/8, W0/8), chunks (1024, 1024), blosc-zstd
```

---

## 4. Tensor / Memory Budget

Single 16 GB GPU, batch 16, 512², UNet-EffNetB0, bf16 AMP:

| Item | Shape | Bytes |
|---|---|---|
| Input batch | `[16, 3, 512, 512]` fp32 | 50 MB |
| Encoder activations (5 stages, cached for skips) | — | ~2.1 GB |
| Decoder activations | — | ~1.4 GB |
| Params + AdamW states (6.3 M × 16 B) | — | 101 MB |
| Gradients | — | 25 MB |
| **Peak observed** | | **~5.8 GB** |

Headroom is deliberate: batch 16 is chosen so a second experiment can share the
card, and so switching to 768² for an ablation does not require a code change.

CPU side, per DataLoader worker:
- 1 open slide handle (~40 MB of libtiff/zarr page cache)
- 1 decoded patch + mask (~1 MB)
- 8 workers → **< 400 MB RSS**, independent of slide size.

---

## 5. Throughput Budget

Measured targets on 1× T4 + 8 vCPU + NVMe (see `verification_checklist.md` V7, V8).

**Training**
```
JPEG-decode 512² patch from SVS  ≈ 12 ms
8 workers                        → ~660 patches/s  (IO ceiling)
fwd+bwd batch16 bf16             ≈ 450 ms → 36 patches/s  (GPU ceiling)
```
GPU-bound by ~18×. Conclusion: **do not pre-extract patches to disk.** The
coordinate-index design costs nothing here and saves ~2 TB (see `decision.md`,
ADR-005).

**Inference, one 45k × 105k slide at 20x**
```
grid positions, stride 256       = (45000/256) × (105000/256) ≈ 72,000
tissue-gated (≈ 28 % tissue)     ≈ 20,000 patches
fwd-only batch 32 bf16           ≈ 190 ms → 168 patches/s
                                 ≈ 120 s GPU
+ IO overlapped via prefetch     → ~2.5 min/slide wall clock
+ 4× D4 TTA                      → ~9 min/slide
```

**Heatmap storage**
```
naive float32 at 20x : 45k × 105k × 4 B        = 18.9 GB   ✗
chosen  float16 at 2.0 µm/px (÷8 from L0)      = 148 MB    ✓
```

---

## 6. Reconstruction Mathematics

Non-overlapping tiling produces visible grid seams: a CNN's receptive field is
truncated at patch borders, so border pixels are systematically less accurate.
The fix is overlap plus weighted averaging.

For a patch at level-0 origin `(x0, y0)` with per-pixel probability `p(u, v)`,
accumulate into two zarr arrays with a separable Gaussian window:

```
w(u, v) = exp( -((u - c)² + (v - c)²) / (2σ²) ),   c = 256, σ = S/8 = 64

num[y, x] += p(u, v) · w(u, v)
den[y, x] += w(u, v)

heat = num / (den + 1e-8)
```

Properties that matter:
- **Partition of unity is not required.** Dividing by `den` normalises any
  window, so the stride and σ can be tuned independently.
- `σ = S/8` gives `w ≈ 0.0003` at the patch border → border predictions are
  effectively vetoed wherever a neighbouring patch covers the same pixel.
- With stride `S/2` every interior pixel is covered by 4 patches, but **`den`
  is not near 4.0** — measured interior range is `[0.074, 1.00]`, because at
  `σ = S/8` the four contributing weights at the worst-case point are ~0.018
  each. An earlier version of this document claimed `[0.9, 4.0]`; that was
  wrong.
- The window carries a **floor of 1e-3** (`w = gauss + 1e-3`). Without it the
  raw weight at a patch corner is ~1.4e-7, and since nothing can extend past
  the slide edge, the outermost ~40 heatmap pixels are covered by exactly one
  patch at that weight. `num/den` is still algebraically correct there, but a
  7e6:1 dynamic range in a float32 accumulator is not worth relying on. The
  floor bounds it at 1000:1 and moves the interior minimum only 0.074 → 0.078.
- Coverage is therefore asserted on `den < 1e-6` (**never written** — a real
  hole) rather than on a low weight. Low-weight area is reported separately,
  not treated as an error.
- Accumulators are **float32**; only the final quotient is cast to float16.
  Accumulating in float16 overflows after ~2000 additions (see BUG-009).

Tissue-gating means the heatmap is undefined outside tissue. Those pixels are
written as `0.0` and masked out of every metric.

---

## 7. Module Responsibility Map

| Module | Owns | Must NOT |
|---|---|---|
| `src/io/slide.py` | backend selection, coord conversion, RGBA→RGB | know about tumors, models, or patches |
| `src/io/annotations.py` | XML → geometry, exclusion subtraction, rasterisation | rasterise anything larger than one patch |
| `src/preprocess/tissue.py` | thumbnail → bool mask, pen/artifact rejection | open the slide at level 0 |
| `src/preprocess/patching.py` | grid generation, tissue/tumor fraction labelling | read RGB pixels |
| `src/data/dataset.py` | lazy pixel materialisation, augmentation | decide sampling ratios |
| `src/data/sampler.py` | class-balanced + hard-negative sampling | touch the filesystem |
| `src/models/` | pure `nn.Module`s, no IO | log, checkpoint, or read config files |
| `src/train/loop.py` | optimisation, AMP, checkpointing, metrics | define architecture |
| `src/infer/sliding_window.py` | position planning, batching, prefetch | stitch |
| `src/infer/stitch.py` | zarr accumulation, normalisation | run the model |
| `src/eval/` | Dice/IoU, FROC, slide AUC | mutate heatmaps |

The dependency graph is strictly acyclic and layered:
`utils ← io ← preprocess ← data ← models ← train ← infer ← eval`.
`models/` importing from `io/` is a design regression and is checked by an
import-linter rule in CI (`verification_checklist.md`, V0.4).

---

## 8. Repository Layout

```
wsi-metastasis-seg/
├── configs/
│   ├── base.yaml                 # paths, seeds, hardware
│   ├── data_camelyon16.yaml      # mpp targets, patch size, split policy
│   ├── model_unet_effb0.yaml
│   ├── model_mask2former.yaml    # ablation, deferred (ADR-007)
│   └── infer_default.yaml
├── src/
│   ├── io/            slide.py  annotations.py
│   ├── preprocess/    tissue.py  patching.py
│   ├── data/          dataset.py  sampler.py  transforms.py  index.py
│   ├── models/        registry.py  unet_effb0.py  mask2former.py
│   ├── train/         loop.py  losses.py  metrics.py  schedule.py
│   ├── infer/         sliding_window.py  stitch.py  heatmap.py  postproc.py
│   ├── eval/          froc.py  slide_metrics.py  report.py
│   └── utils/         config.py  logging.py  seed.py  geometry.py
├── scripts/
│   ├── 00_fetch_data.py
│   ├── 01_build_tissue_masks.py
│   ├── 02_build_patch_index.py
│   ├── 03_train.py
│   ├── 04_infer_slide.py
│   └── 05_evaluate.py
├── tests/             unit/  integration/  fixtures/
├── docs/              Architecture.md  decision.md  flow.md
│                      bug.md  verification_checklist.md  rollback.md
├── artifacts/         (gitignored, DVC-tracked)
│   ├── tissue/  index/  ckpt/  heatmaps/  overlays/  reports/
└── Makefile
```

---

## 9. Failure Domains and Their Containment

| Domain | Blast radius | Containment |
|---|---|---|
| Corrupt / unreadable slide | 1 slide | `SlideReader` retries once, then logs to `artifacts/index/failed_slides.csv` and continues; the index is built from survivors |
| Tissue mask collapse (all-white or all-tissue) | 1 slide | Post-condition A2 raises; slide is quarantined, not silently indexed as empty |
| Annotation/image misregistration | 1 slide, poisons training | V3 gate: rendered overlay of mask on thumbnail must be visually approved and Dice-vs-reference checked on 3 known slides |
| OOM during inference | 1 slide | Batch size halves and retries once; then falls back to stride 512 |
| Stitcher coverage hole | 1 heatmap | `den < 1e-6` inside tissue → hard assert before the heatmap is written. Edge-flush patches are anchored to the far edge of the heatmap, or `ceil(H0/d)` sizing leaves the last row unwritten |
| Bad checkpoint | all downstream heatmaps | Heatmap `.zattrs` records `model_run_id` + git SHA; `rollback.md` R5 invalidates by SHA |

---

## 10. What This Architecture Deliberately Does Not Do

Stated so that reviewers do not read omissions as oversights.

- **No multiple-instance learning (MIL) head.** The stated objective is
  pixel-wise segmentation with a spatial heatmap, so dense supervision is used
  where masks exist. A CLAM/ABMIL slide-level head is a natural extension but
  would change the supervision contract entirely.
- **No multi-scale context fusion.** A 512² patch at 0.5 µm/px is a 256 µm
  field of view, which is adequate for metastatic nests but blind to
  architectural context. Listed as a planned ablation, not a v1 requirement.
- **No test-time stain normalisation by default.** HED augmentation at training
  time is used instead (ADR-008); Macenko remains available behind a flag for
  cross-centre CAMELYON17 evaluation.
- **No distributed training.** The dataset fits a single-GPU schedule in
  ~14 h. DDP is a config change, not an architectural one.
