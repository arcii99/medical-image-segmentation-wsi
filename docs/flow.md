# Execution Flow Map — `wsi-metastasis-seg`

How control and data travel between files. Read this when you need to answer
"where does this value come from?" or "what breaks if I change this function?"

Notation:
- `→` synchronous call
- `⇢` data written to disk and read later by another stage
- `⟳` loop
- `⊗` process/thread boundary

---

## 0. Stage Overview

```
scripts/01_build_tissue_masks.py   ⇢  artifacts/tissue/{slide_id}.png + slides.parquet
scripts/02_build_patch_index.py    ⇢  artifacts/index/patches.parquet
scripts/03_train.py                ⇢  artifacts/ckpt/{run_id}/best.pt
scripts/04_infer_slide.py          ⇢  artifacts/heatmaps/{slide_id}.zarr + grid.parquet
scripts/05_evaluate.py             ⇢  artifacts/reports/{run_id}/
```

Stages are **independently resumable** and communicate only through files. No
stage imports another stage. This is what makes partial reruns cheap and makes
the rollback procedures in `rollback.md` tractable.

---

## 1. Stage 01 — Tissue Localisation

**Entry:** `scripts/01_build_tissue_masks.py::main`

```
main(cfg)
│
├─→ utils.config.load(["configs/base.yaml", "configs/data_camelyon16.yaml"], overrides)
│       ⇢ returns frozen ResolvedConfig
│
├─→ discover_slides(cfg.paths.raw_dir)            # glob *.tif *.svs
│
└─⟳ for slide_path in slides:            ⊗ multiprocessing.Pool(n=cfg.hw.cpu_workers)
    │
    └─→ _process_one(slide_path, cfg)
        │
        ├─→ io.slide.open_slide(path, backend=cfg.io.backend)
        │   │   src/io/slide.py
        │   ├─→ _try_tiffslide(path)                       # ADR-001 chain
        │   ├─→ _try_openslide(path)      (on failure)
        │   └─→ SlideReader(handle, meta=SlideMeta(...))
        │           └─→ _resolve_mpp(handle)               # vendor props → TIFF tags → raise
        │
        ├─→ reader.level_for_mpp(8.0)  -> lt               # ADR-002 machinery, reused
        │
        ├─→ reader.read_region(0, 0, lt, *reader.level_dims[lt])
        │       └─→ _rgba_to_rgb_on_white(arr)             # BUG-001
        │
        ├─→ preprocess.tissue.tissue_mask(thumb, mpp_t, cfg.tissue)
        │   │   src/preprocess/tissue.py
        │   ├─→ cv2.medianBlur(k=7)
        │   ├─→ _otsu_saturation(hsv)      ─┐
        │   ├─→ _otsu_inverted_gray(gray)  ─┼→ union
        │   ├─→ _pen_and_artifact_mask(hsv) ┘ subtract         # ADR-003
        │   ├─→ _bimodality_guard(hist)                        # raises TissueDetectionError
        │   └─→ morphology: close(5) → open(3) → remove_small → fill_holes
        │
        ├─→ _assert_contract_A2(mask)          # 0.01 ≤ mean ≤ 0.90
        │
        └─⇢ write artifacts/tissue/{slide_id}.png       (uint8 {0,255}, at level lt)
           ⇢ append row to artifacts/index/slides.parquet:
              slide_id, path, backend, mpp0, level_w, residual_scale,
              level_t, thumb_shape, tissue_frac, patient_id, centre, split
```

**Failure path:** any exception → row appended to
`artifacts/index/failed_slides.csv` with the traceback; the pool continues. A
non-empty failed list is a **warning at stage 01, an error at stage 02** (you
may tolerate losing slides; you may not tolerate silently training on a
subset you didn't intend).

**Split assignment** happens here, once, in
`utils.splits.assign(patient_id, seed)` — ADR-010. Never recomputed downstream.

---

## 2. Stage 02 — Patch Index Construction

**Entry:** `scripts/02_build_patch_index.py::main`

```
main(cfg)
│
├─→ slides = pd.read_parquet("artifacts/index/slides.parquet")
├─→ _fail_if_failed_slides_present()                 # see above
│
└─⟳ for row in slides:                    ⊗ Pool
    │
    └─→ _index_one(row, cfg)
        │
        ├─→ tissue = imread(artifacts/tissue/{slide_id}.png) > 0
        │
        ├─→ geom = io.annotations.load(xml_path, slide_meta)      # None for normal slides
        │   │   src/io/annotations.py
        │   ├─→ _parse_camelyon_xml(path)
        │   │       Annotation groups: _0/_1 = tumor, _2 = exclusion   # BUG-008
        │   ├─→ tumor = unary_union(polys_0_1)
        │   ├─→ excl  = unary_union(polys_2)
        │   ├─→ geom  = tumor.difference(excl)
        │   ├─→ geom  = geom.buffer(0)                   # repair self-intersections
        │   └─→ STRtree(geom.geoms) cached on the object  # O(log n) window queries
        │
        ├─→ preprocess.patching.grid(slide_meta, level_w, size=512, stride=512)
        │       yields (x0_L0, y0_L0) anchors
        │
        └─⟳ for (x0, y0) in grid:
            │
            ├─→ tissue_frac = _mean_in_window(tissue, x0, y0, ...)   # thumbnail coords
            ├─  if tissue_frac < cfg.patching.min_tissue (0.10): continue
            │
            ├─→ tumor_frac = annotations.area_fraction(geom, x0, y0, level_w, 512)
            │       └─→ STRtree query → clip polygons to window → area / (512²·s²)
            │           (vector area, NOT rasterised — 40× faster)
            │
            └─  emit row
        │
        └─⇢ artifacts/index/patches.parquet  (partitioned by split, zstd)
```

**Why the tumor fraction is computed by polygon area rather than rasterisation:**
raster would need a 512² fill per window (3 M windows → ~45 min); Shapely
clipping against an STRtree is ~1.2 ms/window (~6 min) and exact.

**Post-conditions checked in-script (gate V4):**
- `patches.groupby("split").slide_id.nunique()` — no slide in two splits.
- `patches[patches.tumor_frac > 0].slide_id.nunique()` ≥ expected tumor slides.
- centre distribution per split within ±10 % (ADR-010).

---

## 3. Stage 03 — Training

**Entry:** `scripts/03_train.py::main`

### 3.1 Setup path

```
main(cfg)
├─→ utils.seed.seed_everything(cfg.seed)
├─→ utils.logging.setup(run_dir)                      # stdout + JSONL + tensorboard
├─→ run_dir = artifacts/ckpt/{date}_{git_sha}_{cfg_hash}/     # ADR-014
├─→ _snapshot_environment(run_dir)                    # config, pip freeze, git diff
│
├─→ index = data.index.load("artifacts/index/patches.parquet")
│
├─→ train_ds = data.dataset.PatchDataset(index[split=="train"], cfg, train=True)
├─→ val_ds   = data.dataset.PatchDataset(index[split=="val"],   cfg, train=False)
│
├─→ sampler  = data.sampler.BalancedSampler(index[split=="train"], ratio=(1,3), cfg)
│
├─→ train_dl = DataLoader(train_ds, sampler=sampler, batch_size=16,
│                         num_workers=8, worker_init_fn=data.dataset.worker_init,
│                         persistent_workers=True, pin_memory=True,
│                         prefetch_factor=4)
│
├─→ model = models.registry.build(cfg.model)          # "unet_effb0" → smp.Unet(...)
├─→ crit  = train.losses.BCEDice(w_bce=0.5, w_dice=0.5, pos_weight=2.0)
├─→ opt   = AdamW(param_groups=[encoder@3e-5, decoder@3e-4], wd=1e-4)
├─→ sched = train.schedule.cosine_with_warmup(opt, warmup=3, total=40)
└─→ ema   = ModelEMA(model, decay=0.999)
```

### 3.2 The hot loop

```
⟳ for epoch in range(40):
  │
  ├─  if epoch == 2: model.encoder.requires_grad_(True)           # ADR-007
  ├─  if epoch >= 8 and epoch % 4 == 0:
  │      └─→ sampler.refresh_hard_negatives(model, index, cfg)    # ADR-006
  │              └─→ inference over a normal-slide subset ⊗ (no_grad)
  │
  ├─⟳ for batch in train_dl:
  │   │
  │   │   ⊗ ───────────── worker process boundary ─────────────
  │   │   PatchDataset.__getitem__(i)
  │   │   │   src/data/dataset.py
  │   │   ├─→ row = self.index.iloc[i]
  │   │   ├─→ reader = self._reader(row.slide_id)        # lazy, PID-keyed  # BUG-002
  │   │   ├─→ jitter = rng.integers(-128, 129, 2) if train else (0,0)   # ADR-004
  │   │   ├─→ img  = reader.read_region(x0+jx, y0+jy, level_w, 512, 512)
  │   │   ├─→ mask = annotations.rasterize(geom, x0+jx, y0+jy, level_w, 512, 512)
  │   │   ├─→ if residual_scale != 1: cv2.resize(img, INTER_AREA)
  │   │   │                            cv2.resize(mask, INTER_NEAREST)
  │   │   ├─→ transforms.train_aug(image=img, mask=mask)
  │   │   │      D4 flips/rot90        → image + mask
  │   │   │      HED jitter            → image only          # BUG-011
  │   │   │      blur / JPEG artifact  → image only
  │   │   │      Normalize(ImageNet)   → image only
  │   │   └─  return {"image": f32[3,512,512], "mask": f32[1,512,512], "meta": {...}}
  │   │   ⊗ ──────────────────────────────────────────────────
  │   │
  │   ├─→ x, y = batch["image"].cuda(non_blocking=True), batch["mask"].cuda(...)
  │   ├─→ with autocast(bfloat16):  logits = model(x);  loss = crit(logits, y)
  │   ├─→ loss.backward();  clip_grad_norm_(1.0);  opt.step();  opt.zero_grad()
  │   ├─→ ema.update(model)
  │   └─→ logging.log_step(loss, lr, throughput)     every cfg.log_every steps
  │
  ├─→ sched.step()
  │
  ├─→ val_metrics = train.loop.validate(model, val_dl)
  │   │   accumulates confusion counts on GPU, no per-batch python floats
  │   └─→ train.metrics.dice_iou(tp, fp, fn)  +  sweeps τ ∈ [0.1, 0.9] step 0.05
  │
  ├─→ ema_metrics = train.loop.validate(ema.module, val_dl)
  │
  ├─⇢ save last.pt every epoch;  best.pt when val Dice improves
  │      payload: {state_dict, ema_state_dict, optimizer, epoch,
  │                threshold τ*, resolved_config, git_sha, dirty,
  │                pip_freeze, index_hash, seed}            # ADR-014
  │
  └─  early stop if no improvement for 8 epochs
```

**Where a value comes from — the τ trail.** `τ` is swept in
`train.metrics` at validation → stored in `best.pt["threshold"]` → read by
`infer.postproc.threshold_from_ckpt()` → written into the heatmap `.zattrs` →
read by `eval.report`. Five hops, one source of truth. Any of them defaulting
to `0.5` on its own is the bug described in ADR-006 Consequences.

---

## 4. Stage 04 — Inference and Reconstruction

**Entry:** `scripts/04_infer_slide.py::main`

```
main(cfg, slide_id, ckpt_path)
│
├─→ ckpt   = torch.load(ckpt_path)
├─→ model  = models.registry.build(ckpt["resolved_config"].model)   # config from ckpt,
│                                                                   # NOT from CLI
├─→ model.load_state_dict(ckpt["ema_state_dict"]); model.eval().cuda()
├─→ τ      = ckpt["threshold"]
│
├─→ reader = io.slide.open_slide(path)
├─→ tissue = imread(artifacts/tissue/{slide_id}.png) > 0      # reuse stage 01
│
├─→ planner = infer.sliding_window.Planner(reader, tissue, size=512, stride=256)
│   │   src/infer/sliding_window.py
│   ├─→ positions = grid ∩ tissue                              # ~20k of ~72k
│   └─→ order = _chunk_aligned_order(positions, chunk=1024)    # ADR-009 write locality
│
├─→ stitcher = infer.stitch.GaussianStitcher(
│                   out_shape = level0_shape // 8,
│                   downsample = 8, sigma = 64,
│                   store = zarr.open("artifacts/heatmaps/{slide}.zarr", "w"))
│   │   src/infer/stitch.py
│   └─→ allocates num[f32], den[f32], chunks (1024,1024)
│
├─⟳ for coords_batch in planner.batches(32):
│   │
│   │   ⊗ prefetch thread: reads next batch's regions while GPU works
│   ├─→ imgs = stack([reader.read_region(x, y, level_w, 512, 512) for ...])
│   ├─→ imgs = normalize(imgs).cuda()
│   ├─→ with no_grad(), autocast(bf16):
│   │       probs = sigmoid(model(imgs))                 # [32,1,512,512]
│   │       if cfg.tta: probs = mean over D4 group
│   └─→ stitcher.add(probs.float().cpu().numpy(), coords_batch)
│           └─→ downsample 8× (area) → num += p·w ; den += w
│
├─→ heat = stitcher.finalize()
│   ├─→ _assert_coverage(den, tissue)        # den < 0.05 inside tissue → raise
│   ├─→ heat = (num / (den + 1e-8)).astype(float16)
│   ├─⇢ write heat to zarr, delete num/den
│   └─⇢ write .zattrs {mpp, downsample, level0_shape, stride, sigma,
│                      model_run_id, git_sha, threshold τ, tta}
│
├─→ grid_df = planner.grid_summary(probs_cache)    # per-window mean/max
│   └─⇢ artifacts/heatmaps/{slide_id}_grid.parquet
│
├─→ binary = infer.postproc.apply(heat, τ, mpp=2.0, cfg.postproc)    # ADR-012
│   └─  fill_holes → remove CC < 0.02mm² → opening(10µm)
│
└─→ infer.heatmap.render_overlay(reader, heat, binary, out=artifacts/overlays/...)
        └─→ jet colormap on heat, alpha 0.45 over the level-lt thumbnail,
            written as pyramidal OME-TIFF
```

**The one place the model config must not come from the CLI:** line 3. Building
the architecture from the current config file while loading weights from an old
checkpoint produces either a loud `state_dict` mismatch (good) or a silent
partial load if `strict=False` was ever set (catastrophic). `strict=True` is
non-negotiable here.

---

## 5. Stage 05 — Evaluation

**Entry:** `scripts/05_evaluate.py::main`

```
main(cfg, run_id, split="test")
│
├─→ _refuse_if_dirty(ckpt)                                   # ADR-014
├─→ slides = slides.parquet[split == "test"]
│
├─⟳ for slide in slides:
│   ├─→ heat = zarr.open(artifacts/heatmaps/{slide}.zarr)["heat"]
│   ├─→ _assert_attrs_match(heat.attrs, run_id)      # heatmap ↔ checkpoint pairing
│   ├─→ pred_cc = infer.postproc.apply(heat, τ, ...)
│   ├─→ gt      = io.annotations.load(xml).geoms      # lesion-level geometry
│   │
│   ├─→ eval.slide_metrics.pixel(pred_cc, gt, tissue)  → dice, iou
│   ├─→ eval.froc.match(pred_cc, gt)                   → hits, fps, scores
│   └─→ slide_score = heat[tissue].max()               → for AUC
│
├─→ eval.froc.curve(all_hits, all_fps, n_slides)
│       → sensitivity @ {0.25,0.5,1,2,4,8} FP/slide     # ADR-013 primary metric
├─→ roc_auc_score(labels, slide_scores)
├─→ eval.report.bootstrap(..., n=1000, unit="slide")   → 95% CIs
│
└─⇢ artifacts/reports/{run_id}/
       metrics.json · froc.png · roc.png · per_slide.csv
       config_snapshot.yaml · failure_gallery/   (worst 10 slides, overlay PNGs)
```

---

## 6. Cross-Cutting Call Graph

Who calls `SlideReader.read_region`, the function most likely to be subtly wrong:

```
io.slide.SlideReader.read_region
   ├── preprocess.tissue                (stage 01, thumbnail, 1×/slide)
   ├── data.dataset.PatchDataset        (stage 03, 512², ~10⁶×/epoch)  ⊗ workers
   ├── infer.sliding_window.Planner     (stage 04, 512², ~2×10⁴/slide) ⊗ prefetch thread
   └── infer.heatmap.render_overlay     (stage 04, thumbnail, 1×/slide)
```

Three of four call sites are in a different process or thread from where the
`SlideReader` was constructed. This is why the object is **not** picklable by
design: `__getstate__` drops the handle and `__setstate__` reopens lazily.
Making it naively picklable is the direct cause of BUG-002.

Who reads `mpp`:

```
slide.meta.mpp_x
   ├── slide.level_for_mpp        → working level        (ADR-002)
   ├── preprocess.tissue          → morphology radii in µm (ADR-003)
   ├── preprocess.patching        → nothing (coords are px)
   ├── infer.stitch               → heatmap .zattrs["mpp"]  (ADR-009)
   └── infer.postproc             → min-area in mm² → px    (ADR-012)
```

Changing the target MPP therefore changes morphology radii and post-processing
thresholds automatically — provided nobody hard-codes a pixel constant. That
invariant is checked by a grep gate in `verification_checklist.md` V0.5.

---

## 7. Configuration Resolution Order

```
configs/base.yaml
    ↓ merged under
configs/data_camelyon16.yaml
    ↓ merged under
configs/model_unet_effb0.yaml
    ↓ merged under
CLI --set key=value overrides
    ↓
utils.config.freeze()  →  ResolvedConfig (immutable, hashable)
    ↓
sha1(canonical_json)[:8]  →  cfg_hash  →  run directory name
```

`ResolvedConfig` is passed by value into every stage. **No module reads a YAML
file on its own**, and no module reads environment variables except
`utils.config` (for `WSI_DATA_ROOT`). This keeps the hash honest.

---

## 8. Error Propagation Policy

| Layer | On error | Rationale |
|---|---|---|
| `io.slide` | Retry once, then raise `SlideReadError` | Transient NFS/IO blips are real |
| Stage 01/02 worker | Catch, log to `failed_slides.csv`, continue | One bad slide must not kill a 6-hour job |
| Stage 02 startup | **Abort** if `failed_slides.csv` is non-empty and unacknowledged | Prevents silently training on a subset |
| `data.dataset.__getitem__` | Raise. No silent black-patch substitution. | A returned zero patch is a poisoned label |
| Training step | OOM → halve batch, retry once, log; any other error → raise | OOM is recoverable, wrong math is not |
| `infer.stitch.finalize` | Coverage assert → raise before writing | A hole in a heatmap looks like a confident negative |
| Stage 05 | Refuse on dirty tree or attrs mismatch | Reported numbers must be attributable |

---

## 9. What Runs Where

| Stage | Device | Parallelism | Wall clock (CAMELYON16, 400 slides) |
|---|---|---|---|
| 01 tissue | CPU | `Pool(8)` | ~18 min |
| 02 index | CPU | `Pool(8)` | ~24 min |
| 03 train | 1× GPU | 8 DataLoader workers | ~14 h (40 epochs) |
| 04 infer | 1× GPU | 1 prefetch thread | ~2.5 min/slide → ~2 h for 50 test slides |
| 05 eval | CPU | serial | ~6 min |

Stage 01 and 02 are re-run on any change to ADR-002/003/004; stage 03 onward is
untouched by that as long as the index hash is updated in the checkpoint —
which is exactly why the hash is recorded.
