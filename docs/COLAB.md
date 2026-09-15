# Training on Colab or Kaggle

For when the GPU is somewhere the 230 GB of slides is not.

The pipeline normally reads pixels lazily from the slide archive every epoch
(ADR-005), which makes that archive a runtime dependency of training. That is
the right design on a machine holding the archive and impossible on a hosted
notebook. This document describes the workaround and is honest about what it
costs.

---

## The shape of the plan

```
  LOCAL (your PC, no GPU)                  REMOTE (Colab / Kaggle, GPU)
  ───────────────────────                  ────────────────────────────
  stage 01  tissue masks
  stage 02  patch index
       │
  stage 07  export patchset  ── ~4 GB ──►  extract shards
                                           stage 03  TRAIN
       ◄───── best.pt (~75 MB) ──────────────────┘
                                           (test slides pulled from S3
  stage 04/05 on CPU, slowly                directly inside the notebook)
  — or do them remotely too ───────────►   stage 04  infer
                                           stage 05  evaluate
```

Two separable problems: **training** needs many small patches, which travel
well. **Inference** needs whole slides, which do not — but the notebook can
download the handful of test slides straight from the public S3 bucket, which
is far faster than uploading them.

---

## What the export gives up

Stated plainly, because it changes what the numbers mean.

| Kept | Given up |
|---|---|
| Class balance per split | **Per-epoch coordinate jitter.** Normally each patch is nudged by up to ±128 px every epoch so the model never sees an identical crop twice. Exported crops are fixed. |
| Split assignment (patient-level) | **Hard-negative mining over the full pool.** Only the exported normals exist, so mining is disabled rather than silently run on a subset. |
| Physical scale resolution per slide | **Re-indexing at a different patch size** without re-exporting. |
| Every label, exactly | |
| Geometric + photometric augmentation | |

The jitter loss is the one that matters. Expect slightly more overfitting than
a slide-backed run, and treat a patchset-trained model as a strong first
result rather than the final one.

---

## "I have 4.5 TB of Drive — can I just upload the slides?"

Storage is not the constraint. Two other things are.

**Upload time.** 230 GB at 50 Mbps up is about 10 hours; at 20 Mbps, over a
day.

**Random access over a network mount.** `read_region` pulls a handful of
256×256 tiles from scattered byte offsets inside a 1.8 GB file. Drive is a
FUSE mount designed for streaming whole files, not for seeking inside them.
Local NVMe does this in ~12 ms; expect an order of magnitude or two worse over
Drive, against a GPU that wants ~36 patches/s. Sustained heavy Drive IO can
also trip quota throttling mid-session.

Do not take that on faith — measure it:

```bash
python scripts/bench_read.py --slide /content/drive/MyDrive/wsi/tumor_016.tif
```

It prints latency percentiles and an explicit verdict: GPU-bound, marginal, or
IO-bound. If it says comfortably GPU-bound, ignore everything above and train
off the mount.

**What the space is genuinely worth** is not the slides. It is exporting the
**full** patch index instead of a balanced subset — see the next section.

## Choosing an export size

Drive capacity is almost never the binding constraint. **Colab's local disk
is** — roughly 107 GB on the free tier, shared with the OS and with the
shards you extract from. Plan around ~40 GB of extracted patches, not around
Drive.

| Export | Patches | Size | When |
|---|---|---|---|
| `--normal-per-tumor 3` | ~43k | ~3.3 GB | Minimum viable. Fits anywhere. |
| `--normal-per-tumor 6` | ~75k | ~5.8 GB | More negative variety; sampler repeats less. |
| `--normal-per-tumor 6 --jitter-copies 3` | ~220k | ~17 GB | **Recommended if you have the space.** |
| `--all --jitter-copies 3` | ~760k | ~60 GB | Only with Colab Pro; extraction is the limit. |

`--jitter-copies K` bakes K distinct random offsets per tumor anchor at export
time. This directly recovers most of what an export gives up: instead of one
frozen crop per patch, the model sees K of them. Not unlimited like the
slide-backed path, but a large improvement for a linear cost in disk.

### Why not just put the slides on Drive?

With terabytes of Drive space this is the obvious question, and the answer is
about **access pattern**, not capacity.

Colab mounts Drive over FUSE. The pipeline reads a 512x512 window from a
multi-gigabyte tiled TIFF, which means seeking to a handful of scattered tile
offsets and decoding them — thousands of times per minute:

```
local NVMe    ~12 ms/patch   ->  ~660 patches/s with 8 workers
Drive FUSE    0.5-5 s/patch  ->  roughly 1-5 patches/s
GPU needs                        ~36 patches/s to stay busy at batch 16
```

The GPU would idle 90-99% of the time, and sustained random access through
the Drive mount triggers rate limiting (HTTP 403). Copying the slides to local
disk first does not help either: 230 GB does not fit in ~107 GB, and a partial
copy would have to be redone every session.

The patch export exists precisely because small sequential files are what a
hosted notebook is good at.

Drive space is still worth using — for an off-machine backup of the raw
archive, for checkpoints, and for heatmaps and reports coming back.

## Step 1 — Export locally

### With ample Drive space, export everything

```bash
python scripts/07_export_patchset.py --plan --all --jitter-copies 2
```

~254,000 patches plus jittered tumor copies, roughly **24 GB**. This is the
better export, and not only because it is bigger:

| | balanced subset (~4 GB) | full index (~24 GB) |
|---|---|---|
| Sampler | draws from an already-sampled pool | behaves exactly as against the slide archive |
| Hard negative mining | disabled — nothing to mine | **works**, over all 237,000 negatives |
| Per-epoch jitter | lost | partly recovered via `--jitter-copies` |

`--jitter-copies N` writes N extra randomly-offset crops of each **tumor**
patch. Jitter matters most where data is scarcest, and tumor patches are ~6%
of the index, so copying those is cheap; copying negatives would multiply the
export for almost no benefit.

With the full export, re-enable mining by dropping `configs/patchset.yaml`
from the config list, or with
`--set train.hard_negative_start_epoch=8`.

Colab's local disk is ~107 GB, so 24 GB extracts with plenty of room for
checkpoints.

### Or the compact version

```bash
python scripts/07_export_patchset.py --plan --normal-per-tumor 3
```

For the 131-slide cohort this selects roughly:

```
train    ~43,700  tumor ~10,600 (24%)   slides  43
val       ~4,700  tumor  ~1,150 (24%)   slides   9
estimated size ~3.7 GB at JPEG q90
```

Tune before exporting, not after:

- `--normal-per-tumor 2` → ~33,000 patches, ~2.8 GB
- `--normal-per-tumor 4` → ~55,000 patches, ~4.6 GB
- `--max-train 60000` caps the total, scaling both classes rather than
  truncating one
- `--quality 85` cuts roughly 20% of the size; visually harmless on H&E, but
  it is a real (small) information loss on a dataset already stored as JPEG

Then:

```bash
python scripts/07_export_patchset.py --export --out artifacts/patchset \
    --normal-per-tumor 6 --jitter-copies 3 --workers 8
```

Around 20–40 minutes. Output:

```
artifacts/patchset/
├── shards/shard_0000.tar ...     # ~2000 patches each
├── manifest.parquet              # key, slide_id, split, tumor_frac
└── export_info.json              # index hash, seed, quality, patch size
```

`export_info.json` records the index hash the export came from, so a
checkpoint can be traced back to the exact patch index.

---

## Step 2 — Upload

**Google Drive** (free tier is 15 GB, so ~4 GB fits):

```bash
# also upload these small files -- needed for inference later
tar czf artifacts/meta.tar.gz artifacts/index/slides.parquet \
        artifacts/index/patches.parquet artifacts/tissue
```

Upload `artifacts/patchset/` and `meta.tar.gz` to a Drive folder called
`wsi/`. Also push your repository to GitHub — the notebook clones it rather
than uploading code.

**Kaggle** is the better option if you have it: datasets can be far larger
than Drive's free tier, sessions run up to 12 hours, and the weekly GPU quota
(~30 h) is more generous than free Colab. Upload `artifacts/patchset/` as a
private Kaggle Dataset; everything below works the same with paths changed.

---

## Step 2b — Dry run first (seconds)

Before anything long, on either machine:

```bash
python scripts/03_train.py \
  --config configs/base.yaml configs/data_camelyon16.yaml \
           configs/model_unet_effb0.yaml configs/patchset.yaml \
  --set data.patchset_root=/content/patchset --dry-run
```

One forward and backward pass, then exit. Confirms shapes match, the loss is
finite, gradients actually flow, and the mask is binary. Seconds, and it has
already caught one construction-order bug.

## Step 3 — Train in the notebook

`notebooks/colab_train.ipynb` in this repository does all of the following.
The substance:

```python
# 1. GPU check -- stop if there isn't one
!nvidia-smi

# 2. Code and dependencies
!git clone https://github.com/<you>/wsi-metastasis-seg.git
%cd wsi-metastasis-seg
!pip install -q -e ".[dev]" segmentation-models-pytorch
!python scripts/version.py

# 3. Data: extract shards to LOCAL disk, never train off Drive directly
from google.colab import drive; drive.mount('/content/drive')
!mkdir -p /content/patchset/files
!cp -r /content/drive/MyDrive/wsi/patchset/manifest.parquet /content/patchset/
!for t in /content/drive/MyDrive/wsi/patchset/shards/*.tar; do \
     tar xf "$t" -C /content/patchset/files; done
!ls /content/patchset/files | wc -l

# 4. Train, with checkpoints written to Drive so a disconnect is survivable
!python scripts/03_train.py \
    --config configs/base.yaml configs/data_camelyon16.yaml \
             configs/model_unet_effb0.yaml configs/patchset.yaml \
    --set data.patchset_root=/content/patchset \
    --set paths.ckpt=/content/drive/MyDrive/wsi/ckpt \
    --set train.max_epochs=40
```

**Extract to `/content`, not Drive.** Drive is mounted over the network;
reading 44,000 small files through it will make the GPU wait on IO all
session. Local disk turns a bottleneck into a non-issue.

### Session limits are the real constraint

Free Colab gives roughly 12 hours and a T4. At ~44,000 patches and batch 16,
an epoch is about 2,750 steps ≈ 20 minutes, so 40 epochs is ~14 hours — more
than one session.

Plan for two or three sessions:

```python
# subsequent sessions
!python scripts/03_train.py ... \
    --resume /content/drive/MyDrive/wsi/ckpt/<run_id>/last.pt
```

Resume restores optimiser state, scheduler position and epoch. Gate V7.3
covers it, though that gate has never been run — verify the loss continues
from where it stopped rather than jumping.

If sessions keep dropping, `--set train.max_epochs=20` is a reasonable first
target. Early stopping has patience 8, so a run that is still improving at 20
was going to use most of the 40 anyway.

---

## Step 4 — Inference and evaluation

Whole slides are needed here, but only for the **9 test slides** — and the
notebook can fetch them from the public bucket directly, which is much faster
than uploading.

```python
import pandas as pd
!tar xzf /content/drive/MyDrive/wsi/meta.tar.gz -C /content/
sl = pd.read_parquet('/content/artifacts/index/slides.parquet')
test = sl[sl.split == 'test'].slide_id.tolist()
print(len(test), 'test slides')

!pip install -q awscli
for s in test:
    !aws s3 cp --no-sign-request --quiet \
        s3://camelyon-dataset/CAMELYON16/images/{s}.tif /content/slides/{s}.tif
    !aws s3 cp --no-sign-request --quiet \
        s3://camelyon-dataset/CAMELYON16/annotations/{s}.xml /content/slides/ || true
```

Roughly 16 GB and 10–20 minutes on Colab's connection. Colab's local disk
(~100 GB) holds it comfortably. Then:

```python
ckpt = '/content/drive/MyDrive/wsi/ckpt/<run_id>/best.pt'
for s in test:
    !python scripts/04_infer_slide.py --ckpt {ckpt} \
        --slide /content/slides/{s}.tif \
        --set paths.tissue=/content/artifacts/tissue \
        --set paths.artifacts=/content/artifacts

!python scripts/05_evaluate.py --run-id <run_id> --split test \
    --set paths.artifacts=/content/artifacts
```

Then copy `artifacts/reports/` and `artifacts/heatmaps/` back to Drive —
heatmaps are ~40 MB each compressed, so all nine fit easily.

---

## Doing inference locally instead

Possible, but slow. CPU inference at stride 256 is roughly 4 hours per slide;
`--set infer.stride=512` cuts that to about an hour by removing the overlap —
at the cost of the seam suppression that ADR-004 exists to provide. Fine for
eyeballing one heatmap, not for a number you intend to report.

Prefer the remote route.

---

## Checklist

```
[ ] scripts/version.py agrees locally and in the notebook
[ ] export_info.json index_hash matches your local patches.parquet
[ ] notebook nvidia-smi shows a GPU before anything else runs
[ ] shards extracted to /content, not read from Drive
[ ] checkpoints written to Drive, not /content
[ ] after training: best.pt contains a fitted threshold (gate V7.2)
[ ] test slides downloaded from S3, not uploaded
[ ] report says the model was trained on an exported patchset
    (no coordinate jitter, no hard-negative mining)
```

That last line belongs in the write-up. It is a real difference from the
documented training procedure, and a reader comparing against published
CAMELYON16 numbers deserves to know.
