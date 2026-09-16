# wsi-metastasis-seg

Gigapixel whole-slide-image tumor segmentation for H&E lymph node sections
(CAMELYON16/17 format). Reads pyramidal WSIs, filters non-tissue background,
extracts manageable patches, trains a segmentation network, and reconstructs
a slide-level spatial probability heatmap.

## Quick start

**Install into a dedicated environment.** This project requires numpy >= 2, and
installing it into a shared `base` conda environment will upgrade numpy
underneath anything already there. Packages with compiled numpy extensions
(numba, gensim, contourpy/matplotlib, older imagecodecs) break on that upgrade.

```bash
conda create -n wsi python=3.11 -y      # or: python -m venv .venv
conda activate wsi

pip install -e ".[dev]"                  # add ".[train]" for torch + smp
make fixtures                            # synthetic test slides, ~15 MB
make verify                              # gates V0-V6
```

If `make fixtures` reports that JPEG tile encoding is unavailable, it falls
back to deflate automatically and the fixtures remain valid — just larger.
That message means numpy and imagecodecs are ABI-mismatched in your
environment; `pip install -U imagecodecs` restores the JPEG path.

## Version control

A `.gitignore` ships with the project. It excludes `data/` (the slide archive)
and `artifacts/` (everything regenerable), so `git add -A` picks up code only.
Verify with `git status --short` before the first commit: if you see `.tif`
files listed, the ignore file is not being applied.

## Updating

```bash
bash update.sh          # newest wsi-metastasis-seg*.tar.gz in ~/Downloads
python scripts/version.py
```

Use `bash update.sh`, not `./update.sh` — the execute bit does not survive
every download path. `version.py` prints a tree hash covering src, scripts and
configs; compare it against the hash quoted with the release.

## No GPU on this machine?

See `docs/COLAB.md`. Export a ~4 GB patch set, train on Colab or Kaggle, and
pull the test slides straight from S3 inside the notebook rather than
uploading them.

## Pipeline

```
01_build_tissue_masks  ->  artifacts/tissue/*.png + slides.parquet
02_build_patch_index   ->  artifacts/index/patches.parquet
03_train               ->  artifacts/ckpt/{run_id}/best.pt
04_infer_slide         ->  artifacts/heatmaps/{slide}.zarr + overlay
05_evaluate            ->  artifacts/reports/{run_id}/metrics.json
```

Stages communicate only through files and are independently resumable.

## The one thing to know

Nothing at level-0 resolution is ever materialised in full. The unit of data
moving between modules is a `(slide_id, x0, y0, level)` tuple in **level-0
coordinates**; pixels are read lazily and discarded. A 400-slide patch index
is ~120 MB instead of the ~2 TB a pre-extracted patch dump would cost.

## Documentation

| File | Read it when |
|---|---|
| `docs/Architecture.md` | you need the block diagram, data contracts, or budgets |
| `docs/decision.md` | you want to know why something is the way it is (16 ADRs) |
| `docs/flow.md` | you need to trace where a value comes from |
| `docs/bug.md` | something is broken, or you're about to write a guard |
| `docs/verification_checklist.md` | you're about to call a module done |
| `docs/rollback.md` | something is broken and you need to get back to good |

## Status

Implemented and tested: slide IO, geometry, annotations, tissue localisation,
patch indexing, sliding-window planning, Gaussian stitching, post-processing,
FROC/AUC evaluation. 29 tests pass; stages 01 and 02 run end-to-end on the
fixtures.

Written but not executed here (no GPU/torch in the dev container): the
training loop, model registry, losses, and stages 03-05. These compile and
follow the documented contracts, but have not been run — gates V6.2
(overfit-one-batch) and V7.x are the first things to run on a GPU box.
