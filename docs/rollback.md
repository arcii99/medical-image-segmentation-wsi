# Rollback & Disaster Recovery — `wsi-metastasis-seg`

This project has three independently corruptible state layers. A rollback that
only addresses one of them produces a *worse* state than the failure did,
because the layers silently disagree.

| Layer | Lives in | Rolled back by |
|---|---|---|
| **Code** | git | `git revert` / `git reset` |
| **Data artifacts** (tissue masks, patch index, heatmaps) | `artifacts/`, DVC-tracked | `dvc checkout` at a git SHA |
| **Model checkpoints** | `artifacts/ckpt/`, content-addressed | selection, not reversion — checkpoints are immutable |

**Cardinal rule:** a code rollback that changes ADR-002, 003, 004, 009, or 012
invalidates data artifacts. Reverting code without re-running the affected
stage is the most dangerous state this project can be in — the pipeline runs,
produces numbers, and those numbers are meaningless. Procedure **R5** exists
for exactly this.

---

## 0. Preconditions (set up once)

```bash
# Branch protection: main is never committed to directly
git config branch.main.pushRemote no_push

# Every dependency pinned; the lockfile is part of the rollback surface
pip-compile requirements.in -o requirements.lock

# Data artifacts under DVC so a git SHA implies an artifact state
dvc init
dvc remote add -d store s3://<bucket>/wsi-metastasis-seg
dvc add artifacts/tissue artifacts/index artifacts/heatmaps
git add artifacts/*.dvc .dvc/config .gitignore
git commit -m "chore: track artifacts with dvc"

# Pre-commit hooks that make bad states harder to create
pre-commit install    # ruff, black, import-linter, nbstripout, no-large-files
```

**Tagging discipline.** Every state that produced a reported number is tagged:

```bash
git tag -a v0.3.0-froc0.78 -m "FROC avg 0.78, test split, ckpt 20260912_9fd3c11_a41be208" 
git push origin v0.3.0-froc0.78
```

The tag message carries the run directory name, so a result always has a
code coordinate and an artifact coordinate.

---

## R1 — Revert a merged change that is already on `main`

Use when the change is public and history must stay linear and auditable.
**Never `reset --hard` a shared branch.**

```bash
# 1. Identify the merge
git log --oneline --merges -12
# e.g.  8c1d20f Merge PR #47: switch tissue threshold to adaptive percentile

# 2. Confirm what it touched before reverting
git show --stat 8c1d20f
git diff 8c1d20f^1 8c1d20f -- src/ | head -100

# 3. Revert the merge, keeping mainline parent 1
git revert -m 1 8c1d20f
#    -m 1 == "keep the state of main"; omitting it errors out on a merge commit

# 4. If the revert conflicts
git status --short
# resolve, then
git revert --continue
#   or abandon cleanly:
git revert --abort
```

**Post-revert testing protocol (mandatory, in this order):**

```bash
pytest tests/unit -q                       # V0.3
lint-imports --config .importlinter         # V0.4
make verify                                 # V0–V6, ~12 min
```

Then decide whether artifacts are invalidated → **R5**.

Commit the revert with the reason, not just the SHA:
```bash
git commit --amend -m "Revert PR #47 (adaptive percentile tissue threshold)

Caused tissue_frac > 0.90 on 14/400 slides (V2.1 lower/upper bound gate).
Root cause under investigation in BUG-015. Reverting to Otsu union (ADR-003)
until the percentile variant has a guard for unimodal histograms.

This reverts merge commit 8c1d20f."
```

---

## R2 — Undo local, unpushed work

```bash
# Inspect first. Always.
git status --short --branch
git stash list
git log --oneline origin/main..HEAD

# (a) Discard uncommitted changes to one file
git restore --source=HEAD --staged --worktree src/preprocess/tissue.py

# (b) Discard ALL uncommitted changes but keep them recoverable
git stash push -u -m "wip: percentile threshold experiment $(date -Iseconds)"
#     recover later:  git stash pop 'stash@{0}'

# (c) Undo the last local commit, keep the changes staged
git reset --soft HEAD~1

# (d) Undo the last local commit and the changes
git reset --hard HEAD~1        # only if the commit was never pushed

# (e) Rewind to match origin exactly
git fetch origin && git reset --hard origin/main
```

`git stash push -u` over `git reset --hard` by default: untracked experiment
scripts are the thing people most regret destroying, and `-u` catches them.

---

## R3 — Recover from a "lost" commit or a bad rebase

`reflog` is the safety net. It survives `reset --hard`, a botched rebase, and
a deleted branch for 90 days.

```bash
git reflog --date=iso | head -40
# 9fd3c11 HEAD@{0}: reset: moving to origin/main
# a41be20 HEAD@{1}: commit: feat: gaussian stitcher
# 3ba77e1 HEAD@{2}: rebase (finish): returning to refs/heads/feat/stitch

# Recover the lost work onto a rescue branch — never straight onto main
git branch rescue/stitcher a41be20
git switch rescue/stitcher
pytest tests/unit -q
git switch main && git cherry-pick a41be20
```

Recover a deleted branch:
```bash
git reflog show --all | grep 'feat/stitch'
git branch feat/stitch <sha>
```

---

## R4 — Roll back a dependency (the silent-numerics case)

Library upgrades change numbers without changing code. `tiffslide`, `opencv`,
`shapely`, and `torch` have all shipped versions that alter pixel values,
polygon areas, or reduction order.

```bash
# 1. What actually changed?
git diff HEAD~1 -- requirements.lock

# 2. Restore the lockfile and the environment together
git checkout HEAD~1 -- requirements.lock
pip install -r requirements.lock --force-reinstall
pip check

# 3. Prove the numerics are unchanged, not just that imports work
pytest tests/unit -q
python scripts/qc_backend_parity.py --slides tests/fixtures/mini/*.tif --n-regions 20   # V1.3
python scripts/qc_annotations.py --area-parity --slide tests/fixtures/mini/synth_tumor.tif  # V3.4
pytest tests/integration/test_stitch_constant.py -q                                      # V8.3
```

**Pass criteria:** V1.3 `max|Δ| ≤ 1`, V3.4 rel.err ≤ 0.5 %, V8.3 exact.
If any fails after restoring the old lockfile, the problem is not the
dependency — stop and reopen the investigation.

The checkpoint stores `pip_freeze`, so the exact environment of any past result
is reconstructable:
```bash
python -c "
import torch;print('\n'.join(torch.load('artifacts/ckpt/<run>/best.pt',
  map_location='cpu')['pip_freeze']))" > /tmp/env.txt
python -m venv /tmp/repro && /tmp/repro/bin/pip install -r /tmp/env.txt
```

---

## R5 — Invalidate and rebuild data artifacts after a code rollback

**The most important procedure in this document.**

### Step 1 — Decide what the change invalidates

| Reverted change touches | Invalidates | Rebuild cost |
|---|---|---|
| `src/io/slide.py` (reads, MPP, alpha) | **everything** | full pipeline, ~17 h |
| `src/preprocess/tissue.py` (ADR-003) | tissue masks, index, heatmaps | 18 min + 24 min + infer |
| `src/preprocess/patching.py`, patch size/stride (ADR-004) | index, checkpoints | 24 min + 14 h |
| `src/io/annotations.py` (ADR-008 parsing) | index, checkpoints | 24 min + 14 h |
| `src/data/transforms.py`, `sampler.py` | checkpoints only | 14 h |
| `src/models/`, `src/train/` | checkpoints only | 14 h |
| `src/infer/stitch.py` (ADR-009) | heatmaps only | 2 h |
| `src/infer/postproc.py` (ADR-012) | reports only | 6 min |
| `src/eval/` | reports only | 6 min |

### Step 2 — Quarantine, do not delete

```bash
TS=$(date +%Y%m%dT%H%M%S)
mkdir -p artifacts/_quarantine/$TS
mv artifacts/index artifacts/heatmaps artifacts/reports artifacts/_quarantine/$TS/
echo "invalidated by revert of 8c1d20f (ADR-003 change)" \
  > artifacts/_quarantine/$TS/REASON.txt
```

Quarantine over deletion: if the revert turns out to be the mistake, the
artifacts are still there, and comparing old vs new artifacts is usually how
you find out which one was wrong.

### Step 3 — Restore the artifact state matching the target code SHA

```bash
git checkout <good-sha>
dvc checkout            # pulls artifacts recorded at that SHA
dvc status              # must print "Data and pipelines are up to date."
```

### Step 4 — Or rebuild from scratch

```bash
make pipeline STAGES=01,02          # tissue + index
make verify                         # V0–V6 before spending 14 GPU-hours
make pipeline STAGES=03,04,05
```

`make verify` between the cheap stages and the expensive one is not optional.
Discovering a V4.2 split leak after a 14-hour training run is a self-inflicted
wound.

### Step 5 — Invalidate downstream results by SHA

Because every heatmap's `.zattrs` records `git_sha` and `model_run_id`
(ADR-014), stale outputs are mechanically findable:

```bash
BAD_SHA=8c1d20f
python - <<'PY'
import zarr, pathlib, os, sys
bad = os.environ.get("BAD_SHA", "8c1d20f")
hits = []
for p in pathlib.Path("artifacts/heatmaps").glob("*.zarr"):
    try:
        sha = dict(zarr.open(p).attrs).get("git_sha", "")
    except Exception as e:
        print("UNREADABLE", p, e); continue
    if sha.startswith(bad):
        hits.append(p)
print(f"{len(hits)} heatmaps produced by {bad}:")
for h in hits: print(" ", h)
PY
```

Move the hits into quarantine, then re-run stage 04 for those slides only.

---

## R6 — Checkpoints are selected, never reverted

Checkpoints are immutable content. "Rolling back a model" means *pointing at a
different checkpoint*, which is a config change, not a git operation.

```bash
# What's available, and what produced it?
for d in artifacts/ckpt/*/; do
  python - "$d" <<'PY'
import sys, torch, json, pathlib
p = pathlib.Path(sys.argv[1]) / "best.pt"
if not p.exists(): sys.exit()
c = torch.load(p, map_location="cpu")
print(f"{p.parent.name:40s} sha={c['git_sha'][:7]}{'-dirty' if c['dirty'] else '':6s} "
      f"epoch={c['epoch']:3d} tau={c['threshold']:.2f} index={c['index_hash'][:8]}")
PY
done
```

```bash
# Promote a known-good checkpoint
echo "20260908_3ba77e1_c19f0d42" > artifacts/ckpt/LATEST
git add artifacts/ckpt/LATEST && git commit -m "revert: promote 3ba77e1 checkpoint (FROC 0.78)"
```

**Compatibility check before promoting** — the index hash must match the index
currently on disk, or the checkpoint's τ and its class prior refer to a
different dataset:

```bash
python scripts/qc_ckpt_compat.py --ckpt artifacts/ckpt/$(cat artifacts/ckpt/LATEST)/best.pt \
  --index artifacts/index/patches.parquet
# Expect: index_hash MATCH | config model section MATCH | VERDICT: COMPATIBLE
```

Never delete a checkpoint that has a git tag pointing at its run. Retention:
keep all tagged runs, keep `best.pt` for the last 10 untagged runs, prune
`last.pt` beyond 3 runs.

---

## R7 — Protocol for a risky change (prevention beats rollback)

A change is **risky** if it touches any of: `src/io/slide.py`,
`src/utils/geometry.py`, `src/infer/stitch.py`, `src/io/annotations.py`, the
patch/stride config, or any ADR marked Accepted.

```bash
# 1. Snapshot the current good state
git tag -a pre/$(date +%Y%m%d)-stitch-rework -m "pre-change snapshot; FROC 0.78"
dvc push

# 2. Branch
git switch -c risky/stitch-rework

# 3. Establish the baseline you will be compared against, BEFORE changing anything
make verify 2>&1 | tee artifacts/reports/baseline_verify.log
python scripts/04_infer_slide.py --ckpt artifacts/ckpt/$(cat artifacts/ckpt/LATEST)/best.pt \
  --slide tests/fixtures/mini/synth_tumor.tif --out /tmp/baseline

# 4. Make the change in small commits, each independently revertible
git commit -m "refactor: extract gaussian window into stitch._window()"
git commit -m "feat: chunk-aligned traversal order"
#   NOT one commit called "stitch rework"

# 5. A/B the numerics against the baseline, not against intuition
python scripts/04_infer_slide.py --ckpt ... --slide ... --out /tmp/candidate
python scripts/qc_heatmap_diff.py --a /tmp/baseline --b /tmp/candidate
# Expect an explicit, explainable delta:
#   max|Δ| 0.041  mean|Δ| 0.0006  seam ratio 1.02 -> 1.01  coverage 0.00% -> 0.00%

# 6. Full gate
make verify-full

# 7. Merge with history preserved
git switch main && git merge --no-ff risky/stitch-rework
```

`--no-ff` matters: it creates a single merge commit that `git revert -m 1` can
undo in one operation (**R1**). A fast-forwarded feature branch has to be
reverted commit by commit under pressure.

**Abort criteria — stop and revert immediately, do not debug forward:**
- Any V-gate that previously passed now fails.
- Coverage holes appear (V8.4 non-zero). No tolerance.
- Test-split Dice moves by more than 3 points with no mechanism you can name.
- The change requires disabling an assertion to proceed.

That last one is the reliable signal. An assertion that has to be switched off
is almost always correct.

---

## R8 — Emergency triage: "the pipeline produces garbage and I don't know why"

Work outward from the cheapest, most-constrained check.

```bash
# 1. Am I even where I think I am?
git status --short --branch && git log --oneline -3 && dvc status

# 2. Is the environment the one that produced the last good result?
pip freeze | sha1sum
python -c "
import torch;print(__import__('hashlib').sha1(
 '\n'.join(torch.load('artifacts/ckpt/'+open('artifacts/ckpt/LATEST').read().strip()
 +'/best.pt',map_location='cpu')['pip_freeze']).encode()).hexdigest())"
#    mismatch -> R4

# 3. Do the cheap gates still pass?
make verify                     # V0–V6, 12 min — narrows the layer

# 4. Bisect against a gate, not against a vibe
git bisect start
git bisect bad HEAD
git bisect good v0.3.0-froc0.78
git bisect run bash -c 'pytest tests/unit -q && python scripts/qc_overlay.py \
  --slide tests/fixtures/mini/synth_tumor.tif --corners --assert-min-iou 0.99'
git bisect reset
```

`git bisect run` with a **gate command** rather than manual inspection is the
whole point — it turns 12 commits into 4 automated steps and removes the
judgement call from the loop.

```bash
# 5. If code is clean, the artifacts are stale -> R5
# 6. If artifacts are clean, the checkpoint is mispaired -> R6 compat check
```

---

## R9 — Raw data loss

Raw WSIs are **not** in git or DVC (terabytes). They are a restore-from-source
dependency, and ADR-005 makes training depend on them at runtime.

```bash
# Manifest is versioned even though the pixels are not
sha256sum data/raw/*.tif > data/raw_manifest.sha256
git add data/raw_manifest.sha256 && git commit -m "chore: raw data manifest"

# After a restore, verify before trusting
sha256sum -c data/raw_manifest.sha256 | grep -v ': OK$' || echo "ALL FILES VERIFIED"
```

If a slide fails verification, re-download that slide; do not repair it. A
partially-corrupt TIFF often still opens, reads plausible pixels in the
undamaged tiles, and poisons training silently.

If raw data is unrecoverable, the LMDB cache (ADR-005) — if it was built —
contains the training subset and permits retraining but not re-indexing at a
different patch size.

---

## Recovery Time Objectives

| Scenario | Procedure | RTO |
|---|---|---|
| Bad merge on main | R1 | 15 min + 12 min verify |
| Local mistake | R2 / R3 | < 5 min |
| Dependency regression | R4 | 30 min |
| Tissue/index logic reverted | R5 (stages 01–02) | ~45 min |
| Training logic reverted | R5 (stage 03) | ~14 h |
| Stitcher reverted | R5 (stage 04, 50 slides) | ~2 h |
| Bad model promoted | R6 | 5 min |
| Full rebuild from raw | R5 all stages | ~17 h |
| Raw data restore | R9 | source-dependent |

---

## Quick Reference

```bash
git revert -m 1 <merge-sha>            # undo a merged PR, keep history
git reset --hard origin/main           # local branch == remote (destroys local)
git stash push -u -m "msg"             # park everything, including untracked
git reflog --date=iso | head -40       # find anything "lost" in the last 90d
git branch rescue/<name> <sha>         # recover onto a branch, never onto main
git bisect run <gate-command>          # automated regression hunt
dvc checkout                           # artifacts matching the current SHA
dvc status                             # artifact/code drift detector
make verify                            # V0–V6, 12 min
make verify-full                       # V0–V10
```

**Three things that are never done here:** force-push to `main`; delete
artifacts instead of quarantining them; disable an assertion to make a rollback
proceed.
