# Training on Kaggle

Kaggle suits this project better than Colab for one reason: **datasets persist
between sessions**. On Colab the 12 GB patch set is copied from Drive and
re-extracted every session. On Kaggle it is uploaded once and mounted
read-only thereafter.

| | Colab free | Kaggle |
|---|---|---|
| GPU | T4 | T4 x2 or P100 |
| Session limit | ~12 h, variable | 12 h |
| Weekly quota | not published, ~15-25 h observed | ~30 h, stated |
| Idle disconnect | yes | ~90 min in the interactive editor |
| Data | re-copied from Drive each session | mounted, no copy |
| Unattended runs | no | **yes** — Save & Run All |

That last row matters most. A committed Kaggle run executes server-side for
up to 12 hours with no browser open.

---

## Step 1 — Enable internet (one-time)

Kaggle notebooks have no network access until the account is phone-verified.
Without it, `pip install`, `git clone` and the pretrained encoder download all
fail.

  Account settings -> Phone Verification -> verify
  Then, in any notebook: Session options -> Internet -> On

## Step 2 — Do NOT upload from your machine

The Kaggle CLI uploads single-threaded in small chunks. Measured on a
9.97 Mbit/s uplink it managed 250 KB/s — about 20% of line rate, and roughly
13 hours for 12.3 GB. Even at full line rate it would be 2.7 hours.

Zipping does not help: the shards are already JPEG, so compression gains
almost nothing, and it converts 11 independently retryable files into one
12 GB upload that `kaggle datasets create` cannot resume.

**Have Kaggle pull the data instead.** The patch set is already on Google
Drive. Share that folder as *Anyone with the link → Viewer*, put the link in
cell 1 of the notebook, and Kaggle downloads it over its own connection in
5-15 minutes. Your uplink is never involved.

The notebook then offers to republish it as a Kaggle Dataset — an upload that
happens entirely inside Kaggle — so every later session attaches it and
transfers nothing at all.

### If you do want a Dataset built from your machine

Only worth it if the Drive route fails. Upload one shard at a time so a
failure costs one file rather than all of them:

```bash
pip install kaggle

# Kaggle -> Account -> API -> Create New Token  (downloads kaggle.json)
mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

cd ~/Documents/.../wsi-metastasis-seg
mkdir -p kaggle_upload/shards
cp artifacts/patchset/shards/*.tar          kaggle_upload/shards/
cp artifacts/patchset/manifest.parquet      kaggle_upload/
cp artifacts/patchset/export_info.json      kaggle_upload/

cat > kaggle_upload/dataset-metadata.json <<'EOF'
{
  "title": "CAMELYON16 patchset 512px",
  "id": "YOUR_USERNAME/camelyon16-patchset-512",
  "licenses": [{"name": "CC0-1.0"}]
}
EOF

kaggle datasets create -p kaggle_upload --dir-mode tar
```

Replace `YOUR_USERNAME` with your Kaggle username. Run it in `tmux` — there is
no resume, so a dropped connection means starting over.

Then a second, tiny dataset for the metadata that inference needs:

```bash
mkdir -p kaggle_meta
cp artifacts/index/slides.parquet artifacts/index/patches.parquet kaggle_meta/
cp -r artifacts/tissue kaggle_meta/
cat > kaggle_meta/dataset-metadata.json <<'EOF'
{"title": "CAMELYON16 index and tissue masks",
 "id": "YOUR_USERNAME/camelyon16-index", "licenses": [{"name": "CC0-1.0"}]}
EOF
kaggle datasets create -p kaggle_meta
```

## Step 3 — Create the notebook

Upload `notebooks/kaggle_train.ipynb`: **Code -> New Notebook -> File ->
Import Notebook**. Then in the right-hand panel:

- *Accelerator* -> GPU T4 x2 or GPU P100
- *Internet* -> On
- *Input -> Add Input* -> both datasets

Edit the `REPO` line in cell 2 to your GitHub URL.

## Step 4 — Where things go

```
/kaggle/input/<dataset>/    read-only    your shards and metadata
/kaggle/working/            writable     SAVED as notebook output, ~20 GB
/kaggle/temp/               writable     scratch, discarded
```

Extract the shards to `/kaggle/temp`, not `/kaggle/working`. 12 GB of patches
would eat the working quota that checkpoints need, and re-extracting costs
five minutes.

Checkpoints go to `/kaggle/working/ckpt`, which is what the next session
resumes from.

## Step 5 — Run the short cells interactively

Cells 1 to 5 — environment, code, data, dry run, gate V6.2 — are quick.
Run them in the editor and read the output.

Do not skip the dry run. It has caught two bugs in this project.

## Step 6 — Run training as a committed job

**Save Version -> Save & Run All (Commit)**.

This is the Kaggle-specific part. The interactive editor disconnects after
about 90 minutes of browser inactivity, which is useless for a multi-hour run.
A committed run executes the whole notebook server-side, up to 12 hours, with
the browser closed, and saves `/kaggle/working/` as a versioned output.

Watch the quota: *Notebooks -> your profile* shows GPU hours used this week.

## Step 7 — Resume in the next session

1. Open the notebook, *Add Input -> Notebook Output* -> its previous version.
2. Run the resume cell. It finds the newest `last.pt` under `/kaggle/input`,
   copies it into `/kaggle/working` (resume writes to the run directory, and
   `/kaggle/input` is read-only), and restarts from that epoch.
3. Commit again.

Repeat until validation stops improving. Early stopping has patience 8, so it
will halt on its own.

## Notes specific to the hardware

**P100 is `sm_60`, T4 is `sm_75`.** Neither supports bfloat16 natively. The
training script detects this and selects float16 with a gradient scaler
(BUG-031). The first lines of every run state the GPU and its capability --
check them.

**T4 x2 gives two GPUs**, but the training loop is single-GPU. The second card
sits idle. Distributed training is a config change, not an architectural one,
but it has not been implemented or tested here.

**Kaggle VMs have 2-4 CPU cores.** Keep `hw.dataloader_workers=2`. Data
loading, not the GPU, is the bottleneck at that core count (OBS-002).

## Checklist

```
[ ] phone verified, Internet = On
[ ] both datasets attached under Input
[ ] Accelerator set to GPU
[ ] REPO edited in cell 2
[ ] version.py and check_repo.py both clean
[ ] extracted file count matches manifest x 2
[ ] dry run says "wiring is sound"
[ ] gate V6.2 says PASS
[ ] first lines of the training log name the GPU and the float16 fallback
[ ] training started via Save & Run All, not the editor
```
