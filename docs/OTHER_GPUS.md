# Running on another GPU

Two targets are covered here: a shared institutional GPU with a ~5–6 GB
budget, and a 6 GB laptop card (RTX 4050) with a ~4 GB budget. The pipeline
is the same in both cases; only the memory configuration and the way the data
arrives differ.

---

## How much memory does this actually need?

Measured on a T4 with float16 autocast at 512 × 512:

```
total  ~=  0.60 GB fixed  +  0.150 GB per sample
           ^^^^^^^^^^^^^     ^^^^^^^^^^^^^^^^^^^
           weights, AdamW     activations held for
           states, CUDA       the backward pass
           context, cuDNN
```

| batch | estimated GB | steps per epoch (104,759 patches) |
|---|---|---|
| 8 | 1.8 | 13,094 |
| 16 | 3.0 | 6,547 |
| 20 | 3.6 | 5,237 |
| 24 | 4.2 | 4,364 |
| **32** | **5.4** | **3,273** |
| 40 | 6.6 | 2,618 |

Colab used ~3 GB because the default batch is 16. On a 6 GB budget you can
go **up** to 32, which also halves the number of steps per epoch.

Two mechanisms make the budget real rather than aspirational:

* **`hw.max_gpu_gb`** calls `set_per_process_memory_fraction`, so exceeding
  the cap raises an ordinary CUDA out-of-memory error instead of quietly
  consuming memory another job was relying on. On a shared machine this is
  the difference between a promise and a claim.
* **`train.grad_accum_steps`** sums gradients over N micro-batches before
  stepping the optimiser. The *effective* batch — and therefore the gradient
  noise and the learning rate tuned for it — is `batch_size × accum`, while
  memory is set by `batch_size` alone.

Peak allocated and reserved memory are logged after every epoch, so the
estimate above can be replaced with a measurement on your hardware.

---

## A. College GPU (Linux, shared)

### 1. Environment

```bash
ssh you@cluster
conda create -n wsi python=3.11 -y && conda activate wsi
git clone https://github.com/<you>/medical-image-segmentation-wsi.git wsi-metastasis-seg
cd wsi-metastasis-seg
pip install -e ".[dev]"
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install segmentation-models-pytorch
python scripts/version.py          # every box should be ticked
python -m pytest tests -q          # 84 passed
```

Check what you have been given:

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.get_device_name(), torch.cuda.get_device_capability())"
```

Compute capability 8.0 or above means bfloat16 is used; below that, float16
with a gradient scaler. The training log states which, so you do not have to
remember.

### 2. Data

The patch set is already on Google Drive, so pull it there rather than moving
it from your laptop.

```bash
rclone config          # same account you uploaded with
rclone copy gdrive:wsi/patchset ./patchset -P --transfers 4
```

No rclone on the cluster? `pip install gdown` and share the folder by link,
or copy over `scp` from your PC.

Extract the shards to **local scratch**, never to a network filesystem —
reading 232,138 small files over NFS will starve the GPU exactly as Google
Drive would:

```bash
mkdir -p /scratch/$USER/patchset/files
cp patchset/manifest.parquet patchset/export_info.json /scratch/$USER/patchset/
for t in patchset/shards/*.tar; do tar xf "$t" -C /scratch/$USER/patchset/files; done
ls /scratch/$USER/patchset/files | wc -l     # expect 232138
```

### 3. Verify, then train

```bash
python scripts/03_train.py \
  --config configs/base.yaml configs/data_camelyon16.yaml \
           configs/model_unet_effb0.yaml configs/patchset.yaml \
           configs/gpu_6gb.yaml \
  --set data.patchset_root=/scratch/$USER/patchset --dry-run
```

Then the overfit gate (`--set train.overfit_batches=1 --set train.max_steps=300
--set data.augment=false`), then:

```bash
tmux new -s train
python scripts/03_train.py \
  --config configs/base.yaml configs/data_camelyon16.yaml \
           configs/model_unet_effb0.yaml configs/patchset.yaml \
           configs/gpu_6gb.yaml \
  --set data.patchset_root=/scratch/$USER/patchset \
  --set paths.ckpt=$HOME/wsi_ckpt \
  --set train.max_epochs=40
```

Detach with `Ctrl-b` then `d`; reattach with `tmux attach -t train`.

### 4. If the cluster uses Slurm

```bash
cat > train.sbatch <<'SLURM'
#!/bin/bash
#SBATCH --job-name=wsi-seg
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=wsi-%j.log

source ~/miniconda3/etc/profile.d/conda.sh && conda activate wsi
cd $HOME/wsi-metastasis-seg
python scripts/03_train.py \
  --config configs/base.yaml configs/data_camelyon16.yaml \
           configs/model_unet_effb0.yaml configs/patchset.yaml \
           configs/gpu_6gb.yaml \
  --set data.patchset_root=/scratch/$USER/patchset \
  --set paths.ckpt=$HOME/wsi_ckpt \
  --set train.max_epochs=40
SLURM
sbatch train.sbatch
```

If the wall-clock limit is shorter than the run needs, resume in the next job
with `--resume $HOME/wsi_ckpt/<RUN_ID>/last.pt` and keep every other flag
identical.

---

## B. Windows laptop, RTX 4050 6 GB

The card has 6 GB, but the desktop compositor and a browser hold several
hundred megabytes, so plan against ~5 GB. `configs/gpu_4gb.yaml` caps the job
at 4 GB with batch 20 and accumulation 2, giving an effective batch of 40.

Leaving slack matters here for a specific reason: when a CUDA allocation
cannot be satisfied, recent Windows drivers fall back to *shared system
memory* rather than failing. Training continues at a fraction of the speed
with no error message. A smaller batch is much better than that.

### 1. Environment

Use Miniconda in **Anaconda Prompt**, not PowerShell — conda activation is
more reliable there.

```bat
conda create -n wsi python=3.11 -y
conda activate wsi
git clone https://github.com/<you>/medical-image-segmentation-wsi.git wsi-metastasis-seg
cd wsi-metastasis-seg
pip install -e ".[dev]"
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install segmentation-models-pytorch
python scripts/version.py
python -m pytest tests -q
```

Confirm the GPU is visible:

```bat
nvidia-smi
python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
```

An RTX 4050 is Ada Lovelace, compute capability 8.9, so **bfloat16 is used
natively** — no gradient scaler, and none of the slowdown seen on the T4.

### 2. Data

Install Google Drive for Desktop and let it sync `wsi/patchset`, or use
rclone as above. Then extract with Python rather than `tar`, which Windows
may not have:

```bat
python -c "import glob,tarfile,os; os.makedirs(r'D:\wsi\patchset\files',exist_ok=True); [tarfile.open(t).extractall(r'D:\wsi\patchset\files') for t in sorted(glob.glob(r'G:\My Drive\wsi\patchset\shards\*.tar'))]"
copy "G:\My Drive\wsi\patchset\manifest.parquet" D:\wsi\patchset\
copy "G:\My Drive\wsi\patchset\export_info.json" D:\wsi\patchset\
```

Extract to a **local SSD**, not to the synced Drive folder: reading training
data through the Drive client is the same mistake as reading it through the
Colab mount.

Check the count:

```bat
python -c "import os; print(len(os.listdir(r'D:\wsi\patchset\files')))"
```

### 3. Verify, then train

```bat
python scripts/03_train.py ^
  --config configs/base.yaml configs/data_camelyon16.yaml ^
           configs/model_unet_effb0.yaml configs/patchset.yaml ^
           configs/gpu_4gb.yaml ^
  --set data.patchset_root=D:/wsi/patchset --dry-run
```

Note `^` for line continuation instead of `\`, and forward slashes inside the
config value (Python accepts them on Windows and they avoid backslash-escape
surprises).

Then the overfit gate, then training:

```bat
python scripts/03_train.py ^
  --config configs/base.yaml configs/data_camelyon16.yaml ^
           configs/model_unet_effb0.yaml configs/patchset.yaml ^
           configs/gpu_4gb.yaml ^
  --set data.patchset_root=D:/wsi/patchset ^
  --set paths.ckpt=D:/wsi/ckpt ^
  --set train.max_epochs=40
```

### 4. Windows-specific notes

* **DataLoader workers use `spawn`, not `fork`.** Each worker re-imports the
  module and unpickles the dataset, so startup is slower and 4 workers is
  usually better than 8. If you see pickling errors, `--set
  hw.dataloader_workers=0` isolates whether workers are the cause.
* **Sleep kills training.** Set the power plan to never sleep on AC, or the
  process dies mid-epoch. `last.pt` makes that recoverable but not free.
* **Thermal throttling.** A laptop GPU under sustained load will downclock.
  Expect slower steps after the first twenty minutes; this is normal and not
  a bug.
* **Watch memory live** in a second prompt:
  `nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv -l 5`

---

## Tuning the batch to your card

The configs are starting points. After the first epoch the log prints:

```
peak GPU memory 5.31 GB allocated, 5.60 GB reserved
```

*Reserved* is what `nvidia-smi` shows — PyTorch's caching allocator holds
memory it has finished with, so reserved always exceeds allocated. Judge your
budget against reserved.

| Observation | Action |
|---|---|
| Peak well under budget | Raise `train.batch_size`; fewer steps per epoch |
| CUDA out of memory | Lower `batch_size`, raise `grad_accum_steps` to compensate |
| Want a different effective batch | Change `grad_accum_steps`, not the learning rate — `lr_scale_with_batch` handles the rest |
| Fragmentation errors despite headroom | `set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |

One thing not to do: raise `batch_size` and separately adjust `lr_decoder`.
`base.yaml` already scales the learning rate with batch size, so changing
both applies the correction twice.
