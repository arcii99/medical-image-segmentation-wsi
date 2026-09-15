#!/usr/bin/env python
"""Stage 03 -- training.

The checkpoint is the project's unit of attribution: it carries the resolved
config, git SHA, dirty flag, pip freeze, index hash, seed, and the fitted
threshold tau.  Stage 05 refuses to report numbers from a dirty tree
(ADR-014), which is only enforceable because all of that is recorded here.
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from _common import base_parser, git_state  # noqa: E402
from src.data import index as idx  # noqa: E402
from src.utils.config import config_hash, load  # noqa: E402
from src.utils.logging import MetricWriter, setup  # noqa: E402
from src.utils.seed import seed_everything  # noqa: E402

log = logging.getLogger("stage03")


def _pip_freeze() -> list[str]:
    try:
        return subprocess.check_output([sys.executable, "-m", "pip", "freeze"],
                                       text=True).splitlines()
    except Exception:
        return []


def main() -> int:
    p = base_parser("03_train")
    p.add_argument("--out", default=None)
    p.add_argument("--resume", default=None)
    p.add_argument("--dry-run", action="store_true",
                   help="build everything, run ONE forward+backward, exit")
    a = p.parse_args()
    # Do NOT append the model config unconditionally.
    #
    # Config files are merged left to right, so anything appended here
    # overrides everything the caller passed. Appending the model config meant
    # a trailing `configs/cpu_smoke.yaml` was silently clobbered: the run used
    # batch 16 instead of 4 and was OOM-killed, with nothing in the log to
    # suggest the requested config had been ignored.
    #
    # Insert the default FIRST instead, so caller configs always win, and skip
    # it entirely when the caller named a model config themselves.
    paths = list(a.config)
    if not any("model" in Path(p_).stem for p_ in paths):
        paths.insert(0, "configs/model_unet_effb0.yaml")
    cfg = load(paths, a.set)
    seed_everything(int(cfg.seed))

    import torch
    from torch.utils.data import DataLoader

    from src.data.dataset import PatchDataset, worker_init
    from src.data.sampler import BalancedSampler
    from src.models.registry import build
    from src.models.unet_effb0 import set_encoder_trainable
    from src.train.losses import build_loss
    from src.train.metrics import ConfusionAccumulator
    from src.train.schedule import cosine_with_warmup, param_groups

    sha, dirty = git_state()
    run_id = f"{date.today():%Y%m%d}_{sha[:7]}{'-dirty' if dirty else ''}_{config_hash(cfg)}"
    run_dir = Path(a.out or cfg.paths.ckpt) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    setup(run_dir)
    mw = MetricWriter(run_dir)
    (run_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2, default=str))
    log.info("run %s  (dirty=%s)", run_id, dirty)
    # Echo what was actually resolved. A config that silently did not apply is
    # indistinguishable from one that did, until the run dies.
    log.info("configs   %s", " ".join(paths))
    log.info("effective device=%s batch=%s workers=%s epochs=%s "
             "max_steps=%s augment=%s model=%s",
             cfg.hw.device, cfg.train.batch_size, cfg.hw.dataloader_workers,
             cfg.train.max_epochs, cfg.train.get("max_steps", 0) or "-",
             cfg.data.get("augment", True), cfg.model.name)

    index_path = Path(cfg.paths.index) / "patches.parquet"
    have_index = index_path.exists()
    if have_index:
        df = idx.load(index_path)
        idx.assert_no_split_leakage(df)
        slides = pd.read_parquet(Path(cfg.paths.index) / "slides.parquet")
    else:
        # A patchset export carries its own manifest; the full index and the
        # slide archive stay on the machine that built it.
        df = pd.DataFrame(columns=["slide_id", "split", "tumor_frac"])
        slides = pd.DataFrame(columns=["slide_id", "path", "residual_scale"])
    paths = dict(zip(slides.slide_id, map(Path, slides.path)))
    anns = {s: Path(p).with_suffix(".xml") for s, p in paths.items()
            if Path(p).with_suffix(".xml").exists()}
    resid = dict(zip(slides.slide_id, slides.residual_scale))

    tr, va = df[df.split == "train"], df[df.split == "val"]
    assert not (set(tr.slide_id) & set(va.slide_id)), "slide-level split leakage"


    source = str(cfg.data.get("source", "slides"))
    if source == "patchset":
        # Portable exported patches -- no slide archive required. See
        # src/data/patchset.py for what this trades away.
        from src.data.patchset import PatchSetDataset
        root = cfg.data.patchset_root
        train_ds = PatchSetDataset(root, "train", train=True, seed=cfg.seed)
        val_ds = PatchSetDataset(root, "val", train=False, seed=cfg.seed)
        tr = pd.DataFrame({"tumor_frac": train_ds.index.tumor_frac,
                           "slide_id": train_ds.index.slide_id})
        log.info("training from exported patchset at %s", root)
    else:
        train_ds = PatchDataset(tr, paths, anns, train=True,
                                jitter=cfg.patching.jitter, seed=cfg.seed,
                                residual_scale=resid)
        val_ds = PatchDataset(va, paths, anns, train=False,
                              residual_scale=resid)
    sampler = BalancedSampler(tr, neg_per_pos=cfg.train.neg_per_pos,
                              tumor_threshold=cfg.patching.tumor_threshold,
                              seed=cfg.seed)
    want = str(cfg.hw.device)
    if want.startswith("cuda") and not torch.cuda.is_available():
        log.warning("config asked for %s but no CUDA device is visible; "
                    "falling back to CPU. Full training on CPU is not "
                    "practical -- this path is for the wiring smoke test.",
                    want)
    dev = torch.device(want if (want == "cpu" or torch.cuda.is_available())
                       else "cpu")

    if dev.type == "cpu":
        # Rough activation budget for UNet-EffNetB0 in fp32, measured against
        # the documented 5.8 GB at batch 16 in bf16 and doubled for fp32.
        est_gb = cfg.train.batch_size * 0.72
        try:
            import os as _os
            total_gb = (_os.sysconf("SC_PAGE_SIZE")
                        * _os.sysconf("SC_PHYS_PAGES")) / 1024 ** 3
        except (ValueError, AttributeError):
            total_gb = float("nan")
        log.info("CPU run: batch %d needs roughly %.1f GB; machine has "
                 "%.1f GB", cfg.train.batch_size, est_gb, total_gb)
        if total_gb == total_gb and est_gb > 0.55 * total_gb:
            log.warning(
                "this is likely to be OOM-killed (the kernel kills the "
                "process with no Python traceback -- you just see 'Killed'). "
                "Lower it: --set train.batch_size=%d",
                max(1, int(0.35 * total_gb / 0.72)))

    # pin_memory only helps when staging tensors for a GPU copy; on CPU it
    # just warns.
    dl = dict(num_workers=cfg.hw.dataloader_workers, worker_init_fn=worker_init,
              persistent_workers=cfg.hw.dataloader_workers > 0,
              pin_memory=(dev.type == "cuda"),
              prefetch_factor=4 if cfg.hw.dataloader_workers else None)
    train_dl = DataLoader(train_ds, batch_size=cfg.train.batch_size,
                          sampler=sampler, drop_last=True, **dl)
    val_dl = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False, **dl)

    model = build(cfg.model).to(dev)
    crit = build_loss(cfg.loss).to(dev)
    opt = torch.optim.AdamW(param_groups(model, cfg.train.lr_encoder,
                                         cfg.train.lr_decoder, cfg.train.weight_decay))
    sched = cosine_with_warmup(opt, cfg.train.warmup_epochs, cfg.train.max_epochs)
    amp = dict(device_type=dev.type, dtype=getattr(torch, cfg.hw.amp_dtype),
               enabled=dev.type == "cuda")

    if a.dry_run:
        # Seconds, not minutes. Exercises every construction step and one full
        # optimisation step, which is where ordering and shape bugs live --
        # the kind that otherwise surface twenty minutes into a run.
        log.info("dry run: device=%s batch=%d source=%s",
                 dev, cfg.train.batch_size, source)
        batch = next(iter(train_dl))
        x = batch["image"].to(dev); y = batch["mask"].to(dev)
        log.info("batch image %s %s  mask %s  mask values %s",
                 list(x.shape), x.dtype, list(y.shape),
                 sorted(set(y.unique().tolist())))
        with torch.autocast(**amp):
            logits = model(x)
            loss = crit(logits, y)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               cfg.train.grad_clip)
        opt.step(); opt.zero_grad(set_to_none=True)
        log.info("logits %s   loss %.4f   grad-norm %.4f",
                 list(logits.shape), float(loss), float(gnorm))
        problems = []
        if list(logits.shape) != list(y.shape):
            problems.append(f"logits {list(logits.shape)} != mask {list(y.shape)}")
        if not torch.isfinite(loss):
            problems.append("loss is not finite")
        if float(gnorm) == 0.0:
            problems.append("gradient norm is exactly 0 -- graph detached?")
        if set(y.unique().tolist()) - {0.0, 1.0}:
            problems.append("mask is not binary")
        for m in problems:
            log.error("DRY RUN PROBLEM: %s", m)
        log.info("DRY RUN %s", "FAILED" if problems else "OK -- wiring is sound")
        return 1 if problems else 0

    start_epoch, best, patience = 0, -1.0, 0
    if a.resume:
        ck = torch.load(a.resume, map_location="cpu")
        model.load_state_dict(ck["state_dict"], strict=True)
        opt.load_state_dict(ck["optimizer"]); start_epoch = ck["epoch"] + 1
        best = ck.get("best_dice", -1.0)
        log.info("resumed %s at epoch %d", a.resume, start_epoch)

    max_steps = int(cfg.train.get("max_steps", 0) or 0)
    for epoch in range(start_epoch, int(cfg.train.max_epochs)):
        set_encoder_trainable(model, epoch >= cfg.train.freeze_encoder_epochs)
        model.train()
        for step, batch in enumerate(train_dl):
            x = batch["image"].to(dev, non_blocking=True)
            y = batch["mask"].to(dev, non_blocking=True)
            with torch.autocast(**amp):
                loss = crit(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            opt.step(); opt.zero_grad(set_to_none=True)
            if step % cfg.log_every == 0:
                mw.log(epoch=epoch, step=step, loss=float(loss.detach()),
                       lr=opt.param_groups[-1]["lr"])
                log.info("e%02d s%05d loss=%.4f", epoch, step,
                         float(loss.detach()))
            if max_steps and step + 1 >= max_steps:
                break
        sched.step()

        model.eval(); acc = ConfusionAccumulator()
        with torch.no_grad(), torch.autocast(**amp):
            for batch in val_dl:
                acc.update(model(batch["image"].to(dev)), batch["mask"].to(dev))
        m = acc.summary()
        mw.log(epoch=epoch, phase="val", **m)
        log.info("e%02d val dice=%.4f iou=%.4f tau=%.2f", epoch, m["dice"],
                 m["iou"], m["threshold"])

        payload = dict(state_dict=model.state_dict(), ema_state_dict=model.state_dict(),
                       optimizer=opt.state_dict(), epoch=epoch,
                       threshold=m["threshold"], best_dice=max(best, m["dice"]),
                       resolved_config=cfg.to_dict(), git_sha=sha, dirty=dirty,
                       pip_freeze=_pip_freeze(),
                       index_hash=(idx.index_hash(index_path) if have_index
                                   else "patchset"),
                       seed=int(cfg.seed), run_id=run_id)
        torch.save(payload, run_dir / "last.pt")
        if m["dice"] > best:
            best, patience = m["dice"], 0
            torch.save(payload, run_dir / "best.pt")
        else:
            patience += 1
            if patience >= cfg.train.early_stop_patience:
                log.info("early stop at epoch %d (best %.4f)", epoch, best); break

    (Path(cfg.paths.ckpt) / "LATEST").write_text(run_id)
    log.info("done. best val dice %.4f -> %s", best, run_dir / "best.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
