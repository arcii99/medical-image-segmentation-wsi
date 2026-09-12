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
    a = p.parse_args()
    cfg = load(a.config + ["configs/model_unet_effb0.yaml"], a.set)
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

    index_path = Path(cfg.paths.index) / "patches.parquet"
    df = idx.load(index_path)
    idx.assert_no_split_leakage(df)
    slides = pd.read_parquet(Path(cfg.paths.index) / "slides.parquet")
    paths = dict(zip(slides.slide_id, map(Path, slides.path)))
    anns = {s: Path(p).with_suffix(".xml") for s, p in paths.items()
            if Path(p).with_suffix(".xml").exists()}
    resid = dict(zip(slides.slide_id, slides.residual_scale))

    tr, va = df[df.split == "train"], df[df.split == "val"]
    assert not (set(tr.slide_id) & set(va.slide_id)), "slide-level split leakage"

    train_ds = PatchDataset(tr, paths, anns, train=True,
                            jitter=cfg.patching.jitter, seed=cfg.seed,
                            residual_scale=resid)
    val_ds = PatchDataset(va, paths, anns, train=False, residual_scale=resid)
    sampler = BalancedSampler(tr, neg_per_pos=cfg.train.neg_per_pos,
                              tumor_threshold=cfg.patching.tumor_threshold,
                              seed=cfg.seed)
    dl = dict(num_workers=cfg.hw.dataloader_workers, worker_init_fn=worker_init,
              persistent_workers=cfg.hw.dataloader_workers > 0, pin_memory=True,
              prefetch_factor=4 if cfg.hw.dataloader_workers else None)
    train_dl = DataLoader(train_ds, batch_size=cfg.train.batch_size,
                          sampler=sampler, drop_last=True, **dl)
    val_dl = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False, **dl)

    dev = torch.device(cfg.hw.device if torch.cuda.is_available() else "cpu")
    model = build(cfg.model).to(dev)
    crit = build_loss(cfg.loss).to(dev)
    opt = torch.optim.AdamW(param_groups(model, cfg.train.lr_encoder,
                                         cfg.train.lr_decoder, cfg.train.weight_decay))
    sched = cosine_with_warmup(opt, cfg.train.warmup_epochs, cfg.train.max_epochs)
    amp = dict(device_type=dev.type, dtype=getattr(torch, cfg.hw.amp_dtype),
               enabled=dev.type == "cuda")

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
                mw.log(epoch=epoch, step=step, loss=float(loss),
                       lr=opt.param_groups[-1]["lr"])
                log.info("e%02d s%05d loss=%.4f", epoch, step, float(loss))
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
                       pip_freeze=_pip_freeze(), index_hash=idx.index_hash(index_path),
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
