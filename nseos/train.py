"""Training with early stopping, fixed-RNG validation, cached runs.

Every run is stored under runs/<name>_<tag>/ (model.pt + train_info.json) and is
skipped on re-run unless --force, so ablations are resumable run by run.
"""
import os
import time

import torch
from torch.utils.data import DataLoader, RandomSampler

from .io import load_json, save_json
from .model import EOSCatalogDataset, build_model, collate_fn, move_to_device, n_params


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def amp_dtype(device):
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def make_loader(samples, cfg, shuffle, augment=False, allowed_types=None, catalog_fn=None, workers=None,
                batch_size=None, seed=0):
    tc = cfg["train"]
    workers = tc["num_workers"] if workers is None else workers
    workers = min(workers, os.cpu_count() or 1)
    ds = EOSCatalogDataset(samples, augment=augment, allowed_types=allowed_types, sim_cfg=cfg["sim"],
                           base_seed=cfg["seed"] + seed, catalog_fn=catalog_fn)
    if workers and augment:
        ds.shared_epoch = torch.zeros((), dtype=torch.int64).share_memory_()
    # A newly spawned iterator consumes a worker seed. Keep this independent
    # of the sampler so restarting an epoch yields the same shuffled indices.
    sampler = RandomSampler(ds, generator=torch.Generator().manual_seed(ds.base_seed)) if shuffle else None
    return DataLoader(ds, batch_size=batch_size or tc["batch_size"], sampler=sampler, collate_fn=collate_fn,
                      num_workers=workers, pin_memory=torch.cuda.is_available(),
                      persistent_workers=workers > 0,
                      generator=torch.Generator().manual_seed(ds.base_seed))


@torch.no_grad()
def eval_flow_loss(model, loader, device, seed=1234):
    """Validation loss with a fixed (t, z) RNG so epochs are compared on identical draws."""
    model.eval()
    g = torch.Generator(device=device).manual_seed(seed)
    tot, n = 0.0, 0
    for batch in loader:
        batch = move_to_device(batch, device)
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype(device), enabled=device.type == "cuda"):
            loss = model.compute_loss(batch, batch["theta"], generator=g)
        tot += loss.item() * batch["num_catalogs"]
        n += batch["num_catalogs"]
    return tot / max(n, 1)


def train_flow_model(model, train_loader, val_loader, device, max_epochs=400, patience=40, lr=3e-4,
                     wd=1e-4, compile_model=False, log_every=5, tag="model", ckpt_path=None):
    """Early-stopped flow-matching training. With ckpt_path, the full state (weights, optimiser,
    scheduler, scaler, history, best weights, patience counter, elapsed time) is saved atomically
    after every epoch, and an existing checkpoint is resumed from the next epoch."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs, eta_min=1e-5)
    adt = amp_dtype(device)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda" and adt == torch.float16)
    run = model
    if compile_model:
        try:
            run = torch.compile(model, dynamic=True)
        except Exception as e:  # pragma: no cover
            print(f"torch.compile unavailable ({e})")
    hist = {"train": [], "val": []}
    best, best_epoch, best_state, bad = float("inf"), -1, None, 0
    start_epoch, elapsed_before = 0, 0.0
    if ckpt_path and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        hist, best, best_epoch, bad = ck["hist"], ck["best"], ck["best_epoch"], ck["bad"]
        best_state, start_epoch, elapsed_before = ck["best_state"], ck["epoch"] + 1, ck["elapsed"]
        torch.set_rng_state(ck["torch_rng"].cpu())   # map_location moved it to the GPU
        if device.type == "cuda" and ck.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(ck["cuda_rng"].cpu(), device)
        print(f"[{tag}] resuming from epoch {start_epoch} (best {best:.4f} @ {best_epoch})", flush=True)
    t0 = time.time()
    stopped = bad >= patience
    for epoch in range(start_epoch, 0 if stopped else max_epochs):
        train_loader.dataset.epoch = epoch
        train_loader.sampler.generator.manual_seed(train_loader.dataset.base_seed + epoch)
        model.train()
        tt, ts = 0.0, 0
        for batch in train_loader:
            batch = move_to_device(batch, device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=adt, enabled=device.type == "cuda"):
                loss = run.compute_loss(batch, batch["theta"])
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            tt += loss.item()
            ts += 1
        sched.step()
        hist["train"].append(tt / max(ts, 1))
        hist["val"].append(eval_flow_loss(model, val_loader, device))
        if hist["val"][-1] < best:
            best, best_epoch, bad = hist["val"][-1], epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if epoch % log_every == 0:
            print(f"[{tag}] ep {epoch:4d} train {hist['train'][-1]:.4f} val {hist['val'][-1]:.4f} "
                  f"(best {best:.4f} @ {best_epoch})", flush=True)
        if ckpt_path:
            tmp = ckpt_path + ".tmp"
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "scaler": scaler.state_dict(), "hist": hist, "best": best, "best_epoch": best_epoch,
                        "bad": bad, "best_state": best_state, "epoch": epoch,
                        "elapsed": elapsed_before + time.time() - t0, "torch_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None}, tmp)
            os.replace(tmp, ckpt_path)
        if bad >= patience:
            print(f"[{tag}] early stop at epoch {epoch}")
            break
    model.load_state_dict(best_state)
    info = {"best_epoch": best_epoch, "best_val_loss": best, "epochs_run": len(hist["val"]),
            "max_epochs": max_epochs, "patience": patience, "wall_s": elapsed_before + time.time() - t0,
            "resumed_from_epoch": start_epoch if start_epoch else None,
            "n_params": n_params(model), "history": hist,
            "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"}
    print(f"[{tag}] best val {best:.4f} @ epoch {best_epoch} / {info['epochs_run']} epochs, "
          f"{info['wall_s'] / 3600:.2f} h")
    return info


def train_run(name, cfg, paths, splits, aggregator="set_transformer", seed=0, max_epochs=None,
              patience=None, allowed_types=None, train_samples=None, val_samples=None, stats=None,
              catalog_fn=None, force=False):
    """Train (or load) one cached run. Returns (model, info)."""
    device = get_device()
    rdir = paths.run(name)
    mpath, ipath = os.path.join(rdir, "model.pt"), os.path.join(rdir, "train_info.json")
    stats = splits["stats"] if stats is None else stats
    model = build_model(cfg, stats, aggregator=aggregator, seed=cfg["seed"] + seed, device=device)
    if os.path.exists(mpath) and os.path.exists(ipath) and not force:
        model.load_state_dict(torch.load(mpath, map_location=device))
        print(f"[cache] loaded run {name}")
        return model.eval(), load_json(ipath, {})
    tc = cfg["train"]
    tr = splits["train"] if train_samples is None else train_samples
    va = splits["val"] if val_samples is None else val_samples
    tl = make_loader(tr, cfg, shuffle=True, augment=True, allowed_types=allowed_types,
                     catalog_fn=catalog_fn, seed=seed)
    vl = make_loader(va, cfg, shuffle=False, augment=False, allowed_types=allowed_types)
    ckpt = os.path.join(rdir, "checkpoint.pt")
    if force and os.path.exists(ckpt):
        os.remove(ckpt)
    info = train_flow_model(model, tl, vl, device, max_epochs=max_epochs or tc["max_epochs"],
                            patience=patience or tc["patience"], lr=tc["lr"], wd=tc["weight_decay"],
                            compile_model=tc["compile"], tag=name, ckpt_path=ckpt)
    info.update({"aggregator": aggregator, "seed": seed, "allowed_types": allowed_types,
                 "batch_size": tc["batch_size"], "lr": tc["lr"], "weight_decay": tc["weight_decay"]})
    torch.save(model.state_dict(), mpath + ".tmp")
    os.replace(mpath + ".tmp", mpath)
    save_json(ipath, info)
    if os.path.exists(ckpt):
        os.remove(ckpt)     # finished: the final model.pt supersedes the epoch checkpoint
    return model.eval(), info


def load_run(name, cfg, paths, stats, aggregator="set_transformer"):
    device = get_device()
    model = build_model(cfg, stats, aggregator=aggregator, device=device)
    model.load_state_dict(torch.load(os.path.join(paths.run(name), "model.pt"), map_location=device))
    return model.eval()
