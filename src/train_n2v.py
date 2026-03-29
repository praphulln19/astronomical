# src/train_n2v.py
from __future__ import annotations
import argparse
import atexit
import os
from pathlib import Path
import time

import torch
from torch.optim import AdamW

# ---- AMP imports (compat for old/new PyTorch) ----
try:
    from torch.amp import autocast, GradScaler  # PyTorch 2.x+
    HAS_TORCH_AMP = True
except Exception:
    from torch.cuda.amp import autocast, GradScaler  # PyTorch < 2.0
    HAS_TORCH_AMP = False

from dataio.loaders import make_loaders
from models.unet_blindspot import UNetBlindspot
from losses.masked_loss import make_center_mask, masked_l2


def _log(msg: str):
    # Always flush so VS Code terminal shows progress immediately.
    print(msg, flush=True)


def _pid_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def acquire_run_lock(outdir: str | Path, force: bool = False):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    lock_path = outdir / "train.lock"

    if lock_path.exists() and not force:
        try:
            prev_pid = int(lock_path.read_text(encoding="utf-8").strip() or "0")
        except Exception:
            prev_pid = 0
        if _pid_exists(prev_pid):
            raise RuntimeError(
                f"Another training process appears active (pid={prev_pid}). "
                f"Stop it first or rerun with --force_lock."
            )

    lock_path.write_text(str(os.getpid()), encoding="utf-8")

    def _cleanup_lock():
        try:
            if lock_path.exists():
                lock_path.unlink()
        except Exception:
            pass

    atexit.register(_cleanup_lock)
    return lock_path


def save_ckpt(model, opt, step, outdir: str | Path):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "opt": opt.state_dict(), "step": step},
        outdir / f"ckpt_{step}.pt",
    )


def train(args):
    lock_path = acquire_run_lock(args.outdir, force=args.force_lock)
    _log(f"[lock] acquired {lock_path}")

    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"

    if use_cuda:
        torch.backends.cudnn.benchmark = True  # faster on fixed 512x512

    train_loader, val_loader, _ = make_loaders(
        Path(args.patch_csv),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=use_cuda,   # only pin on CUDA
        use_aug=True,
    )

    n_train = len(train_loader.dataset)
    n_val = len(val_loader.dataset)
    if n_train == 0:
        raise RuntimeError("Train split is empty. Check patches_splits.csv and split column.")
    if n_val == 0:
        raise RuntimeError("Val split is empty. Check patches_splits.csv and split column.")

    _log(
        f"[setup] device={device} amp={bool(args.amp and use_cuda)} "
        f"train_samples={n_train} val_samples={n_val} "
        f"train_batches={len(train_loader)} val_batches={len(val_loader)}"
    )

    model = UNetBlindspot(in_ch=3, base=args.base).to(device)
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # ---- Load checkpoint if resuming ----
    global_step = 0
    start_epoch = 1
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            _log(f"[resume] Loading checkpoint from {resume_path}")
            # Explicitly set weights_only to avoid ambiguous defaults in newer PyTorch.
            ckpt = torch.load(resume_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model"])
            opt.load_state_dict(ckpt["opt"])
            global_step = ckpt.get("step", 0)
            _log(f"[resume] Resuming from step {global_step}")
        else:
            _log(f"[warn] Checkpoint {resume_path} not found, starting from scratch")

    # ---- AMP setup (PyTorch 2.6 prefers positional device arg) ----
    if HAS_TORCH_AMP:
        scaler = GradScaler("cuda" if use_cuda else "cpu",
                            enabled=(args.amp and use_cuda))
        autocast_args = dict(device_type=("cuda" if use_cuda else "cpu"),
                             enabled=(args.amp and use_cuda))
    else:
        scaler = GradScaler(enabled=(args.amp and use_cuda))
        autocast_args = dict(enabled=(args.amp and use_cuda))

    best_val = float("inf")
    no_improve = 0  # for early stopping

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        _log(f"[epoch] {epoch}/{args.epochs} started")
        model.train()
        running = 0.0

        for it, batch in enumerate(train_loader, 1):
            x = batch["image"].to(device, non_blocking=use_cuda)  # [B,3,512,512]
            B, C, H, W = x.shape

            # Blind-spot mask: 0 = hidden region, 1 = contributes to loss
            mask = make_center_mask(B, H, W, hole=args.hole, device=device)
            with torch.no_grad():
                target = x  # predict x itself; masked region excluded from loss

            opt.zero_grad(set_to_none=True)
            with autocast(**autocast_args):
                y = model(x)
                loss = masked_l2(y, target, mask)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running += loss.item()
            global_step += 1

            if global_step % args.log_every == 0:
                _log(f"[train] epoch {epoch} step {global_step} loss {running/args.log_every:.5f}")
                running = 0.0

            # Limit work per epoch if requested
            if args.max_steps_per_epoch and it >= args.max_steps_per_epoch:
                break

        # ---- validation ----
        model.eval()
        with torch.no_grad():
            vloss = 0.0
            vcount = 0
            for batch in val_loader:
                x = batch["image"].to(device, non_blocking=use_cuda)
                B, C, H, W = x.shape
                mask = make_center_mask(B, H, W, hole=args.hole, device=device)
                with autocast(**autocast_args):
                    y = model(x)
                    vloss += masked_l2(y, x, mask).item()
                vcount += 1
            vloss /= max(1, vcount)

        _log(f"[val] epoch {epoch} masked-L2 {vloss:.5f}")
        _log(f"[epoch] {epoch}/{args.epochs} finished in {(time.time() - epoch_start):.1f}s")

        # ---- early stopping + best checkpoint ----
        if vloss + args.min_delta < best_val:
            best_val = vloss
            no_improve = 0
            save_ckpt(model, opt, global_step, args.outdir)
            _log(f"[ckpt] saved at step {global_step}")
        else:
            no_improve += 1
            _log(f"[val] no improvement count = {no_improve}")
            if no_improve >= args.early_stop_patience:
                _log(f"[early stop] stopping after {args.early_stop_patience} epochs without improvement")
                break

    _log(f"[done] best val masked-L2: {best_val:.5f}  ckpts in {args.outdir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch_csv", type=str, default="data/manifests/patches_splits.csv")
    ap.add_argument("--outdir", type=str, default="checkpoints/n2v_unet")
    ap.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=0)  # Windows-safe; bump on Linux/Colab
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--base", type=int, default=48)
    ap.add_argument("--hole", type=int, default=5)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--log_every", type=int, default=100)

    # New controls
    ap.add_argument("--max_steps_per_epoch", type=int, default=0,
                    help="If >0, limit number of training batches per epoch.")
    ap.add_argument("--early_stop_patience", type=int, default=3,
                    help="Stop after N epochs with no val improvement.")
    ap.add_argument("--min_delta", type=float, default=1e-4,
                    help="Minimum improvement in val loss to count as progress.")
    ap.add_argument("--force_lock", action="store_true",
                    help="Force start even if a previous train.lock exists.")

    args = ap.parse_args()
    train(args)
