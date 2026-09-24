"""Paired language-model training with exact sampler/RNG/Adam resume state."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from .data import TokenBatcher, verify as verify_data
from .model import MODES, ModelConfig, TransformerLM
from .runtime import (atomic_json, autocast_context, canonical_hash, capture_rng,
                      environment_info, load_checkpoint, resolve_device,
                      resolve_precision, restore_rng, save_checkpoint,
                      seed_everything, sha256_file)


def load_config(path):
    config = json.loads(Path(path).read_text())
    if "model" not in config or "train" not in config:
        raise ValueError("Config requires model and train objects.")
    return config


def build_model(config, mode="standard"):
    return TransformerLM(ModelConfig(**config["model"]), mode=mode)


def make_optimizer(model, config):
    tc = config.get("train", config)
    return torch.optim.AdamW(model.parameters(), lr=float(tc.get("learning_rate", 3e-4)),
                             betas=(float(tc.get("beta1", 0.9)), float(tc.get("beta2", 0.95))),
                             eps=float(tc.get("adam_eps", 1e-8)),
                             weight_decay=float(tc.get("weight_decay", 0.1)),
                             foreach=False, fused=False)


def lr_for_step(step, train_config):
    """Learning rate for zero-indexed update ``step``; horizon never uses cap."""
    total = int(train_config["steps"])
    warmup = int(train_config.get("warmup_steps", 0))
    base = float(train_config.get("learning_rate", 3e-4))
    floor = float(train_config.get("min_lr_ratio", 0.1))
    if warmup and step < warmup:
        return base * (step + 1) / warmup
    fraction = min(1.0, max(0.0, (step - warmup) / max(1, total - warmup - 1)))
    return base * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * fraction)))


def train_step(model, optimizer, batches, train_config, device, amp_dtype, scaler):
    """Execute one real optimizer update, including both Adam moments and AMP.

    Caller supplies the complete microbatch list. The function does not sample
    data or alter the learning rate, making checkpoint forks reproducible.
    """
    if not batches:
        raise ValueError("An optimizer update requires at least one microbatch.")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model.train()
    total_tokens = sum(y.numel() for _, y in batches)
    clip = float(train_config.get("clip_grad_norm", 1.0))
    threshold = clip if clip > 0 else float("inf")
    update_rng = capture_rng()
    for retries in range(9):
        if retries:
            restore_rng(update_rng)
        optimizer.zero_grad(set_to_none=True)
        nll = 0.0
        for x, y in batches:
            with autocast_context(device, amp_dtype):
                logits = model(x)
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
                weighted_loss = loss * (y.numel() / total_tokens)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss; optimizer step was not performed.")
            nll += float(loss.detach()) * y.numel() / total_tokens
            scaler.scale(weighted_loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), threshold,
                                                   error_if_nonfinite=False, foreach=False)
        if torch.isfinite(grad_norm):
            break
        # A counted update must actually occur. Retry the SAME microbatches at
        # a lower AMP scale instead of silently advancing a skipped step.
        if not scaler.is_enabled() or retries == 8:
            optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError("Non-finite gradient after AMP recovery; optimizer step was not performed.")
        scaler.update()
    scaler.step(optimizer)
    scaler.update()
    return {"train_nll": nll, "grad_norm": float(grad_norm),
            "clip_applied": bool(clip > 0 and float(grad_norm) > clip),
            "clip_coefficient": min(1.0, clip / (float(grad_norm) + 1e-6)) if clip > 0 else 1.0,
            "tokens": total_tokens, "loss_scale": float(scaler.get_scale()),
            "amp_overflow_retries": retries,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None}


@torch.no_grad()
def evaluate(model, data_dir, batch_size, seq_len, device, amp_dtype, eval_batches=8):
    """Fixed validation sampler, independent of training RNG and arm."""
    if eval_batches < 1:
        raise ValueError("eval_batches must be positive.")
    batcher = TokenBatcher(data_dir, "validation", batch_size, seq_len, 987654321, device)
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for _ in range(eval_batches):
        x, y = batcher.next()
        with autocast_context(device, amp_dtype):
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="sum")
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite validation loss.")
        total_loss += float(loss)
        total_tokens += y.numel()
    model.train()
    return total_loss / total_tokens


def model_sha256(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _validate_config(config):
    ModelConfig(**config["model"])
    tc = config["train"]
    for name in ("steps", "batch_size", "grad_accum"):
        if int(tc.get(name, 1)) < 1:
            raise ValueError(f"train.{name} must be positive.")
    if int(tc.get("warmup_steps", 0)) < 0 or int(tc.get("warmup_steps", 0)) > int(tc["steps"]):
        raise ValueError("warmup_steps must lie between zero and steps.")
    if not 0 <= float(tc.get("min_lr_ratio", 0.1)) <= 1:
        raise ValueError("min_lr_ratio must lie in [0,1].")
    if float(tc.get("learning_rate", 3e-4)) <= 0:
        raise ValueError("learning_rate must be positive.")


def _append_log(path, record):
    with open(path, "a") as handle:
        handle.write(json.dumps(record, allow_nan=False) + "\n")
        handle.flush()


def train_run(config, data_dir, out, mode="standard", seed=11, resume=None, max_steps=None):
    _validate_config(config)
    if mode not in MODES:
        raise ValueError(f"Invalid mode {mode}")
    data_dir, out = Path(data_dir), Path(out)
    data_manifest = data_dir / "manifest.json"
    if not data_manifest.exists():
        raise FileNotFoundError("Data manifest missing; run qknlab.data before training.")
    manifest = verify_data(data_dir)
    if manifest.get("synthetic", False) and not config.get("study", {}).get("synthetic_allowed", False):
        raise ValueError("Synthetic data are permitted only for explicitly marked smoke tests.")
    if manifest.get("vocab_size") is not None and manifest["vocab_size"] != config["model"]["vocab_size"]:
        raise ValueError("Dataset and model vocab_size differ.")
    out.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = out / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    if resume == "auto":
        resume = checkpoint_dir / "latest.pt" if (checkpoint_dir / "latest.pt").exists() else None
    if resume is None and (out / "run_manifest.json").exists():
        raise FileExistsError(f"Output already contains a run. Use --resume auto: {out}")
    tc = config["train"]
    device = resolve_device(tc.get("device", "auto"))
    precision, amp_dtype = resolve_precision(tc.get("precision", "auto"), device)
    seed_everything(seed, bool(tc.get("deterministic", True)))
    model = build_model(config, mode).to(device)
    init_hash = model_sha256(model)
    optimizer = make_optimizer(model, config)
    scaler = torch.amp.GradScaler("cuda", enabled=(precision == "fp16"),
                                  init_scale=float(tc.get("amp_init_scale", 1024.0)))
    batcher = TokenBatcher(data_dir, "train", int(tc["batch_size"]),
                           config["model"]["max_seq_len"], seed + 100000, device)
    config_hash = canonical_hash(config)
    core_config_hash = canonical_hash({"model": config["model"], "train": config["train"]})
    data_hash = sha256_file(data_manifest)
    start_step = 0
    if resume is not None:
        checkpoint = load_checkpoint(resume, map_location="cpu")
        expected = {"config_sha256": config_hash, "data_manifest_sha256": data_hash,
                    "mode": mode, "seed": seed, "precision": precision}
        for name, value in expected.items():
            if checkpoint.get(name) != value:
                raise ValueError(f"Resume mismatch for {name}: {checkpoint.get(name)!r} != {value!r}.")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        batcher.load_state_dict(checkpoint["batcher_state"])
        start_step = int(checkpoint["step"])
        init_hash = checkpoint["initialization_sha256"]
        restore_rng(checkpoint["rng"])
        # Rewinding a checkpoint truncates later log entries so a resume cannot
        # accidentally count stale updates/evaluations as additional trials.
        logpath = out / "metrics.jsonl"
        if logpath.exists():
            kept = [line for line in logpath.read_text().splitlines()
                    if int(json.loads(line)["step"]) <= start_step]
            logpath.write_text("\n".join(kept) + ("\n" if kept else ""))
    manifest_out = {"seed": seed, "phase": config.get("study", {}).get("phase", config.get("phase", "unspecified")), "arm": mode,
                    "mode": mode, "config_sha256": config_hash,
                    "core_config_sha256": core_config_hash,
                    "data_manifest_sha256": data_hash, "initialization_sha256": init_hash,
                    "config": config, "precision": precision, "device": str(device),
                    "n_parameters": sum(p.numel() for p in model.parameters()),
                    "environment": environment_info()}
    atomic_json(out / "run_manifest.json", manifest_out)
    atomic_json(out / "config.json", config)
    horizon = int(tc["steps"])
    cap = horizon if max_steps is None else min(horizon, int(max_steps))
    if cap < 0:
        raise ValueError("max_steps cannot be negative.")
    tokens_per_step = int(tc["batch_size"]) * int(tc.get("grad_accum", 1)) * config["model"]["max_seq_len"]
    def checkpoint_at(step, keep_numbered=True):
        state = {"schema_version": 1, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "scaler": scaler.state_dict(), "rng": capture_rng(), "batcher_state": batcher.state_dict(),
                 "step": step, "config": config, "model_config": config["model"], "mode": mode,
                 "seed": seed, "precision": precision, "config_sha256": config_hash,
                 "core_config_sha256": core_config_hash,
                 "data_manifest_sha256": data_hash, "initialization_sha256": init_hash}
        path = checkpoint_dir / (f"step_{step:06d}.pt" if keep_numbered else "latest.pt")
        save_checkpoint(path, state)
        if keep_numbered:
            # A second file survives Drive filesystems that do not support
            # symlinks. Recovery intervals keep only the rolling latest file.
            temp = checkpoint_dir / "latest.pt.tmp"
            shutil.copyfile(path, temp)
            temp.replace(checkpoint_dir / "latest.pt")
    if resume is None:
        checkpoint_at(0)
    selected = set(int(x) for x in tc.get("checkpoint_steps", []))
    interval = int(tc.get("checkpoint_interval", 0))
    eval_interval = int(tc.get("eval_interval", 100))
    eval_batches = int(tc.get("eval_batches", 8))
    last_val = None
    final_step = start_step
    for index in range(start_step, cap):
        rate = lr_for_step(index, tc)
        for group in optimizer.param_groups:
            group["lr"] = rate
        batches = [batcher.next() for _ in range(int(tc.get("grad_accum", 1)))]
        if device.type == "cuda":
            torch.cuda.synchronize()
        clock = time.perf_counter()
        result = train_step(model, optimizer, batches, tc, device, amp_dtype, scaler)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - clock
        final_step = index + 1
        result.update(step=final_step, split="train", lr=rate,
                      tokens_seen=final_step * tokens_per_step,
                      tokens_per_second=result["tokens"] / elapsed,
                      update_seconds=elapsed)
        _append_log(out / "metrics.jsonl", result)
        if (eval_interval > 0 and final_step % eval_interval == 0) or final_step == cap:
            last_val = evaluate(model, data_dir, int(tc["batch_size"]),
                                config["model"]["max_seq_len"], device, amp_dtype, eval_batches)
            _append_log(out / "metrics.jsonl", {"step": final_step, "split": "validation",
                        "val_nll": last_val, "tokens_seen": final_step * tokens_per_step})
            print(json.dumps({"step": final_step, "mode": mode, "train_nll": result["train_nll"],
                              "val_nll": last_val, "tokens_per_second": result["tokens_per_second"]}), flush=True)
        if final_step in selected or (interval > 0 and final_step % interval == 0) or final_step == cap:
            checkpoint_at(final_step, keep_numbered=final_step in selected or final_step == horizon)
    if last_val is None:
        last_val = evaluate(model, data_dir, int(tc["batch_size"]),
                            config["model"]["max_seq_len"], device, amp_dtype, eval_batches)
    summary = {"completed_steps": final_step, "configured_steps": horizon,
               "run_complete": final_step >= horizon, "final_validation_nll": last_val,
               "tokens_seen": final_step * tokens_per_step, "mode": mode, "seed": seed,
               "config_sha256": config_hash, "data_manifest_sha256": data_hash,
               "core_config_sha256": core_config_hash,
               "initialization_sha256": init_hash}
    # Include prior resumed updates, and use null for unmeasured CPU runs.
    # Reserved memory includes the CUDA allocator's retained cache. These are
    # training-update peaks, not a measurement of diagnostic/JVP memory use.
    records = []
    metrics_path = out / "metrics.jsonl"
    if metrics_path.exists():
        records = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    for field in ("peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes"):
        values = [record[field] for record in records if record.get(field) is not None]
        summary[field] = max(values) if values else None
    atomic_json(out / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--mode", choices=MODES, default="standard")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--resume", help="Trusted .pt checkpoint path, or auto")
    parser.add_argument("--max-steps", type=int, help="Invocation cap; never changes LR schedule horizon")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], help="Override train.device; recorded in effective config")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.device:
        config["train"]["device"] = args.device
    train_run(config, args.data_dir, args.out, args.mode, args.seed, args.resume, args.max_steps)


if __name__ == "__main__":
    main()
