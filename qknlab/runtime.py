"""Reproducibility, checkpoint and device utilities."""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def seed_everything(seed: int, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def capture_rng():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda") and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable. In Colab select a GPU runtime.")
    return device


def resolve_precision(requested: str, device: torch.device):
    if device.type != "cuda":
        if requested not in ("auto", "fp32"):
            raise ValueError("CPU runs use fp32. Select precision auto or fp32.")
        return "fp32", None
    if requested == "auto":
        requested = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    mapping = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}
    if requested not in mapping:
        raise ValueError("precision must be auto, fp32, fp16, or bf16")
    if requested == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("This GPU does not support bf16; use auto or fp16.")
    return requested, mapping[requested]


def autocast_context(device, dtype):
    return torch.autocast(device_type=device.type, dtype=dtype) if dtype else contextlib.nullcontext()


def atomic_json(path: str | Path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(temp, path)


def save_checkpoint(path: str | Path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    torch.save(state, temp)
    os.replace(temp, path)


def load_checkpoint(path: str | Path, map_location="cpu"):
    # Checkpoints include Python/NumPy RNG state; load only trusted files made
    # by this package, never arbitrary untrusted pickled checkpoint files.
    return torch.load(path, map_location=map_location, weights_only=False)


def environment_info():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
                                         text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
            "numpy": np.__version__, "cuda_build": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "git_commit": commit,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}
