#!/usr/bin/env python3
"""Sequential, resumable paired experiments with a frozen confirmation protocol.

Only trusted local checkpoints should be loaded. This script launches child
commands with argument lists (no shell interpolation).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def source_fingerprint():
    """Hash executable experiment code, dependency declaration and protocol text."""
    paths = list((ROOT / "qknlab").rglob("*.py")) + list((ROOT / "scripts").rglob("*.py"))
    paths += [ROOT / "requirements.txt"]
    paths += [ROOT / "configs" / "pg19_test_sources.json"]
    paths += [ROOT / "docs" / name for name in ("EXPERIMENTS.md", "ANALYSIS.md", "DATA.md", "METRICS.md")]
    rows = {str(p.relative_to(ROOT)): sha_file(p) for p in sorted(set(paths)) if p.is_file()}
    return {"sha256": hashlib.sha256(canonical(rows)).hexdigest(), "files": rows}


def get_git_commit():
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True)
    return result.stdout.strip() if result.returncode == 0 else None


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=["smoke", "pilot", "confirm", "robustness"], default="pilot")
    p.add_argument("--config", type=Path)
    p.add_argument("--data-dir", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path, help="One unique output directory per protocol/configuration")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--seeds", nargs="+", type=int, help="Full planned seed set; changes require a new protocol")
    p.add_argument("--run-seeds", nargs="+", type=int, help="Execute only these members of the planned seed set")
    p.add_argument("--arms", nargs="+", choices=["standard", "q_detach", "k_detach", "both_detach"], default=["standard", "q_detach"])
    p.add_argument("--run-arms", nargs="+", choices=["standard", "q_detach", "k_detach", "both_detach"])
    p.add_argument("--diagnostic-trajectories", nargs="+", default=["standard"], choices=["standard", "q_detach", "k_detach", "both_detach"])
    p.add_argument("--diagnostic-arms", nargs="+", default=["standard", "q_detach"], choices=["standard", "q_detach", "k_detach", "both_detach"])
    p.add_argument("--tasks", nargs="+", choices=["train", "diagnostics"], default=["train", "diagnostics"])
    p.add_argument("--max-steps", type=int, help="Execution cap only; does not change the full training schedule")
    p.add_argument("--protocol", type=Path, help="Frozen JSON protocol; required to execute confirmation")
    p.add_argument("--freeze-only", action="store_true", help="Write a new protocol after reviewing pilot; do not run experiments")
    p.add_argument("--dry-run", action="store_true", help="Print commands without launching jobs or writing run outputs")
    return p.parse_args(argv)


def make_spec(args, config, manifest_path):
    phase = config.get("study", {}).get("phase")
    if phase != args.stage:
        raise ValueError(f"Config study.phase={phase!r} does not match --stage {args.stage!r}")
    seeds = args.seeds or config.get("study", {}).get("seeds")
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Provide a nonempty set of unique planned seeds")
    if len(args.arms) != len(set(args.arms)):
        raise ValueError("Duplicate planned arms")
    if not {"standard", "q_detach"}.issubset(args.arms):
        raise ValueError("The primary paired study requires standard and q_detach arms")
    if not set(args.diagnostic_trajectories).issubset(args.arms):
        raise ValueError("Diagnostic trajectories must be included in the planned training arms")
    if not {"standard", "q_detach"}.issubset(args.diagnostic_arms):
        raise ValueError("Diagnostic arms must include standard and q_detach")
    study = config.get("study", {})
    return {
        "schema_version": 1,
        "stage": args.stage,
        "configuration": config,
        "configuration_sha256": hashlib.sha256(canonical(config)).hexdigest(),
        "data_manifest_sha256": sha_file(manifest_path),
        "source": source_fingerprint(),
        "seeds": seeds,
        "arms": args.arms,
        "diagnostic_trajectories": args.diagnostic_trajectories,
        "diagnostic_arms": args.diagnostic_arms,
        "diagnostic_steps": config["train"]["checkpoint_steps"],
        "diagnostic_batch_size": study.get("measurement_batch_size", 1),
        "diagnostic_seq_len": study.get("measurement_seq_len", min(128, config["model"]["max_seq_len"])),
        "alphas": [0.125, 0.25, 0.5, 1.0],
        "measurement_split": "validation",
        "diagnostic_precision": "float32",
        "device_policy": args.device,
        "primary_endpoint": study.get("primary_endpoint"),
        "inference_unit": "independent seed; checkpoints, layers, heads and rows are repeated measurements",
    }


def check_or_freeze_protocol(args, spec):
    digest = hashlib.sha256(canonical(spec)).hexdigest()
    if args.freeze_only:
        if args.protocol is None:
            raise ValueError("--freeze-only requires --protocol PATH")
        frozen = {"protocol_sha256": digest, "frozen_utc": datetime.now(timezone.utc).isoformat(),
                  "git_commit": get_git_commit(), "specification": spec}
        if args.protocol.exists():
            old = json.loads(args.protocol.read_text(encoding="utf-8"))
            if old.get("protocol_sha256") != digest or old.get("specification") != spec:
                raise ValueError("Refusing to replace a different frozen protocol; use a new versioned filename")
            print(f"Existing identical protocol: {args.protocol} ({digest})", flush=True)
        else:
            write_json(args.protocol, frozen)
            print(f"Frozen protocol: {args.protocol} ({digest})", flush=True)
        return digest
    if args.stage == "confirm" and args.protocol is None:
        raise ValueError("Confirmation requires a reviewed frozen --protocol; first run the same command with --freeze-only")
    if args.protocol is not None:
        frozen = json.loads(args.protocol.read_text(encoding="utf-8"))
        if frozen.get("protocol_sha256") != digest or frozen.get("specification") != spec:
            raise ValueError("Frozen protocol mismatch: code, data, config, seeds or diagnostics changed. Keep old results separate; explicitly freeze a new protocol version.")
    return digest


def launch(command, dry_run):
    print(shlex.join([str(part) for part in command]), flush=True)
    if not dry_run:
        subprocess.run([str(part) for part in command], cwd=ROOT, check=True)


def main(argv=None):
    args = parse_args(argv)
    if args.stage == "robustness" and args.config is None:
        raise ValueError("Robustness requires an explicit --config configs/robustness_*.json")
    args.config = (args.config or ROOT / "configs" / f"{args.stage}.json").resolve()
    args.data_dir = args.data_dir.resolve()
    args.out = args.out.resolve()
    if args.protocol:
        args.protocol = args.protocol.resolve()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    manifest = args.data_dir / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"Prepare and verify data first; missing {manifest}")
    spec = make_spec(args, config, manifest)
    run_seeds = args.run_seeds or spec["seeds"]
    run_arms = args.run_arms or spec["arms"]
    if not set(run_seeds).issubset(spec["seeds"]):
        raise ValueError("--run-seeds must be a subset of the planned --seeds")
    if not set(run_arms).issubset(spec["arms"]):
        raise ValueError("--run-arms must be a subset of the planned --arms")
    digest = check_or_freeze_protocol(args, spec)
    if args.freeze_only:
        return
    suite_manifest = args.out / "suite_manifest.json"
    if suite_manifest.exists():
        previous = json.loads(suite_manifest.read_text(encoding="utf-8"))
        if previous.get("protocol_sha256") != digest:
            raise ValueError("Output directory belongs to a different protocol. Choose a new --out directory.")
    elif not args.dry_run:
        write_json(suite_manifest, {"protocol_sha256": digest, "git_commit": get_git_commit(), "specification": spec})
    if not args.dry_run:
        launch([sys.executable, "-m", "qknlab.data", "verify", "--data-dir", args.data_dir], False)

    total = config["train"]["steps"]
    cap = min(total, args.max_steps) if args.max_steps is not None else total
    if cap < 0:
        raise ValueError("--max-steps must be nonnegative")
    for seed in run_seeds:
        for arm in run_arms:
            run = args.out / "runs" / f"seed_{seed}" / arm
            if "train" in args.tasks:
                command = [sys.executable, "-m", "qknlab.train", "--config", args.config,
                           "--data-dir", args.data_dir, "--out", run, "--mode", arm,
                           "--seed", str(seed), "--device", args.device, "--resume", "auto"]
                if args.max_steps is not None:
                    command += ["--max-steps", str(args.max_steps)]
                # The trainer validates identity before resuming and verifies
                # completion from the actual checkpoint, not a filename alone.
                launch(command, args.dry_run)
            if "diagnostics" not in args.tasks or arm not in spec["diagnostic_trajectories"]:
                continue
            for step in spec["diagnostic_steps"]:
                if step > cap:
                    continue
                checkpoint = run / "checkpoints" / f"step_{step:06d}.pt"
                output = args.out / "diagnostics" / f"seed_{seed}" / arm / f"diagnostic_step_{step:06d}.json"
                sidecar = output.with_suffix(".suite_provenance.json")
                command = [sys.executable, "-m", "qknlab.diagnostics", "--checkpoint", checkpoint,
                           "--data-dir", args.data_dir, "--out", output,
                           "--batch-size", str(spec["diagnostic_batch_size"]),
                           "--seq-len", str(spec["diagnostic_seq_len"]), "--device", args.device,
                           "--precision", spec["diagnostic_precision"], "--measurement-split", "validation",
                           "--arms", *spec["diagnostic_arms"], "--alphas", *map(str, spec["alphas"])]
                if args.dry_run:
                    launch(command, True)
                    continue
                if not checkpoint.is_file():
                    raise FileNotFoundError(f"Missing planned checkpoint {checkpoint}; train this trajectory first")
                expected = {"protocol_sha256": digest, "checkpoint_sha256": sha_file(checkpoint),
                            "seed": seed, "trajectory": arm, "step": step}
                if output.exists() and sidecar.exists():
                    previous = json.loads(sidecar.read_text(encoding="utf-8"))
                    actual = json.loads(output.read_text(encoding="utf-8"))
                    if previous != expected:
                        raise ValueError(f"Existing diagnostic provenance mismatch: {output}")
                    if actual.get("checkpoint_sha256") != expected["checkpoint_sha256"]:
                        raise ValueError(f"Existing diagnostic checkpoint hash mismatch: {output}")
                    print(f"Verified completed diagnostic; skip {output}", flush=True)
                    continue
                launch(command, False)
                actual = json.loads(output.read_text(encoding="utf-8"))
                if actual.get("checkpoint_sha256") != expected["checkpoint_sha256"]:
                    raise ValueError(f"Diagnostic result does not identify its source checkpoint: {output}")
                write_json(sidecar, expected)
    print(f"Suite execution finished. Protocol SHA256: {digest}", flush=True)


if __name__ == "__main__":
    main()
