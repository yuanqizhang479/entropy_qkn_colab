"""Audit and summarize independent, paired seeds; never pool attention rows.

The reader deliberately fails closed on ambiguous cohorts.  A partial CSV is
still useful for auditing an interrupted Colab run, but is not a confidence
interval or a confirmatory result.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t


ARMS = ("standard", "q_detach")
DIAGNOSTIC_METRICS = (
    "endpoint_kl_mean", "shape_energy_fraction",
    "jvp_fisher_relative_rmse", "temperature_fisher_relative_rmse",
)


def mean_interval(values, confidence=0.95):
    """Student t interval across independent seeds, with explicit n=1 behavior.

    These are approximate intervals, especially with three seeds.  They are
    not distribution-free intervals, and bounded metrics are not clipped.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise ValueError("values must be a finite one-dimensional sequence")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    n = len(array)
    result = {"n_independent_seeds": n, "mean": None, "sample_sd": None,
              "standard_error": None, "ci_low": None, "ci_high": None,
              "confidence": confidence, "method": "Student t across seeds"}
    if not n:
        return result
    result["mean"] = float(array.mean())
    if n == 1:
        result["method"] = "one seed: descriptive value only, no interval"
        return result
    sd = float(array.std(ddof=1))
    se = sd / math.sqrt(n)
    half = float(student_t.ppf((1 + confidence) / 2, n - 1)) * se
    result.update(sample_sd=sd, standard_error=se,
                  ci_low=result["mean"] - half, ci_high=result["mean"] + half)
    return result


def paired_interval(standard, q_detach):
    """Return a paired Q-detach minus standard effect, keyed by seed."""
    if set(standard) != set(q_detach):
        raise ValueError("missing paired seeds: both arms need the same seeds")
    seeds = sorted(standard)
    effects = [float(q_detach[s]) - float(standard[s]) for s in seeds]
    return {"contrast": "q_detach_minus_standard", "seed_order": seeds,
            "paired_effects": effects, **mean_interval(effects)}


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def _read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")


def _write_csv(path, rows, fields):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _single(values, label, errors):
    values = set(values)
    if len(values) != 1:
        errors.append(f"{label}: expected one common value, found {len(values)}")
        return None
    return next(iter(values))


def _alpha_one(results):
    matches = [r for r in results if abs(float(r.get("alpha", -1)) - 1.0) < 1e-12]
    if len(matches) != 1:
        raise ValueError("exactly one alpha=1 result is required")
    return matches[0]


def collect_diagnostics(root, phase, expected_seeds, step=None):
    """Read only the common-checkpoint contrast, not layer or head replicates."""
    rows, inputs, errors, warnings = [], [], [], []
    candidates = []
    for path in sorted(Path(root).rglob("*.json")):
        try:
            doc = _read_json(path)
        except (ValueError, OSError) as exc:
            errors.append(f"Cannot read {path}: {exc}")
            continue
        if not isinstance(doc, dict) or "contrasts" not in doc:
            continue
        if doc.get("phase") != phase:
            if "phase" not in doc:
                errors.append(f"{path}: missing phase; cannot assign to pilot or confirm")
            continue
        if step is not None and doc.get("step") != step:
            continue
        candidates.append((path, doc))
    if not candidates:
        return {"status": "not_available", "rows": [], "errors": errors,
                "warnings": ["No matching diagnostic records; no diagnostic inference."],
                "input_files": [], "summaries": {}}
    steps = {doc.get("step") for _, doc in candidates}
    if len(steps) != 1 or None in steps:
        errors.append("Multiple or missing diagnostic steps: select one preregistered --step")
    by_seed = {}
    for path, doc in candidates:
        seed = doc.get("seed")
        if seed not in expected_seeds:
            warnings.append(f"Excluded unrequested seed {seed} at {path}")
            continue
        if seed in by_seed:
            errors.append(f"Duplicate diagnostic seed {seed}; use a narrower runs root")
            continue
        by_seed[seed] = doc
        inputs.append({"path": str(path), "sha256": _hash(path)})
        if doc.get("source_arm") != "standard":
            errors.append(f"Seed {seed}: primary diagnostics require standard-trajectory checkpoint")
        required = ("checkpoint_sha256", "data_manifest_sha256", "core_config_sha256")
        for field in required:
            if not doc.get(field):
                errors.append(f"{path}: missing {field}")
        try:
            contrast = doc["contrasts"]["q_detach_vs_standard"]
            aggregate = _alpha_one(contrast["alpha_results"])["aggregate"]
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"{path}: invalid alpha=1 contrast: {exc}")
            continue
        row = {"seed": seed, "phase": phase, "step": doc.get("step"),
               "checkpoint_sha256": doc.get("checkpoint_sha256"),
               "input_path": str(path)}
        for metric in DIAGNOSTIC_METRICS:
            value = aggregate.get(metric)
            if value is not None and not _finite(value):
                errors.append(f"{path}: {metric} must be finite or null")
                value = None
            row[metric] = value
        if row["endpoint_kl_mean"] is None:
            errors.append(f"{path}: primary endpoint_kl_mean missing")
        full = row["jvp_fisher_relative_rmse"]
        temp = row["temperature_fisher_relative_rmse"]
        row["relative_rmse_improvement_over_temperature"] = (
            1 - full / temp if full is not None and temp is not None and temp > 0 else None)
        row["n_rows_descriptive_only"] = aggregate.get("n_rows", aggregate.get("row_count"))
        rows.append(row)
    missing = sorted(set(expected_seeds) - set(by_seed))
    if missing:
        errors.append(f"Missing diagnostic seeds: {missing}")
    for field in ("data_manifest_sha256", "core_config_sha256"):
        if by_seed:
            _single([d.get(field) for d in by_seed.values()],
                    f"Diagnostic {field}", errors)
    measurement = [_canonical_hash(d.get("measurement")) for d in by_seed.values()]
    if measurement:
        _single(measurement, "Diagnostic measurement specification", errors)
    summaries = {}
    if not errors:
        for metric in (*DIAGNOSTIC_METRICS, "relative_rmse_improvement_over_temperature"):
            values = [r[metric] for r in rows]
            if any(value is None for value in values):
                warnings.append(f"{metric}: undefined for one or more seeds; no pooled interval")
                continue
            summaries[metric] = mean_interval(values)
    return {"status": "invalid_or_incomplete" if errors else "complete",
            "primary_endpoint": "endpoint_kl_mean",
            "unit": "independent seed; within-seed rows pooled by diagnostics only",
            "step": next(iter(steps)) if len(steps) == 1 else None,
            "rows": sorted(rows, key=lambda row: row["seed"]), "summaries": summaries,
            "errors": errors, "warnings": warnings, "input_files": inputs}


def collect_training(root, phase, expected_seeds, step=None, split="validation"):
    """Manifest/metric reader; implementation follows train.py's saved schema."""
    # The package's trainer writes one run_manifest.json per seed and arm.
    rows, inputs, errors, warnings = [], [], [], []
    selected = {}
    for path in sorted(Path(root).rglob("run_manifest.json")):
        try:
            manifest = _read_json(path)
        except (ValueError, OSError) as exc:
            errors.append(f"Cannot read {path}: {exc}")
            continue
        if manifest.get("phase") != phase:
            continue
        seed, arm = manifest.get("seed"), manifest.get("arm")
        if seed not in expected_seeds or arm not in ARMS:
            continue
        key = (seed, arm)
        if key in selected:
            errors.append(f"Duplicate training run for seed {seed}, arm {arm}")
            continue
        selected[key] = (path, manifest)
    if not selected:
        return {"status": "not_available", "rows": [], "errors": errors,
                "warnings": ["No matching paired training runs; no NLL inference."],
                "input_files": [], "summary": None}
    for seed in expected_seeds:
        for arm in ARMS:
            if (seed, arm) not in selected:
                errors.append(f"Missing training pair: seed={seed}, arm={arm}")
    for (seed, arm), (path, manifest) in selected.items():
        for field in ("core_config_sha256", "data_manifest_sha256", "initialization_sha256", "precision"):
            if not manifest.get(field):
                errors.append(f"{path}: missing {field}")
        completion_path = path.parent / "summary.json"
        if not completion_path.exists():
            errors.append(f"{path}: training completion summary missing")
        else:
            completion = _read_json(completion_path)
            inputs.append({"path": str(completion_path), "sha256": _hash(completion_path)})
            if completion.get("run_complete") is not True:
                errors.append(f"{path}: training run is incomplete")
            if completion.get("completed_steps") != completion.get("configured_steps"):
                errors.append(f"{path}: completed and configured steps differ")
        metrics_path = path.parent / "metrics.jsonl"
        if not metrics_path.exists():
            errors.append(f"Missing metrics: {metrics_path}")
            continue
        inputs.extend({"path": str(p), "sha256": _hash(p)} for p in (path, metrics_path))
        measurements = {}
        for lineno, line in enumerate(metrics_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except ValueError as exc:
                errors.append(f"{metrics_path}:{lineno}: {exc}")
                continue
            value = record.get(f"{split}_nll")
            if value is None and split == "validation":
                value = record.get("val_nll")
            if value is None and record.get("split") == split:
                value = record.get("nll")
            if value is None:
                continue
            current_step = record.get("step")
            if not isinstance(current_step, int) or not _finite(value):
                errors.append(f"{metrics_path}:{lineno}: invalid step or NLL")
                continue
            if current_step in measurements and measurements[current_step] != float(value):
                errors.append(f"{metrics_path}: conflicting evaluation at step {current_step}")
            measurements[current_step] = float(value)
        chosen_step = step if step is not None else max(measurements, default=None)
        if chosen_step is None or chosen_step not in measurements:
            errors.append(f"{metrics_path}: no {split} evaluation at requested/final step {chosen_step}")
            continue
        rows.append({"seed": seed, "phase": phase, "arm": arm, "step": chosen_step,
                     "split": split, "nll": measurements[chosen_step], "input_path": str(path)})
    if rows:
        _single([r["step"] for r in rows], "Training evaluation step", errors)
    for field in ("core_config_sha256", "data_manifest_sha256", "precision"):
        _single([m.get(field) for _, m in selected.values()], f"Training {field}", errors)
    for seed in expected_seeds:
        pair = [selected.get((seed, arm)) for arm in ARMS]
        if all(pair):
            _single([item[1].get("initialization_sha256") for item in pair],
                    f"Seed {seed} initial state", errors)
    summary = None
    if not errors:
        standard = {r["seed"]: r["nll"] for r in rows if r["arm"] == "standard"}
        q_detach = {r["seed"]: r["nll"] for r in rows if r["arm"] == "q_detach"}
        summary = paired_interval(standard, q_detach)
        summary["interpretation"] = "Positive means Q-detach has higher (worse) NLL."
    return {"status": "invalid_or_incomplete" if errors else "complete",
            "unit": "independent paired seed", "split": split,
            "rows": sorted(rows, key=lambda row: (row["seed"], row["arm"])),
            "summary": summary, "errors": errors, "warnings": warnings, "input_files": inputs}


def collect_evaluations(root, phase, expected_seeds, step=None):
    """Pair frozen-checkpoint test or external-domain evaluation JSON files."""
    selected, rows, inputs, errors, warnings = {}, [], [], [], []
    for path in sorted(Path(root).rglob("*.json")):
        try:
            doc = _read_json(path)
        except (ValueError, OSError) as exc:
            errors.append(f"Cannot read {path}: {exc}")
            continue
        if not isinstance(doc, dict) or "nll_nats_per_token" not in doc:
            continue
        if doc.get("phase") != phase:
            if "phase" not in doc:
                errors.append(f"{path}: evaluation missing phase")
            continue
        seed, arm = doc.get("seed"), doc.get("mode")
        if seed not in expected_seeds or arm not in ARMS:
            continue
        if step is not None and doc.get("checkpoint_step") != step:
            continue
        key = (seed, arm)
        if key in selected:
            errors.append(f"Duplicate evaluation seed={seed}, arm={arm}; use one dataset/split/step per directory")
            continue
        selected[key] = doc
        inputs.append({"path": str(path), "sha256": _hash(path)})
        for field in ("phase", "core_config_sha256", "initialization_sha256",
                      "checkpoint_sha256", "data_manifest_sha256", "token_file_sha256",
                      "train_data_manifest_sha256", "context_length", "protocol", "dtype",
                      "configured_training_steps", "source_training_precision"):
            if doc.get(field) is None:
                errors.append(f"{path}: evaluation missing {field}")
        if doc.get("checkpoint_step") != doc.get("configured_training_steps"):
            errors.append(f"{path}: held-out primary evaluation must use the full-budget checkpoint")
        value = doc.get("nll_nats_per_token")
        if not _finite(value):
            errors.append(f"{path}: evaluation NLL must be finite")
            continue
        count = doc.get("n_scored_tokens")
        if not isinstance(count, int) or count < 1:
            errors.append(f"{path}: invalid scored token count")
        if not doc.get("full_split"):
            warnings.append(f"{path}: evaluation uses a fixed prefix, not the full split")
        rows.append({"seed": seed, "phase": phase, "arm": arm,
                     "step": doc.get("checkpoint_step"), "split": doc.get("split"),
                     "n_scored_tokens": count, "nll": value, "input_path": str(path)})
    if not selected:
        return {"status": "not_available", "rows": [], "errors": errors,
                "warnings": ["No matching frozen-checkpoint evaluation JSON files."],
                "input_files": [], "summary": None}
    for seed in expected_seeds:
        for arm in ARMS:
            if (seed, arm) not in selected:
                errors.append(f"Missing evaluation pair: seed={seed}, arm={arm}")
    for field in ("core_config_sha256", "data_manifest_sha256", "token_file_sha256",
                  "train_data_manifest_sha256", "context_length", "protocol", "dtype",
                  "checkpoint_step", "configured_training_steps", "source_training_precision",
                  "split", "n_scored_tokens", "full_split"):
        _single([doc.get(field) for doc in selected.values()], f"Evaluation {field}", errors)
    for seed in expected_seeds:
        pair = [selected.get((seed, arm)) for arm in ARMS]
        if all(pair):
            _single([doc.get("initialization_sha256") for doc in pair],
                    f"Evaluation seed {seed} initialization", errors)
    summary = None
    if not errors:
        standard = {r["seed"]: r["nll"] for r in rows if r["arm"] == "standard"}
        q_detach = {r["seed"]: r["nll"] for r in rows if r["arm"] == "q_detach"}
        summary = paired_interval(standard, q_detach)
        summary["interpretation"] = "Positive means Q-detach has higher (worse) NLL."
    return {"status": "invalid_or_incomplete" if errors else "complete",
            "unit": "independent paired seed; token-weighted evaluation within a seed",
            "rows": sorted(rows, key=lambda row: (row["seed"], row["arm"])),
            "summary": summary, "errors": errors, "warnings": warnings, "input_files": inputs}


def make_figures(report, out):
    """Draw observed seeds and approximate seed-level intervals only."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    train = report["training"]
    if train["rows"]:
        fig, ax = plt.subplots(figsize=(6.2, 4.0), constrained_layout=True)
        for seed in report["expected_seeds"]:
            rows = {r["arm"]: r for r in train["rows"] if r["seed"] == seed}
            if set(rows) == set(ARMS):
                ax.plot([0, 1], [rows[a]["nll"] for a in ARMS], marker="o", label=str(seed))
        ax.set_xticks([0, 1], ["Standard QK RMS", "Q-detach"])
        ax.set_ylabel(f"{train['split'].capitalize()} NLL (nats/token)")
        ax.set_title(f"{report['phase']}: paired seeds ({train['status']})")
        ax.legend(title="Seed", loc="best", fontsize=8)
        fig.savefig(out / "paired_nll.png", dpi=220)
        fig.savefig(out / "paired_nll.pdf")
        plt.close(fig)
    diag = report["diagnostics"]
    if diag["rows"]:
        metrics = [("endpoint_kl_mean", "Attention endpoint KL (nats)"),
                   ("shape_energy_fraction", "First-order shape energy fraction"),
                   ("jvp_fisher_relative_rmse", "Full JVP relative Fisher RMSE"),
                   ("temperature_fisher_relative_rmse", "Temperature-only relative Fisher RMSE")]
        fig, axes = plt.subplots(2, 2, figsize=(8, 6), constrained_layout=True)
        for ax, (metric, label) in zip(axes.flat, metrics):
            pairs = [(r["seed"], r.get(metric)) for r in diag["rows"] if r.get(metric) is not None]
            if pairs:
                ax.scatter(range(len(pairs)), [value for _, value in pairs], color="tab:blue")
                ax.set_xticks(range(len(pairs)), [str(seed) for seed, _ in pairs])
                result = diag["summaries"].get(metric)
                if result:
                    ax.axhline(result["mean"], color="black", linewidth=1)
                    if result["ci_low"] is not None:
                        ax.axhspan(result["ci_low"], result["ci_high"], color="gray", alpha=.15)
            ax.set_ylabel(label)
            ax.set_xlabel("Independent seed")
        fig.suptitle(f"{report['phase']}: actual step alpha=1 ({diag['status']})")
        fig.savefig(out / "diagnostic_seeds.png", dpi=220)
        fig.savefig(out / "diagnostic_seeds.pdf")
        plt.close(fig)
    evaluation = report.get("evaluations")
    if evaluation and evaluation["rows"]:
        fig, ax = plt.subplots(figsize=(6.2, 4.0), constrained_layout=True)
        for seed in report["expected_seeds"]:
            pair = {r["arm"]: r for r in evaluation["rows"] if r["seed"] == seed}
            if set(pair) == set(ARMS):
                ax.plot([0, 1], [pair[a]["nll"] for a in ARMS], marker="o", label=str(seed))
        ax.set_xticks([0, 1], ["Standard QK RMS", "Q-detach"])
        ax.set_ylabel("Frozen-checkpoint evaluation NLL (nats/token)")
        ax.set_title(f"{report['phase']}: held-out paired seeds ({evaluation['status']})")
        ax.legend(title="Seed", loc="best", fontsize=8)
        fig.savefig(out / "paired_evaluation_nll.png", dpi=220)
        fig.savefig(out / "paired_evaluation_nll.pdf")
        plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--evaluations-root", type=Path,
                        help="Optional one-dataset/split directory of qknlab.evaluate JSON results")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--phase", choices=("pilot", "confirm", "robustness"), default="pilot")
    parser.add_argument("--expected-seeds", default="11,22,33")
    parser.add_argument("--step", type=int, help="One predeclared step for both training and diagnostics")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    seeds = [int(seed.strip()) for seed in args.expected_seeds.split(",") if seed.strip()]
    if not seeds or len(set(seeds)) != len(seeds):
        parser.error("--expected-seeds must contain unique integer seeds")
    if not args.runs_root.is_dir():
        parser.error("--runs-root is not a directory")
    args.out.mkdir(parents=True, exist_ok=True)
    train = collect_training(args.runs_root, args.phase, seeds, args.step, "validation")
    diag = collect_diagnostics(args.runs_root, args.phase, seeds, args.step)
    errors = train["errors"] + diag["errors"]
    evaluations = None
    if args.evaluations_root:
        if not args.evaluations_root.is_dir():
            parser.error("--evaluations-root is not a directory")
        evaluations = collect_evaluations(args.evaluations_root, args.phase, seeds, args.step)
        errors.extend(evaluations["errors"])
        if evaluations["status"] == "not_available":
            errors.append("Requested evaluations root has no matching evaluation records")
    if train["status"] == diag["status"] == "not_available" and not evaluations:
        errors.append("No matching training or diagnostic records")
    report = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
              "phase": args.phase, "expected_seeds": seeds, "requested_step": args.step,
              "status": "invalid_or_incomplete" if errors else "complete",
              "caution": ("Pilot and robustness results are exploratory. Confirmatory interpretation additionally "
                          "requires a frozen protocol and fresh seeds. All intervals are approximate "
                          "Student t intervals across independent seeds, never tokens or attention rows."),
              "training": train, "diagnostics": diag, "evaluations": evaluations, "errors": errors}
    _write_json(args.out / "summary.json", report)
    _write_csv(args.out / "training_seeds.csv", train["rows"],
               ["seed", "phase", "arm", "step", "split", "nll", "input_path"])
    _write_csv(args.out / "diagnostic_seeds.csv", diag["rows"],
               ["seed", "phase", "step", *DIAGNOSTIC_METRICS,
                "relative_rmse_improvement_over_temperature", "n_rows_descriptive_only",
                "checkpoint_sha256", "input_path"])
    if evaluations:
        _write_csv(args.out / "evaluation_seeds.csv", evaluations["rows"],
                   ["seed", "phase", "arm", "step", "split", "n_scored_tokens", "nll", "input_path"])
    if not args.no_plots:
        make_figures(report, args.out)
    print(json.dumps({"status": report["status"], "out": str(args.out), "errors": errors},
                     ensure_ascii=False, indent=2))
    return 2 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
