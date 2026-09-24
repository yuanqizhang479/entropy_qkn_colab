"""Matched actual-optimizer-state forks and numerical-forward attention diagnostics.

The detached-RMS backward is deliberately NEVER used to obtain a forward JVP.
See docs/METRICS.md for estimands, weighting, and limits of interpretation.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def causal_distribution(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return float64 p/logp; masked entries in logp are zero for safe products."""
    z = logits.double()
    rows, cols = z.shape[-2:]
    if rows != cols:
        raise ValueError("Diagnostics require square, fixed causal attention masks.")
    mask = torch.ones(rows, cols, device=z.device, dtype=torch.bool).tril()
    lp = torch.log_softmax(z.masked_fill(~mask, -torch.inf), dim=-1)
    return lp.exp(), lp.masked_fill(~mask, 0.0)


def weighted_center(value: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    return value.double() - (p * value.double()).sum(-1, keepdim=True)


def stable_attention_kl(base: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """KL(p(base)||p(target)) per row; causal masked keys contribute exactly zero.

    Center the logit increment and use expm1(x)-x for small increments. This
    prevents a false negative KL from cancellation in logsumexp(z+u)-logsumexp(z).
    A logsumexp fallback handles unusually large finite increments.
    """
    p, lp = causal_distribution(base)
    u = target.double() - base.double()
    x = weighted_center(u, p)
    valid = p > 0
    x = torch.where(valid, x, torch.zeros_like(x))
    small = x.abs() < 1e-3
    polynomial = x.square() * (0.5 + x * (1 / 6 + x * (1 / 24 + x * (1 / 120 + x / 720))))
    # Clamp only the unused expm1 branch; the large-increment result uses LSE.
    remainder = torch.where(small, polynomial, torch.expm1(x.clamp(-700, 700)) - x)
    second = (p * remainder).sum(-1)
    mean = (p * x).sum(-1)
    # log(1+mean+second)-mean, stable even when second is of order 1e-30.
    mean_remainder = torch.where(
        mean.abs() < 1e-3,
        mean.square() * (-0.5 + mean * (1 / 3 + mean * (-0.25 + mean / 5))),
        torch.log1p(mean) - mean,
    )
    result = torch.log1p(second / (1 + mean)) + mean_remainder
    lse = torch.logsumexp(torch.where(valid, lp + x, -torch.inf), -1) - mean
    result = torch.where(x.abs().amax(-1) <= 40, result, lse)
    # Roundoff only: materially negative results are an implementation failure.
    if not torch.isfinite(result).all() or (result < -1e-12).any():
        raise FloatingPointError("Invalid attention KL: nonfinite or materially negative.")
    return result.clamp_min(0.0)


def _rows_mean(value: torch.Tensor) -> float:
    return float(value[..., 1:].mean().item())


def _safe_relative(numerator: float, denominator: float) -> float | None:
    return math.sqrt(numerator / denominator) if denominator > 1e-28 else None


@torch.no_grad()
def attention_metrics(
    baseline: torch.Tensor,
    increment: torch.Tensor,
    prediction: torch.Tensor,
    *,
    endpoint_base: torch.Tensor | None = None,
    endpoint_target: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Fisher-weighted per-row diagnostics, plus equal-row pooled summaries.

    Inputs have [layer,batch,head,query,key] shape. The first causal row (one
    allowed key) is omitted. Temperature is fitted to the PREDICTED increment;
    endpoint information is never used to fit the temperature comparator.
    """
    if baseline.ndim != 5 or baseline.shape != increment.shape or baseline.shape != prediction.shape:
        raise ValueError("Expected equally shaped [layer,batch,head,query,key] tensors.")
    if baseline.shape[-1] < 2:
        raise ValueError("At least two tokens are required for attention diagnostics.")
    z, u, v = baseline.double(), increment.double(), prediction.double()
    p, lp = causal_distribution(z)
    zc, uc, vc = (weighted_center(x, p) for x in (z, u, v))
    varz = (p * zc.square()).sum(-1)
    varu = (p * uc.square()).sum(-1)
    varv = (p * vc.square()).sum(-1)
    covzv = (p * zc * vc).sum(-1)
    identifiable = varz > 1e-24
    beta = torch.where(identifiable, covzv / varz.clamp_min(1e-24), 0.0)
    vtemp = beta.unsqueeze(-1) * zc
    vshape = vc - vtemp
    error = (p * (uc - vc).square()).sum(-1)
    temp_error = (p * (uc - vtemp).square()).sum(-1)
    shape = (p * vshape.square()).sum(-1)
    temp_energy = (p * vtemp.square()).sum(-1)
    predicted_dh = -covzv
    p_end, lp_end = causal_distribution(z + u)
    exact_dh = -(p_end * lp_end).sum(-1) + (p * lp).sum(-1)
    endpoint_kl = None
    if (endpoint_base is None) != (endpoint_target is None):
        raise ValueError("Provide both endpoint tensors or neither.")
    if endpoint_base is not None:
        endpoint_kl = stable_attention_kl(endpoint_base, endpoint_target)
        p_start, lp_start = causal_distribution(endpoint_base)
        p_finish, lp_finish = causal_distribution(endpoint_target)
        exact_dh = -(p_finish * lp_finish).sum(-1) + (p_start * lp_start).sum(-1)

    def summarize(index: int | None) -> dict[str, Any]:
        def select(x: torch.Tensor) -> torch.Tensor:
            return x if index is None else x[index:index + 1]
        def avg(x: torch.Tensor) -> float:
            return _rows_mean(select(x))
        vu, vv, err, terr = avg(varu), avg(varv), avg(error), avg(temp_error)
        shaped, temped = avg(shape), avg(temp_energy)
        mask = select(identifiable)[..., 1:]
        betas = select(beta)[..., 1:][mask]
        summary: dict[str, Any] = {
            "row_count": int(select(varu)[..., 1:].numel()),
            "weighting": "equal_valid_causal_rows; first_query_omitted",
            "exact_fisher_energy": vu,
            "predicted_fisher_energy": vv,
            "jvp_fisher_rmse": math.sqrt(err),
            "jvp_fisher_relative_rmse": _safe_relative(err, vu),
            "temperature_fisher_rmse": math.sqrt(terr),
            "temperature_fisher_relative_rmse": _safe_relative(terr, vu),
            "shape_energy_fraction": shaped / vv if vv > 1e-28 else None,
            "temperature_energy_fraction": temped / vv if vv > 1e-28 else None,
            "temperature_identifiable_row_fraction": float(mask.double().mean().item()),
            "temperature_beta_mean_identifiable": float(betas.mean().item()) if betas.numel() else None,
            "entropy_change_mean": avg(exact_dh),
            "predicted_entropy_change_mean": avg(predicted_dh),
            "entropy_prediction_rmse": math.sqrt(avg((exact_dh - predicted_dh).square())),
            "jvp_vs_temperature_rmse_reduction": 1 - math.sqrt(err / terr) if terr > 1e-28 else None,
        }
        if endpoint_kl is not None:
            summary["endpoint_kl_mean"] = avg(endpoint_kl)
        return summary

    return {"aggregate": summarize(None), "layers": [{"layer": i, **summarize(i)} for i in range(z.shape[0])]}


def numerical_attention_function(model: torch.nn.Module, ids: torch.Tensor):
    """Build a functional NUMERICAL forward, overriding detached backward modes."""
    model.set_mode("standard")
    model.eval()
    buffers = dict(model.named_buffers())

    def function(params: dict[str, torch.Tensor]) -> torch.Tensor:
        output = torch.func.functional_call(model, (params, buffers), (ids,), {"return_attn": "only"})
        attn = output["attn_logits"] if isinstance(output, dict) else output
        return torch.stack(attn) if isinstance(attn, (list, tuple)) else attn

    return function


def actual_update_jvp(model: torch.nn.Module, ids: torch.Tensor, delta: dict[str, torch.Tensor]):
    function = numerical_attention_function(model, ids)
    params = {name: p.detach() for name, p in model.named_parameters()}
    tangents = {name: delta[name].to(device=p.device, dtype=p.dtype) for name, p in params.items()}
    with torch.no_grad():
        baseline, prediction = torch.func.jvp(function, (params,), (tangents,))
    return baseline.detach(), prediction.detach(), function, params, tangents


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _actual_checkpoint_update(checkpoint, data_dir, mode, device):
    """Replay a saved optimizer update; checkpoint tensors are never mutated."""
    from .data import TokenBatcher
    from .model import ModelConfig, TransformerLM
    from .runtime import resolve_precision, restore_rng
    from .train import lr_for_step, make_optimizer, train_step

    config, tc = checkpoint["config"], checkpoint["config"]["train"]
    precision, amp_dtype = resolve_precision(checkpoint["precision"], device)
    model = TransformerLM(ModelConfig(**checkpoint["model_config"]), mode=mode).to(device)
    model.load_state_dict(checkpoint["model"])
    optimizer = make_optimizer(model, config)
    # Without deepcopy, CPU Adam state tensors can alias and corrupt subsequent
    # forks (including their step counter). This is a scientific correctness issue.
    optimizer.load_state_dict(copy.deepcopy(checkpoint["optimizer"]))
    rate = lr_for_step(int(checkpoint["step"]), tc)
    for group in optimizer.param_groups:
        group["lr"] = rate
    scaler = torch.amp.GradScaler("cuda", enabled=(precision == "fp16"),
                                 init_scale=float(tc.get("amp_init_scale", 1024.0)))
    scaler.load_state_dict(copy.deepcopy(checkpoint["scaler"]))
    batcher = TokenBatcher(data_dir, "train", int(tc["batch_size"]),
                           int(checkpoint["model_config"]["max_seq_len"]),
                           int(checkpoint["seed"]) + 100000, device)
    batcher.load_state_dict(copy.deepcopy(checkpoint["batcher_state"]))
    batches = [batcher.next() for _ in range(int(tc.get("grad_accum", 1)))]
    # Restore after model initialization. Initializing a clone must not advance
    # the stochastic state of its replayed update.
    restore_rng(checkpoint["rng"])
    log = train_step(model, optimizer, batches, tc, device, amp_dtype, scaler)
    delta = {name: parameter.detach().cpu().double() - checkpoint["model"][name].cpu().double()
             for name, parameter in model.named_parameters()}
    log.update({"learning_rate": rate, "precision": precision,
                "parameter_update_l2": math.sqrt(sum(float(d.square().sum()) for d in delta.values())),
                "n_parameters": sum(d.numel() for d in delta.values()),
                "optimizer_state_entries_before_step": len(checkpoint["optimizer"]["state"]),
                "training_microbatches": len(batches)})
    digest = hashlib.sha256()
    for x, y in batches:
        digest.update(x.detach().cpu().contiguous().numpy().tobytes())
        digest.update(y.detach().cpu().contiguous().numpy().tobytes())
    log["training_batch_sha256"] = digest.hexdigest()
    del optimizer, model, batches, scaler, batcher
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return delta, log


def run_diagnostics(args) -> dict[str, Any]:
    from .data import TokenBatcher, verify
    from .model import MODES, ModelConfig, TransformerLM
    from .runtime import canonical_hash, environment_info, load_checkpoint, resolve_device, seed_everything

    if args.batch_size < 1 or args.seq_len < 2:
        raise ValueError("Measurement batch_size >= 1 and seq_len >= 2 are required.")
    if len(set(args.alphas)) != len(args.alphas) or any(not math.isfinite(a) or a <= 0 for a in args.alphas):
        raise ValueError("Alphas must be distinct, positive finite numbers.")
    if 1.0 not in args.alphas:
        raise ValueError("Include alpha=1: the actual update is a required diagnostic.")
    arms = list(dict.fromkeys(["standard", *args.arms]))
    if any(mode not in MODES for mode in arms):
        raise ValueError(f"Arms must be members of {MODES}.")
    device = resolve_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    mandatory = ["model", "optimizer", "scaler", "rng", "batcher_state", "precision",
                 "model_config", "config", "step", "mode", "seed", "data_manifest_sha256"]
    if any(name not in checkpoint for name in mandatory):
        raise ValueError("Incomplete checkpoint: optimizer, RNG, sampler and precision state are all required.")
    data_hash = file_sha256(Path(args.data_dir) / "manifest.json")
    if data_hash != checkpoint["data_manifest_sha256"]:
        raise ValueError("The data manifest does not match this checkpoint.")
    verify(args.data_dir)
    if args.seq_len > checkpoint["model_config"]["max_seq_len"]:
        raise ValueError("Measurement sequence exceeds the model's trained maximum context.")
    seed_everything(int(checkpoint["seed"]), bool(checkpoint["config"]["train"].get("deterministic", True)))
    measurement = TokenBatcher(args.data_dir, args.measurement_split, args.batch_size,
                               args.seq_len, args.measurement_seed, device)
    ids, _ = measurement.next()
    updates, step_logs = {}, {}
    for arm in arms:
        updates[arm], step_logs[arm] = _actual_checkpoint_update(checkpoint, args.data_dir, arm, device)
    replay_delta, replay_log = _actual_checkpoint_update(checkpoint, args.data_dir, "standard", device)
    replay_difference = {name: replay_delta[name] - updates["standard"][name] for name in replay_delta}
    replay = {
        "parameter_max_abs_difference": max(float(d.abs().max()) for d in replay_difference.values()),
        "parameter_l2_difference": math.sqrt(sum(float(d.square().sum()) for d in replay_difference.values())),
        "training_batch_identical": replay_log["training_batch_sha256"] == step_logs["standard"]["training_batch_sha256"],
        "bitwise_parameter_update_identical": all(torch.equal(replay_delta[n], updates["standard"][n]) for n in replay_delta),
    }
    if not replay["training_batch_identical"]:
        raise RuntimeError("Matched replay produced a different training batch.")
    if len({log["training_batch_sha256"] for log in step_logs.values()}) != 1:
        raise RuntimeError("Forks did not consume identical training batches.")
    diagnostic_dtype = torch.float64 if args.precision == "float64" else torch.float32
    model = TransformerLM(ModelConfig(**checkpoint["model_config"]), mode="standard")
    model.load_state_dict(checkpoint["model"])
    model.to(device=device, dtype=diagnostic_dtype)
    function = numerical_attention_function(model, ids)
    params = {name: p.detach() for name, p in model.named_parameters()}
    with torch.no_grad():
        baseline = function(params).detach()
    predictions, endpoints, arm_results = {}, {}, {}
    for arm in arms:
        delta = {name: updates[arm][name].to(device=device, dtype=p.dtype) for name, p in params.items()}
        with torch.no_grad():
            _, prediction = torch.func.jvp(function, (params,), (delta,))
        predictions[arm] = prediction.detach()
        endpoints[arm], values = {}, []
        for alpha in args.alphas:
            # Form interpolation in float64 before casting. At alpha=1, FP32
            # evaluation uses the EXACT updated FP32 parameters, not rounded
            # delta addition that might perturb an endpoint by another ULP.
            endpoint_params = {
                name: (checkpoint["model"][name].cpu().double() + alpha * updates[arm][name]).to(device=device, dtype=p.dtype)
                for name, p in params.items()
            }
            with torch.no_grad():
                endpoint = function(endpoint_params).detach()
            endpoints[arm][alpha] = endpoint
            metrics = attention_metrics(baseline, endpoint - baseline, alpha * prediction,
                                        endpoint_base=baseline, endpoint_target=endpoint)
            values.append({"alpha": alpha, **metrics})
            del endpoint_params
        arm_results[arm] = {"step": step_logs[arm], "alpha_results": values}
        del delta
    # Translate the optimizer replay floor into the same observable as the
    # primary arm contrast. Standard replay uses the exact updated parameters.
    replay_params = {name: (checkpoint["model"][name].cpu().double() + replay_delta[name]).to(device=device, dtype=p.dtype)
                     for name, p in params.items()}
    with torch.no_grad():
        replay_endpoint = function(replay_params).detach()
    replay["attention_logit_max_abs_difference"] = float((replay_endpoint - endpoints["standard"][1.0]).abs().max())
    replay["endpoint_kl_mean"] = _rows_mean(stable_attention_kl(endpoints["standard"][1.0], replay_endpoint))
    contrast_results = {}
    for arm in arms:
        if arm == "standard":
            continue
        # Subtract parameter updates in float64 BEFORE the JVP. Subtracting
        # two nearly equal float32 JVP outputs can introduce a false floor.
        contrast_delta = {name: (updates[arm][name] - updates["standard"][name]).to(device=device, dtype=p.dtype)
                          for name, p in params.items()}
        with torch.no_grad():
            _, contrast_prediction = torch.func.jvp(function, (params,), (contrast_delta,))
        numerical_linearity_difference = float((contrast_prediction - (predictions[arm] - predictions["standard"])).abs().max())
        values = []
        for alpha in args.alphas:
            start, finish = endpoints["standard"][alpha], endpoints[arm][alpha]
            metrics = attention_metrics(baseline, finish - start,
                                        alpha * contrast_prediction,
                                        endpoint_base=start, endpoint_target=finish)
            values.append({"alpha": alpha, **metrics})
        contrast_results[f"{arm}_vs_standard"] = {
            "prediction_method": "direct JVP of float64-subtracted parameter updates",
            "jvp_linearity_max_abs_discrepancy": numerical_linearity_difference,
            "alpha_results": values,
        }
        del contrast_delta, contrast_prediction
    result = {
        "schema_version": 1,
        "seed": int(checkpoint["seed"]), "step": int(checkpoint["step"]),
        "phase": checkpoint["config"].get("study", {}).get("phase", checkpoint["config"].get("phase", "unspecified")),
        "source_arm": checkpoint["mode"],
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "config_sha256": checkpoint.get("config_sha256", canonical_hash(checkpoint["config"])),
        "core_config_sha256": checkpoint.get("core_config_sha256", canonical_hash(checkpoint["config"])),
        "data_manifest_sha256": data_hash,
        "measurement": {"split": args.measurement_split, "batch_size": args.batch_size,
                        "seq_len": args.seq_len, "sampling_seed": args.measurement_seed,
                        "batch_sha256": hashlib.sha256(ids.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
                        "precision": args.precision, "attention_temperature": "fixed_1/sqrt(head_dim)",
                        "mask": "causal", "first_query_excluded": True},
        "actual_update_precision": checkpoint["precision"],
        "alpha_meaning": "post-update parameter interpolation; not an optimizer learning-rate rerun",
        "jvp_meaning": "all-parameter true numerical forward in standard mode",
        "replay": replay, "arms": arm_results, "contrasts": contrast_results,
        "environment": environment_info(),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="Atomic JSON result file")
    parser.add_argument("--batch-size", type=int, default=1, help="Measurement batch only")
    parser.add_argument("--seq-len", type=int, default=128, help="Measurement sequence only")
    parser.add_argument("--measurement-split", choices=["validation", "test"], default="validation")
    parser.add_argument("--measurement-seed", type=int, default=92017)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", choices=["float32", "float64"], default="float32")
    parser.add_argument("--arms", nargs="+", default=["standard", "q_detach"])
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.125, 0.25, 0.5, 1.0])
    args = parser.parse_args()
    result = run_diagnostics(args)
    _atomic_json(args.out, result)
    print(json.dumps({"saved": str(args.out), "seed": result["seed"], "step": result["step"], "replay": result["replay"]}))


if __name__ == "__main__":
    main()
