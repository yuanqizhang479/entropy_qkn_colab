"""CUDA branch acceptance tests; explicitly SKIPPED on CPU-only machines.

Running the complete pytest suite in a Colab GPU session executes these tests.
They check real mixed-precision updates and Adam-state replay, not throughput.
"""
import copy
import json
from types import SimpleNamespace

import pytest
import torch

from qknlab.data import prepare
from qknlab.diagnostics import _actual_checkpoint_update
from qknlab.model import ModelConfig, TransformerLM
from qknlab.runtime import load_checkpoint, seed_everything
from qknlab.train import make_optimizer, train_run, train_step


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a real CUDA GPU; not CPU-validated")


def _cuda_config(precision):
    return {"model": {"vocab_size": 32, "max_seq_len": 8, "d_model": 16,
                      "n_heads": 2, "n_layers": 1, "mlp_ratio": 2, "norm_eps": 1e-5},
            "train": {"steps": 2, "batch_size": 2, "grad_accum": 2,
                      "learning_rate": 0.001, "weight_decay": 0.1, "beta1": 0.9,
                      "beta2": 0.95, "adam_eps": 1e-8, "clip_grad_norm": 1.0,
                      "warmup_steps": 0, "eval_interval": 2, "eval_batches": 1,
                      "checkpoint_steps": [0, 2], "precision": precision,
                      "device": "cuda", "amp_init_scale": 128.0, "deterministic": True},
            "study": {"phase": "smoke", "synthetic_allowed": True}}


@pytest.mark.parametrize("precision", ["auto", "fp16"])
def test_cuda_mixed_precision_actual_update_and_adam_replay(tmp_path, precision):
    """Auto selects bf16 on capable GPUs; fp16 explicitly exercises GradScaler."""
    data = tmp_path / "data"
    prepare(SimpleNamespace(out=str(data), force=False, synthetic=True,
                            synthetic_vocab_size=32, max_train_tokens=512,
                            max_eval_tokens=128, dataset="wikitext103"))
    config = _cuda_config(precision)
    out = tmp_path / "run"
    result = train_run(config, data, out, "q_detach", seed=29)
    checkpoint = load_checkpoint(out / "checkpoints/latest.pt")
    assert result["run_complete"] and result["completed_steps"] == 2
    assert result["peak_cuda_allocated_bytes"] > 0
    assert result["peak_cuda_reserved_bytes"] >= result["peak_cuda_allocated_bytes"]
    assert checkpoint["precision"] in ("fp16", "bf16")
    if precision == "fp16":
        assert checkpoint["scaler"]["scale"] > 0
    assert checkpoint["optimizer"]["state"]
    assert all(float(state["step"]) == 2 for state in checkpoint["optimizer"]["state"].values())
    device = torch.device("cuda")
    first, first_log = _actual_checkpoint_update(checkpoint, data, "q_detach", device)
    second, second_log = _actual_checkpoint_update(checkpoint, data, "q_detach", device)
    assert first_log["training_batch_sha256"] == second_log["training_batch_sha256"]
    assert first_log["parameter_update_l2"] > 0
    assert all(torch.isfinite(value).all() for value in first.values())
    assert all(torch.equal(first[name], second[name]) for name in first)
    # Forks must not mutate the checkpoint Adam state shared by later arms.
    assert all(float(state["step"]) == 2 for state in checkpoint["optimizer"]["state"].values())
    logs = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
    assert all(row["peak_cuda_allocated_bytes"] > 0 for row in logs if row["split"] == "train")


def test_cuda_fp16_overflow_retries_same_batch_without_skipped_update():
    """Inject one backward Inf; recovery must equal one clean lower-scale step."""
    seed_everything(47, deterministic=True)
    device = torch.device("cuda")
    config = _cuda_config("fp16")
    tc = config["train"]
    retried = TransformerLM(ModelConfig(**config["model"]), mode="q_detach").to(device)
    clean = copy.deepcopy(retried)
    retried_optimizer = make_optimizer(retried, config)
    clean_optimizer = make_optimizer(clean, config)
    x = torch.randint(0, 32, (2, 8), device=device)
    y = torch.randint(0, 32, (2, 8), device=device)
    batches = [(x, y)]
    injected = {"count": 0}

    def one_time_inf(gradient):
        injected["count"] += 1
        return torch.full_like(gradient, float("inf")) if injected["count"] == 1 else gradient

    handle = retried.token_embedding.weight.register_hook(one_time_inf)
    retry_scaler = torch.amp.GradScaler("cuda", init_scale=128.0, growth_interval=1000)
    clean_scaler = torch.amp.GradScaler("cuda", init_scale=64.0, growth_interval=1000)
    try:
        retry_log = train_step(retried, retried_optimizer, batches, tc, device, torch.float16, retry_scaler)
    finally:
        handle.remove()
    clean_log = train_step(clean, clean_optimizer, batches, tc, device, torch.float16, clean_scaler)
    assert retry_log["amp_overflow_retries"] == 1
    assert clean_log["amp_overflow_retries"] == 0
    assert retry_log["loss_scale"] == clean_log["loss_scale"] == 64.0
    assert retry_log["train_nll"] == clean_log["train_nll"]
    assert all(torch.equal(p, dict(clean.named_parameters())[name]) for name, p in retried.named_parameters())
    assert all(float(state["step"]) == 1 for state in retried_optimizer.state.values())
