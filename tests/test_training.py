"""Scientific invariants: same forward, changed backward, causal map, resume."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from qknlab.data import prepare
from qknlab.model import ModelConfig, TransformerLM, qk_rms_normalize
from qknlab.runtime import load_checkpoint
from qknlab.train import train_run


@pytest.fixture(autouse=True)
def small_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_detach_preserves_forward_and_changes_radial_gradient():
    torch.manual_seed(4)
    x = torch.randn(2, 3, 8, dtype=torch.float64, requires_grad=True)
    g = torch.randn_like(x)
    eps = 1e-5
    y = qk_rms_normalize(x, eps)
    detached = qk_rms_normalize(x, eps, detach=True)
    assert torch.equal(y, detached)
    gs, = torch.autograd.grad((y * g).sum(), x, retain_graph=True)
    gd, = torch.autograd.grad((detached * g).sum(), x)
    s = (x.square().mean(-1, keepdim=True) + eps).sqrt()
    expected_difference = x * (x * g).sum(-1, keepdim=True) / (x.shape[-1] * s.pow(3))
    torch.testing.assert_close(gd - gs, expected_difference, atol=2e-15, rtol=2e-14)


def test_model_modes_forward_identity_and_causality():
    torch.manual_seed(8)
    model = TransformerLM(ModelConfig(vocab_size=32, max_seq_len=8, d_model=16, n_heads=2, n_layers=2)).double()
    x = torch.randint(0, 32, (2, 8))
    standard = model(x, return_attn=True)
    for mode in ("q_detach", "k_detach", "both_detach"):
        model.set_mode(mode)
        changed = model(x, return_attn=True)
        assert torch.equal(standard["logits"], changed["logits"])
        assert torch.equal(standard["attn_logits"], changed["attn_logits"])
    assert standard["attn_logits"].shape == (2, 2, 2, 8, 8)
    assert torch.isfinite(standard["attn_logits"]).all()
    assert torch.equal(model(x, return_attn="only"), standard["attn_logits"])
    changed_input = x.clone()
    changed_input[:, 4:] = (changed_input[:, 4:] + 1) % 32
    torch.testing.assert_close(model(x)[:, :4], model(changed_input)[:, :4], rtol=0, atol=0)


def test_standard_forward_jvp_matches_finite_difference():
    torch.manual_seed(19)
    model = TransformerLM(ModelConfig(vocab_size=16, max_seq_len=5, d_model=8, n_heads=2, n_layers=1)).double()
    x = torch.randint(0, 16, (1, 5))
    params = dict(model.named_parameters())
    direction = {name: torch.randn_like(p) * 0.001 for name, p in params.items()}
    def f(p):
        return torch.func.functional_call(model, p, (x,), {"return_attn": "only"})
    _, tangent = torch.func.jvp(f, (params,), (direction,))
    epsilon = 1e-4
    plus = f({k: p + epsilon * direction[k] for k, p in params.items()})
    minus = f({k: p - epsilon * direction[k] for k, p in params.items()})
    torch.testing.assert_close(tangent, (plus - minus) / (2 * epsilon), rtol=3e-7, atol=2e-9)


def _setup_data(tmp_path):
    data_dir = tmp_path / "data"
    args = SimpleNamespace(out=str(data_dir), force=False, synthetic=True,
                           synthetic_vocab_size=32, max_train_tokens=512,
                           max_eval_tokens=128, dataset="wikitext103")
    prepare(args)
    return data_dir


def _config():
    return {"model": {"vocab_size": 32, "max_seq_len": 8, "d_model": 16,
                      "n_heads": 2, "n_layers": 1, "mlp_ratio": 2, "norm_eps": 1e-5},
            "train": {"steps": 4, "batch_size": 2, "grad_accum": 2,
                      "learning_rate": 0.001, "weight_decay": 0.1, "beta1": 0.9,
                      "beta2": 0.95, "adam_eps": 1e-8, "clip_grad_norm": 1.0,
                      "warmup_steps": 1, "eval_interval": 2, "eval_batches": 2,
                      "checkpoint_steps": [0, 2, 4], "precision": "fp32", "device": "cpu"},
            "study": {"phase": "smoke", "synthetic_allowed": True}}


def _assert_nested_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            _assert_nested_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for av, bv in zip(a, b):
            _assert_nested_equal(av, bv)
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b)
    else:
        assert a == b


def test_training_resume_reproduces_model_adam_and_next_sampler(tmp_path):
    data = _setup_data(tmp_path)
    config = _config()
    whole, partial = tmp_path / "whole", tmp_path / "partial"
    train_run(config, data, whole, "q_detach", seed=7)
    train_run(config, data, partial, "q_detach", seed=7, max_steps=2)
    result = train_run(config, data, partial, "q_detach", seed=7, resume="auto")
    assert result["run_complete"] is True
    a = load_checkpoint(whole / "checkpoints/latest.pt")
    b = load_checkpoint(partial / "checkpoints/latest.pt")
    for key in ("model", "optimizer", "batcher_state", "rng", "scaler"):
        _assert_nested_equal(a[key], b[key])
    a_logs = [json.loads(line) for line in (whole / "metrics.jsonl").read_text().splitlines()]
    b_logs = [json.loads(line) for line in (partial / "metrics.jsonl").read_text().splitlines()]
    a_losses = [(x["step"], x["train_nll"]) for x in a_logs if x["split"] == "train"]
    b_losses = [(x["step"], x["train_nll"]) for x in b_logs if x["split"] == "train"]
    assert a_losses == b_losses
    with pytest.raises(ValueError, match="Resume mismatch"):
        train_run(config, data, partial, "q_detach", seed=8, resume="auto")


def test_paired_arms_share_initialization_and_reject_bad_data(tmp_path):
    data = _setup_data(tmp_path)
    config = _config()
    left, right = tmp_path / "standard", tmp_path / "detached"
    train_run(config, data, left, "standard", 4, max_steps=0)
    train_run(config, data, right, "q_detach", 4, max_steps=0)
    a = load_checkpoint(left / "checkpoints/latest.pt")
    b = load_checkpoint(right / "checkpoints/latest.pt")
    _assert_nested_equal(a["model"], b["model"])
    _assert_nested_equal(a["batcher_state"], b["batcher_state"])
    assert a["initialization_sha256"] == b["initialization_sha256"]
    invalid = copy.deepcopy(config)
    invalid["study"]["synthetic_allowed"] = False
    with pytest.raises(ValueError, match="Synthetic"):
        train_run(invalid, data, tmp_path / "invalid")
    raw = bytearray((data / "train.bin").read_bytes())
    raw[0] ^= 1
    (data / "train.bin").write_bytes(raw)
    with pytest.raises(ValueError, match="checksum"):
        train_run(config, data, tmp_path / "corrupted")
