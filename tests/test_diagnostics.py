import torch
from qknlab.diagnostics import (
    actual_update_jvp, attention_metrics, causal_distribution, stable_attention_kl,
)


def test_kl_agrees_with_direct_formula_and_ignores_masked_keys():
    torch.manual_seed(42)
    z = torch.randn(2, 2, 2, 7, 7, dtype=torch.float64)
    endpoint = z + torch.randn_like(z) * 0.4
    p, lp = causal_distribution(z)
    _, lq = causal_distribution(endpoint)
    expected = (p * (lp - lq)).sum(-1)
    actual = stable_attention_kl(z, endpoint)
    assert torch.allclose(actual, expected, atol=2e-15, rtol=2e-12)
    mask = torch.ones(7, 7, dtype=torch.bool).triu(1)
    corrupted = endpoint.masked_fill(mask, 1e5)
    assert torch.equal(actual, stable_attention_kl(z, corrupted))
    assert actual[..., 0].abs().max().item() == 0.0


def test_tiny_kl_is_nonnegative_and_matches_fisher():
    torch.manual_seed(9)
    z = torch.randn(1, 1, 2, 5, 5, dtype=torch.float64)
    direction = torch.randn_like(z)
    p, _ = causal_distribution(z)
    centered = direction - (p * direction).sum(-1, keepdim=True)
    for epsilon in [1e-4, 1e-6, 1e-8]:
        kl = stable_attention_kl(z, z + epsilon * direction)
        fisher = 0.5 * epsilon ** 2 * (p * centered.square()).sum(-1)
        assert (kl >= 0).all()
        assert torch.allclose(kl[..., 1:], fisher[..., 1:], rtol=2e-4, atol=1e-29)


def test_fisher_temperature_decomposition_and_degenerate_rows():
    torch.manual_seed(6)
    z = torch.randn(2, 1, 2, 5, 5, dtype=torch.float64)
    v = torch.randn_like(z) * 0.03
    result = attention_metrics(z, v, v, endpoint_base=z, endpoint_target=z + v)
    aggregate = result["aggregate"]
    assert aggregate["jvp_fisher_relative_rmse"] == 0.0
    assert abs(aggregate["shape_energy_fraction"] + aggregate["temperature_energy_fraction"] - 1) < 1e-12
    assert aggregate["row_count"] == 2 * 1 * 2 * 4
    degenerate = attention_metrics(torch.zeros_like(z), torch.zeros_like(z), torch.zeros_like(z))
    assert degenerate["aggregate"]["temperature_identifiable_row_fraction"] == 0.0
    assert degenerate["aggregate"]["temperature_beta_mean_identifiable"] is None
    assert degenerate["aggregate"]["jvp_fisher_relative_rmse"] is None


class _ToyAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.wq = torch.nn.Parameter(torch.randn(4, 4, dtype=torch.float64))
        self.wk = torch.nn.Parameter(torch.randn(4, 4, dtype=torch.float64))
        self.mode = "q_detach"

    def set_mode(self, mode):
        self.mode = mode

    def forward(self, x, return_attn=False):
        q, k = x @ self.wq, x @ self.wk
        sq = (q.square().mean(-1, keepdim=True) + 1e-6).sqrt()
        if self.mode == "q_detach":
            sq = sq.detach()
        q = q / sq
        k = k / (k.square().mean(-1, keepdim=True) + 1e-6).sqrt()
        attn = (q @ k.transpose(-1, -2)).unsqueeze(0).unsqueeze(2)
        return attn if return_attn == "only" else {"attn_logits": attn}


def test_jvp_uses_true_numerical_forward_and_all_parameter_update():
    torch.manual_seed(13)
    model = _ToyAttention()
    x = torch.randn(1, 5, 4, dtype=torch.float64)
    delta = {name: torch.randn_like(p) * 0.02 for name, p in model.named_parameters()}
    z, jvp, function, params, tangents = actual_update_jvp(model, x, delta)
    assert model.mode == "standard"
    epsilon = 1e-4
    plus = function({name: p + epsilon * tangents[name] for name, p in params.items()})
    minus = function({name: p - epsilon * tangents[name] for name, p in params.items()})
    finite_difference = (plus - minus) / (2 * epsilon)
    assert torch.allclose(jvp, finite_difference, atol=2e-9, rtol=2e-7)
    assert z.shape == (1, 1, 1, 5, 5)


def test_actual_optimizer_state_changes_matched_fork_result():
    """A fresh Adam instance is observably different from the checkpoint's Adam."""
    torch.manual_seed(21)
    start = torch.nn.Parameter(torch.tensor([0.8, -0.4], dtype=torch.float64))
    optimizer = torch.optim.AdamW([start], lr=0.03, betas=(0.8, 0.95), weight_decay=0.01)
    for gradient in [torch.tensor([0.3, 0.2]), torch.tensor([-0.1, 0.5])]:
        start.grad = gradient.to(start.dtype)
        optimizer.step()
    checkpoint = optimizer.state_dict()
    def update(with_state):
        p = torch.nn.Parameter(start.detach().clone())
        opt = torch.optim.AdamW([p], lr=0.03, betas=(0.8, 0.95), weight_decay=0.01)
        if with_state:
            import copy
            opt.load_state_dict(copy.deepcopy(checkpoint))
        p.grad = torch.tensor([0.02, -0.2], dtype=p.dtype)
        opt.step()
        return p.detach()
    assert torch.equal(update(True), update(True))
    assert not torch.allclose(update(True), update(False))


def test_full_checkpoint_fork_preserves_state_and_replays(tmp_path):
    """Exercise data -> mature Adam checkpoint -> detached fork -> full JVP."""
    from argparse import Namespace
    import copy
    from qknlab.data import prepare
    from qknlab.train import train_run
    from qknlab.runtime import load_checkpoint
    from qknlab.diagnostics import _actual_checkpoint_update, run_diagnostics

    data = tmp_path / "data"
    prepare(Namespace(out=str(data), force=False, synthetic=True, dataset="wikitext103",
                      synthetic_vocab_size=16, max_train_tokens=512, max_eval_tokens=128))
    config = {
        "model": {"vocab_size": 16, "max_seq_len": 8, "d_model": 8, "n_heads": 2,
                  "n_layers": 2, "mlp_ratio": 2, "norm_eps": 1e-5},
        "train": {"steps": 4, "batch_size": 2, "grad_accum": 2, "learning_rate": 3e-4,
                  "precision": "fp32", "device": "cpu", "eval_batches": 1,
                  "eval_interval": 2, "checkpoint_steps": [2], "deterministic": True},
        "study": {"phase": "smoke", "synthetic_allowed": True},
    }
    train_run(config, data, tmp_path / "run", mode="standard", seed=11, max_steps=2)
    path = tmp_path / "run" / "checkpoints" / "step_000002.pt"
    checkpoint = load_checkpoint(path)
    original_optimizer = copy.deepcopy(checkpoint["optimizer"])
    standard, standard_log = _actual_checkpoint_update(checkpoint, data, "standard", torch.device("cpu"))
    detached, detached_log = _actual_checkpoint_update(checkpoint, data, "q_detach", torch.device("cpu"))
    assert standard_log["training_batch_sha256"] == detached_log["training_batch_sha256"]
    assert standard_log["optimizer_state_entries_before_step"] > 0
    assert any(not torch.equal(standard[name], detached[name]) for name in standard)
    for key, state in original_optimizer["state"].items():
        for name, value in state.items():
            assert torch.equal(value, checkpoint["optimizer"]["state"][key][name])
    result = run_diagnostics(Namespace(
        checkpoint=path, data_dir=data, batch_size=1, seq_len=8,
        measurement_split="validation", measurement_seed=92017,
        device="cpu", precision="float64", arms=["standard", "q_detach"], alphas=[0.5, 1.0],
    ))
    assert result["replay"]["bitwise_parameter_update_identical"]
    assert result["replay"]["endpoint_kl_mean"] == 0.0
    assert result["phase"] == "smoke"
    contrast = result["contrasts"]["q_detach_vs_standard"]["alpha_results"]
    assert contrast[1]["aggregate"]["endpoint_kl_mean"] > 0
    assert contrast[0]["aggregate"]["jvp_fisher_relative_rmse"] < contrast[1]["aggregate"]["jvp_fisher_relative_rmse"]
