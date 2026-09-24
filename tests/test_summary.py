import json
import math
from pathlib import Path

import pytest

from qknlab.summarize import (
    collect_diagnostics, collect_evaluations, collect_training, mean_interval, paired_interval,
)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def make_run(root, seed, arm, value, *, step=10, data_hash="data", phase="pilot", complete=True):
    run = Path(root) / phase / f"seed{seed}" / arm
    write_json(run / "run_manifest.json", {
        "seed": seed, "phase": phase, "arm": arm, "precision": "fp32",
        "data_manifest_sha256": data_hash, "core_config_sha256": "core",
        "initialization_sha256": f"init{seed}",
    })
    write_json(run / "summary.json", {
        "run_complete": complete, "completed_steps": step, "configured_steps": 10,
    })
    (run / "metrics.jsonl").write_text(json.dumps({
        "step": step, "split": "validation", "val_nll": value,
    }) + "\n")
    return run


def make_diagnostic(root, seed, *, step=10, core="core", source_arm="standard"):
    path = Path(root) / f"diagnostic_seed{seed}_step{step}.json"
    write_json(path, {
        "phase": "pilot", "seed": seed, "step": step, "source_arm": source_arm,
        "checkpoint_sha256": f"checkpoint{seed}", "core_config_sha256": core,
        "data_manifest_sha256": "data", "measurement": {"split": "validation", "batch_size": 1},
        "contrasts": {"q_detach_vs_standard": {"alpha_results": [{
            "alpha": 1.0, "aggregate": {
                "endpoint_kl_mean": 0.001 * seed, "shape_energy_fraction": 0.2,
                "jvp_fisher_relative_rmse": 0.1, "temperature_fisher_relative_rmse": 0.5,
                "n_rows": 128,
            },
        }] }},
    })
    return path


def test_t_interval_known_values():
    result = mean_interval([1.0, 2.0, 3.0])
    assert result["mean"] == 2
    assert result["sample_sd"] == 1
    assert result["ci_low"] == pytest.approx(-0.484137711719546, abs=1e-10)
    assert result["ci_high"] == pytest.approx(4.484137711719546, abs=1e-10)


def test_single_seed_is_descriptive_only():
    result = mean_interval([3.0])
    assert result["n_independent_seeds"] == 1
    assert result["mean"] == 3
    assert result["ci_low"] is None
    assert result["standard_error"] is None


def test_pairing_not_unpaired_spread():
    result = paired_interval({11: 3, 22: 8, 33: 12}, {11: 3.25, 22: 8.25, 33: 12.25})
    assert result["mean"] == 0.25
    assert result["ci_low"] == result["ci_high"] == 0.25
    with pytest.raises(ValueError, match="missing paired seeds"):
        paired_interval({11: 1}, {22: 2})


def test_complete_paired_training(tmp_path):
    for seed, base in [(11, 3.0), (22, 4.0), (33, 5.0)]:
        make_run(tmp_path, seed, "standard", base)
        make_run(tmp_path, seed, "q_detach", base + 0.5)
    result = collect_training(tmp_path, "pilot", [11, 22, 33])
    assert result["status"] == "complete", result["errors"]
    assert result["summary"]["mean"] == 0.5
    assert result["summary"]["n_independent_seeds"] == 3


def test_missing_pair_suppresses_interval(tmp_path):
    make_run(tmp_path, 11, "standard", 3)
    result = collect_training(tmp_path, "pilot", [11])
    assert result["status"] == "invalid_or_incomplete"
    assert result["summary"] is None
    assert any("Missing training pair" in error for error in result["errors"])


def test_mismatched_data_suppresses_interval(tmp_path):
    make_run(tmp_path, 11, "standard", 3, data_hash="data_one")
    make_run(tmp_path, 11, "q_detach", 3, data_hash="data_two")
    result = collect_training(tmp_path, "pilot", [11])
    assert result["summary"] is None
    assert any("data_manifest_sha256" in error for error in result["errors"])


def test_incomplete_runs_are_not_terminal_results(tmp_path):
    make_run(tmp_path, 11, "standard", 3)
    make_run(tmp_path, 11, "q_detach", 3, step=5, complete=False)
    result = collect_training(tmp_path, "pilot", [11])
    assert result["summary"] is None
    assert any("incomplete" in error for error in result["errors"])
    assert any("evaluation step" in error for error in result["errors"])


def test_diagnostic_seed_unit_and_actual_alpha_one(tmp_path):
    for seed in [11, 22, 33]:
        make_diagnostic(tmp_path, seed)
    result = collect_diagnostics(tmp_path, "pilot", [11, 22, 33], step=10)
    assert result["status"] == "complete", result["errors"]
    assert result["summaries"]["endpoint_kl_mean"]["n_independent_seeds"] == 3
    assert result["summaries"]["relative_rmse_improvement_over_temperature"]["mean"] == pytest.approx(.8)


def test_diagnostics_do_not_pool_steps(tmp_path):
    make_diagnostic(tmp_path, 11, step=10)
    make_diagnostic(tmp_path, 11, step=20)
    result = collect_diagnostics(tmp_path, "pilot", [11])
    assert result["status"] == "invalid_or_incomplete"
    assert result["summaries"] == {}
    selected = collect_diagnostics(tmp_path, "pilot", [11], step=10)
    assert selected["status"] == "complete"


def test_diagnostic_config_or_source_mismatch_rejected(tmp_path):
    make_diagnostic(tmp_path, 11)
    make_diagnostic(tmp_path, 22, core="different", source_arm="q_detach")
    result = collect_diagnostics(tmp_path, "pilot", [11, 22], step=10)
    assert result["summaries"] == {}
    assert any("standard-trajectory" in error for error in result["errors"])
    assert any("core_config" in error for error in result["errors"])


def make_evaluation(root, seed, arm, value, *, token_count=1000, step=10):
    path = Path(root) / f"eval_{seed}_{arm}.json"
    write_json(path, {
        "phase": "confirm", "seed": seed, "mode": arm, "nll_nats_per_token": value,
        "checkpoint_step": step, "configured_training_steps": 10,
        "checkpoint_sha256": f"checkpoint-{seed}-{arm}", "core_config_sha256": "core",
        "initialization_sha256": f"init{seed}", "data_manifest_sha256": "evaldata",
        "token_file_sha256": "tokens", "train_data_manifest_sha256": "traindata",
        "context_length": 256, "protocol": "nonoverlapping_targets_block_reset_gpt2_bpe",
        "dtype": "float32", "source_training_precision": "bf16",
        "split": "test", "n_scored_tokens": token_count, "full_split": False,
    })


def test_frozen_evaluation_uses_paired_seed_not_tokens(tmp_path):
    for seed in [101, 202]:
        make_evaluation(tmp_path, seed, "standard", 5)
        make_evaluation(tmp_path, seed, "q_detach", 5.2)
    result = collect_evaluations(tmp_path, "confirm", [101, 202], step=10)
    assert result["status"] == "complete", result["errors"]
    assert result["summary"]["n_independent_seeds"] == 2
    assert result["summary"]["mean"] == pytest.approx(.2)
    assert len(result["warnings"]) == 4  # Prefix protocol must be visible.


def test_frozen_evaluation_rejects_different_token_counts(tmp_path):
    make_evaluation(tmp_path, 101, "standard", 5, token_count=1000)
    make_evaluation(tmp_path, 101, "q_detach", 5, token_count=500)
    result = collect_evaluations(tmp_path, "confirm", [101], step=10)
    assert result["summary"] is None
    assert any("n_scored_tokens" in error for error in result["errors"])


def test_frozen_evaluation_rejects_nonterminal_checkpoint(tmp_path):
    make_evaluation(tmp_path, 101, "standard", 5, step=5)
    make_evaluation(tmp_path, 101, "q_detach", 5, step=5)
    result = collect_evaluations(tmp_path, "confirm", [101], step=5)
    assert result["summary"] is None
    assert any("full-budget" in error for error in result["errors"])
