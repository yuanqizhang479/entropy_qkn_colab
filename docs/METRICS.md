# Actual-state attention diagnostics

## What is being tested

The primary intervention replaces the query RMS denominator's backward derivative by a detached denominator. Standard and query-detached arms have **identical numerical forward maps at identical parameters**. They can have different gradients, Adam moments, clipping coefficients, and next parameter updates.

For each saved checkpoint, diagnostics reconstruct **the complete saved AdamW state**, loss-scaler state, RNG, and next training microbatches. The same checkpoint is forked into standard and query-detached updates. Standard is replayed twice to measure a numerical replay floor. No optimizer is restarted. The code logs both clipping and actual update norms, so an observed difference must not automatically be attributed to an isolated query projection: upstream weights, both Adam moments, and global clipping can participate.

The checkpoint step counts completed training updates. The diagnostic applies the next scheduled update. At the last planned training checkpoint this is a counterfactual extra update at the configured LR floor; it is not included in training NLL curves. A checkpoint at step zero has no accumulated Adam moments and must be identified as initialization, not mature optimizer-state evidence.

## Forward derivative and finite endpoints

Let the common saved parameters be theta, and let d_a be the **actual parameter update** from arm a. Let z(theta) denote the unmasked attention logits under the ordinary numerical forward. We calculate:

```
v_a = J_z(theta) d_a
u_a(alpha) = z(theta + alpha d_a) - z(theta)
```

`torch.func.jvp` is evaluated with **standard** RMS backward rules, even for an update produced by the detached arm. Differentiating a detached forward would give a surrogate derivative and would be mathematically incorrect here. All named parameters participate in the update and JVP; this is not a query-weight-only approximation.

Alpha is a parameter interpolation AFTER an actual optimizer step. It does not mean rerunning Adam with a scaled learning rate. Alpha 1 is mandatory and corresponds to the actual new parameters. Endpoints are formed using float64 parameter differences before casting, which recovers the exact updated float32 parameters at alpha 1.

The actual update uses the checkpoint's training precision and scaler. Attention measurements use a common float32 numerical evaluation by default, or float64 when selected. Thus an AMP-trained checkpoint is analyzed through its float32 evaluation map; the diagnostic does not claim to reconstruct every native fp16/bf16 rounding event. Using a different GPU or software version can also prevent bitwise reproduction of the original training process. The within-run replay comparison is the relevant measured numerical floor.

## KL, mask, and row weighting

For a fixed causal row, p = softmax(z) on the allowed keys. For a finite logit change u,

```
KL(p || p_new) = log E_p[exp(u)] - E_p[u].
```

Masked keys never participate. The first query has one allowed key and identically zero entropy/KL, so it is excluded. Each remaining query/head/example/layer row receives equal descriptive weight. Per-layer and pooled values are saved. **Rows, heads, tokens, layers, and checkpoints are not independent statistical replicates.** Statistical inference uses independently initialized paired seeds at one predeclared checkpoint.

KL uses centered increments, an `expm1(x)-x` remainder, and a small-argument polynomial. A log-sum-exp fallback handles large changes. This avoids negative tiny KL values caused by subtracting nearly equal log-normalizers. The direction saved as `endpoint_kl_mean` is always explicitly start-to-finish.

For an individual arm it is KL(common checkpoint || arm endpoint). For a contrast it is **KL(standard endpoint || detached endpoint)** at the same alpha. There is no fictitious “baseline plus contrast” KL in the saved contrast results.

## Fisher errors and temperature/shape

The local Fisher seminorm is Var_p(v). Constants added to a row do not affect attention. A prediction error is measured after subtracting its p-weighted row mean.

```
Fisher MSE = mean_rows E_p[(center(u) - center(v))^2]
relative RMSE = sqrt(Fisher MSE / mean_rows Var_p(u))
```

For an arm contrast, u = z_detached_endpoint - z_standard_endpoint and v = alpha times (v_detached - v_standard), both measured with the **common checkpoint** Fisher weights. The KL itself uses the actual standard endpoint probabilities.

Numerically, the contrast prediction is obtained by applying the JVP directly to the float64-subtracted parameter updates. This is mathematically equivalent to subtracting two JVPs but avoids their float32 cancellation. The discrepancy between the two calculations is recorded as a numerical linearity diagnostic.

If an effect approaches that discrepancy or float32 endpoint resolution, repeat a small measurement batch with `--precision float64` before interpreting it. Float64 may be slow on a Colab T4. A bitwise-zero same-arm replay does not by itself establish that all smaller cross-arm effects are numerically resolved.

The temperature comparator is fitted to the predicted increment, not the observed endpoint:

```
beta = Cov_p(z, v) / Var_p(z)
v_temperature = beta (z - E_p[z])
v_shape = v - E_p[v] - v_temperature
```

Its reported error compares u with v_temperature. Shape and temperature energies sum to predicted Fisher energy up to rounding. Zero-temperature-variance rows are marked unidentifiable; beta is not divided by zero. Near-zero total energy has undefined relative metrics and is recorded as JSON `null`, not invented zero or infinity.

`jvp_vs_temperature_rmse_reduction = 1 - RMSE_full / RMSE_temperature` is descriptive. A richer full first-order predictor often fits better than its temperature projection by construction. Its improvement is **not by itself evidence of a novel mechanism or optimization benefit**. The finite alpha 1 effect size, shape share, replay floor, checkpoint dependence, and controlled analytic experiments must be interpreted together.

## Entropy

The first-order entropy change is `-Cov_p(z, v)`. Under the temperature projection it equals `-beta Var_p(z)` by algebra. Agreement between these two expressions is not independent empirical support. Exact entropy changes are measured between the actual endpoints; for a contrast this is H(detached endpoint) - H(standard endpoint). Attention entropy is a mechanistic observable, not a performance score. Lower entropy alone does not imply harm, and higher entropy alone does not imply improvement; held-out NLL supplies the separate behavioral measure.

## Interpretation limits

- A measured remainder is an a posteriori diagnostic, not a certified safe learning rate.
- Small-alpha Taylor convergence is a numerical check; it is not the primary finding.
- A query-only backward intervention can alter upstream gradients throughout the model.
- Actual AdamW forks include moment, denominator, weight decay, and clipping effects; a fixed-preconditioner theorem has narrower assumptions.
- The package does not use the invalid old random placebo whose residual energy differed from the intervention. No energy-matched placebo is claimed in the GPU study.
- The small MHA language model establishes a controlled setting, not a universal conclusion about all large models, GQA, or MLA.
- Synthetic smoke data validate code only and must never appear as natural-language evidence.

## Output and reproducibility

The JSON includes checkpoint/config/data hashes, actual training-batch hash, measurement-batch hash, seed, phase, source trajectory, software/device information, replay errors, arm results, and paired contrast results. Files are written atomically. The saved config and protocol distinguish pilot from confirmation. The same measurement specification is used across seeds; aggregated summaries must preserve a single source trajectory and checkpoint rather than pool unrelated forks.
