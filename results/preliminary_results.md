# Preliminary results

> **Preliminary — not fully validated.** Everything below is from **single-seed, single-market (CSI300)** runs with varying loss configs. There is no multi-seed averaging, no ablation significance testing, and no statistical inference. These numbers are **feasibility checks**, not conclusions.

## 1. Predictive signal (CSI300, test set)

| Metric | Logged range |
| --- | --- |
| RankIC (IC Mean) | −0.0213 to +0.0006 |
| ICIR | −0.1607 to +0.0150 |
| Long-only ARR | −19.36% to +0.53% |
| Long-only Sharpe | negative (e.g., −0.42) |
| Long-short Sharpe | negative (e.g., −3.30) |

**Interpretation:** training converges (MSE decreases) but the model does not yet produce a positive, statistically-significant cross-sectional signal on real CSI300 data. RankIC is effectively at zero.

## 2. Mechanism verification (synthetic probes)

Passed via `_validate_v2.py` / `_smoke_train.py`:

| Probe | Result |
| --- | --- |
| Graph-constructor gradients flow | ✅ non-zero |
| Lag identifiability (synthetic lag = 3) | ✅ recovers p = [0, 0, 1.0] |
| Residual asynchronous injection changes predictions | ✅ |
| 14 ablation variants forward pass | ✅ 14/14 |

## 3. Learned lag structure (CSI300, recorded run)

With `max_possible_lag = 3`, the learned per-stock lag distribution is **near-uniform** (weights ≈ 1/3 each, expected lag ≈ 2.0), so heterogeneous per-stock conduction lag is **not yet demonstrated** on real data. See `results/figures/lag_distribution.png`.

## 4. Example outputs

- `figures/result_backtest.png` — long-only / long-short equity curves (currently weak/negative).
- `figures/lag_distribution.png` — learned lag distribution (near-uniform in the recorded run).

## 5. What would make this a real result

- Strict walk-forward / leakage-free temporal splits.
- Multi-seed runs with Newey–West / Diebold–Mariano / bootstrap significance.
- A complete ablation table across the 14 variants.
- Turnover / transaction-cost sensitivity and regime analysis.
