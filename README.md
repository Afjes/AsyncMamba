# AsyncMamba

A deep-learning pipeline for cross-sectional stock return prediction on CSI300, CSI500, S&P 500, and NASDAQ100, built around an **asynchronous spatio-temporal Mamba** architecture.

> ⚠️ **Status: prototype / preliminary.** The pipeline runs end-to-end and the core mechanisms pass synthetic validation, but predictive performance on real data is **not yet demonstrated** (see [Known limitations](#5-known-limitations) and [results/preliminary_results.md](results/preliminary_results.md)). Treat the current numbers as feasibility checks, not validated results.

## 1. What this repo does

Most "temporal-then-spatial" models first encode each stock's own history, then aggregate cross-stock relations at a single snapshot. This assumes information propagates synchronously and with a single uniform delay. AsyncMamba instead couples a **Mamba state-space sequence model** with **graph attention inside the temporal recurrence loop**, so that past cross-stock structure directly shapes the next time step.

Three core components:

1. **Asynchronous conduction** — historical spatial features are injected into the Mamba state update as a residual (`x_t + λ·gate·s_lagged`), so the current day's factors are never overwritten.
2. **Adaptive lag pooling** — a per-stock lag distribution is learned from (a) a static prior, (b) a state-conditioned modulation head, and (c) a monotone survival gate that truncates long lags.
3. **Dual-path spatial aggregation** — a sparse dynamic graph (micro / peer effects) fused with multi-head market pooling (macro / common factors) via a learnable coefficient.

## 2. Data

Raw data is **not** committed to this repository. See [data/README.md](data/README.md) for sources and how to obtain it.

- Features: Qlib **Alpha158** (158 factors) + Qlib `LABEL0` (next-day return), per market.
- CSI300: 578 stocks × 2384 days; CSI500: 1075 × 2384; S&P 500: 589 × 2516; NASDAQ100: 162 × 2516.

Processed files are expected under `$DATASET_ROOT/processed_datasets/{csi300,csi500,sp500,nasdaq100}_qlib_official/`.

## 3. Quick start

The pipeline runs end-to-end **with or without a dataset**. When the expected data file is missing, it automatically generates a small synthetic stand-in so the full flow can still be verified.

### 3.1 Install

```bash
cd AsyncMamba
python -m venv .venv && source .venv/bin/activate   # optional, conda
pip install -r requirements.txt
```

### 3.2 Run without any dataset (synthetic fallback)

```bash
# No dataset required. main.py prints a warning, synthesizes a [T,N,F] dataset +
# stocks.npy under $DATASET_ROOT/processed_datasets/csi300_qlib_official/,
# then trains / tests / backtests / plots on it.
python main.py --mode train --device cpu
```

Metrics from the synthetic fallback are **meaningless** — they only prove the code runs. Use `--synthetic` to force this path even when real data exists.

### 3.3 Run with real data

```bash
# Point to your local data root (or place the data under ./dataset)
export DATASET_ROOT=/path/to/dataset

# Train + test + backtest + plots on CSI300 (CPU-safe)
python main.py --mode train --device cpu
```

- `--mode train` runs the full pipeline; `--mode ablation` runs the 14 ablation variants.
- `--device` accepts `auto / cpu / cuda / mps` (Apple Silicon).
- ⚠️ `main.py` defaults to `--mode ablation`; always pass `--mode train` for a single run.

### 3.4 Smoke / mechanism checks (no dataset required)

```bash
python validate_mechanisms.py     # graph gradients, lag identifiability, residual injection, 14 ablation forward passes
python _smoke_train.py     # two synthetic training steps
```

These only need `torch`; they no longer require the dataset.


## 4. Preliminary results

A single-seed CSI300 run converges (loss decreases) but the predictive signal is at/around zero. See [results/preliminary_results.md](results/preliminary_results.md) for the numbers and the honest interpretation.

| Metric (CSI300, single seed) | Preliminary value |
| --- | --- |
| Test RankIC (IC Mean) | ≈ −0.02 to +0.001 |
| Test ICIR | mostly negative |
| Long-only / long-short Sharpe | negative |

These are **feasibility checks, not validated results**: no multi-seed averaging, no ablation significance, no statistical inference.

## 5. Known limitations

- **No positive, statistically-significant alpha yet** on real CSI300 data (test RankIC ≈ 0).
- **Single seed**; no multi-seed / bootstrap / Newey–West / Diebold–Mariano inference.
- **No full ablation report**; the 14 variants only pass forward / identifiability checks.
- **Learned lag structure is near-uniform** in the recorded run (expected lag ≈ 2.0 with `L=3`), so per-stock lag heterogeneity is not yet demonstrated on real data.
- **Backtest is simplified**: transaction costs are modeled, but there is no walk-forward re-training, regime analysis, or turnover-sensitivity study.
- **Potential leakage / temporal-split issues** still need a strict audit before any conclusion.

## 6. Repository layout

```
AsyncMamba/
├── README.md
├── requirements.txt
├── main.py                 # entry point: Config + model + train/test loop
├── data_self.py            # data loading & normalization (strict 7:1:2 split)
├── simple_mamba.py         # single-step Mamba cell
├── metric.py               # MSE/ListNet losses, RankIC/ICIR/ARR/Sharpe metrics
├── plot.py                 # training curves, backtest, lag/network plots
├── ablation.py             # 14 ablation variants
├── experiment_timer.py     # timing
├── dataset_paths.py        # dataset path resolution (DATASET_ROOT override)
├── _smoke_train.py         # smoke test
├── _validate_v2.py         # mechanism validation
├── dataset_process/        # data acquisition / conversion scripts
├── data/README.md          # data documentation
└── results/
    ├── preliminary_results.md
    └── figures/
```

## 7. Contact

Wang Zhe — 599641678@qq.com | GitHub: https://github.com/Afjes/AsyncMamba
