# Data

Raw data is **not** committed to this repository.

## Processed format

Training uses processed tensors per market:

```
$DATASET_ROOT/processed_datasets/
├── csi300_qlib_official/
│   ├── tnf_data.npy     # [T, N, 159]  = 158 Alpha158 factors + LABEL0
│   ├── tnf_close.npy    # [T, N] unnormalized close (backtest)
│   ├── stocks.npy       # ticker list
│   ├── times.npy        # trading-day list
│   └── dataset_config.json
├── csi500_qlib_official/ ...
├── sp500_qlib_official/ ...
└── nasdaq100_qlib_official/ ...
```

`DATASET_ROOT` defaults to `<repo_root>/dataset` and can be overridden with an environment variable (see `dataset_paths.py`).

## Source & acquisition

- **A-shares (CSI300 / CSI500)**: Qlib Alpha158 features from daily bar data (`cn_data`). The utility scripts in `dataset_process/` (`dataset_download.py`, `qlib_bin_to_tnf.py`) document the download → Qlib bin → `tnf_*.npy` conversion.
- **US equities (S&P 500 / NASDAQ100)**: Qlib Alpha158 features from Yahoo Finance daily bars (`us_data_yahoo`), converted the same way.

To regenerate the processed tensors, install Qlib and run the `dataset_process/` scripts in order. They are provided for transparency/reproducibility; the large `.npy` files themselves are not committed.

## Synthetic fallback (no data required)

If `tnf_data.npy` is missing at run time, `main.py` automatically generates a small synthetic `[T, N, 159]` dataset (plus a `stocks.npy` ticker list) under `$DATASET_ROOT/processed_datasets/csi300_qlib_official/`, then proceeds with the full train → test → backtest → plot pipeline. This exists purely as a self-check so the code can be exercised without any proprietary data; metrics obtained this way have **no financial meaning**.


## Shape reference

| Market | Stocks (N) | Days (T) | Features (F) | Period |
| --- | --- | --- | --- | --- |
| CSI300 | 578 | 2384 | 159 (158 + label) | 2010–2020 |
| CSI500 | 1075 | 2384 | 159 | 2010–2020 |
| S&P 500 | 589 | 2516 | 159 | 2010–2020 |
| NASDAQ100 | 162 | 2516 | 159 | 2010–2020 |
