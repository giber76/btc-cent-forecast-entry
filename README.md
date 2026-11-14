# btc-cent-forecast-01 — ridge forecaster

Forecasts Bitcoin's price 240 minutes after each checkpoint for make.fun's
`btc-cent-forecast-01` ($250). Entry point: `main.py` (Python 3.13, stdlib only).

## Method

The 240-minute forward log-return is predicted by a ridge regression
(lambda=100) on 13 features of the 72h of 1-minute closes ending at the
checkpoint: returns over 15m / 1h / 4h / 12h / 24h / 48h / 72h, distance from
the 1h / 4h / 24h moving averages, 1h / 4h realised volatility, and a
1h-vs-4h acceleration term (all in basis points). The regression was fit on
480 walk-forward checkpoints cut from the kit's six practice cases (every
15 minutes, 4h-ahead targets), standardised with the training mean/SD
hard-coded in `main.py`. Forecast = checkpoint close x exp(predicted bp/10000),
clamped to the scorer's price grid. Deterministic, no network, milliseconds
per case.

## Local grades (mean absolute error, cents)

- Practice p00–p04 (answers recoverable via case overlap): **27,783**
  (persistence 47,055; house 6-case baseline 44,522)
- Expanding-window held-out walk-forward: **39,147** vs persistence 44,430

## Run locally

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## License

MIT (see LICENSE).
