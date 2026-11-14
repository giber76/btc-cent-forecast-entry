#!/usr/bin/env python3
"""btc-cent-forecast-01 entry: ridge-regression forecaster for BTC's price 240 minutes ahead.

Method (2-minute version): the 240-minute forward log-return is predicted by a
ridge regression (lambda=100) on 13 features of the 72 hours of 1-minute closes
ending at the checkpoint: short/long returns (15m, 1h, 4h, 12h, 24h, 48h, 72h),
distance from 1h/4h/24h moving averages, 1h/4h realised volatility, and a
1h-vs-4h acceleration term. All features are in basis points. The regression was
fit on 480 walk-forward checkpoints cut from the kit's six practice cases
(every 15 minutes, 48h+ history, 4h-ahead targets), standardised with the
training mean/SD below. Forecast = checkpoint close * exp(predicted bp / 10000),
clamped to the scorer's price grid. Deterministic, stdlib only, ~milliseconds.

Local grades (5 practice cases with recoverable answers, cents MAE):
  practice p00-p04: 27,783 (persistence 47,055; house 6-case baseline 44,522)
  expanding-window held-out walk-forward: 39,147 vs persistence 44,430.
"""

import json
import math
import os
from pathlib import Path

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

MIN_PRICE = 0.00001
MAX_PRICE = 9_999_999.99999

# Ridge fit on the kit practice walk-forward set (480 checkpoints, lambda=100).
# Features, in order: r15m, r1h, r4h, r12h, r24h, r48h, r72h, d_ma1h, d_ma4h,
# d_ma24h, vol1h, vol4h, accel -- all in basis points of log price.
MU = [-0.7702307697406607, -3.5807833077906626, -11.612422086389342, -20.060640895772405, -64.12114217854139, -162.4022010327854, -167.06068877500172, -1.9488768232240126, -6.8350951759292435, -25.427528927023733, 4.646885352197704, 4.860145611262627, -0.6776777861933195]
SD = [20.613799515759577, 38.44810108001834, 63.81655740080403, 98.63411531101923, 102.31637391285523, 74.40031388081941, 61.96336966749504, 22.269113565191095, 37.96535305843996, 68.59916841361839, 2.3989182864203613, 1.9137578016805643, 32.915940212393394]
W = [1.3120451060219203, 0.8233202552479828, -1.604329564077701, 13.518501084503397, -28.587280841388505, 9.627810578631724, -11.965480350526132, 1.728571542488634, -12.648696982361635, -7.440832176934973, -11.153468396955688, 8.38572358798136, 1.7393031295636743]
INTERCEPT_BP = -17.199198404060073


def _mean(xs):
    return sum(xs) / len(xs)


def _std(xs):
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def features(lp, t):
    """13 features (basis points) from log prices lp[0..t], checkpoint at t."""

    def ret(w):
        return (lp[t] - lp[t - w]) * 1e4

    def ma(w):
        return _mean(lp[t - w + 1 : t + 1])

    d1 = [lp[i + 1] - lp[i] for i in range(t - 60, t)]
    d4 = [lp[i + 1] - lp[i] for i in range(t - 240, t)]
    return [
        ret(15),
        ret(60),
        ret(240),
        ret(720),
        ret(1440),
        ret(2880),
        (lp[t] - lp[0]) * 1e4,
        (lp[t] - ma(60)) * 1e4,
        (lp[t] - ma(240)) * 1e4,
        (lp[t] - ma(1440)) * 1e4,
        _std(d1) * 1e4,
        _std(d4) * 1e4,
        ret(60) - ret(240) / 4.0,
    ]


def forecast_price(prices):
    lp = [math.log(p) for p in prices]
    t = len(lp) - 1
    feats = features(lp, t)
    pred_bp = INTERCEPT_BP + sum(
        wi * (fi - mu) / sd for wi, fi, mu, sd in zip(W, feats, MU, SD)
    )
    price = math.exp(lp[t] + pred_bp / 1e4)
    price = min(MAX_PRICE, max(MIN_PRICE, price))
    return round(price, 5)


def main():
    in_dir = Path(INPUT_DIR)
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(in_dir.glob("*.in")):
        case = json.loads(path.read_text())
        prices = case["markets"][0]["prices"]
        price = forecast_price(prices)
        out = {"forecasts": [{"case": case["case"], "price": price}]}
        (out_dir / (path.stem + ".out")).write_text(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
