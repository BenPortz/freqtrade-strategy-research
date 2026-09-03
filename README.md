# Freqtrade Strategy Research

Backtests of three crypto trading strategies on the same data, to see whether a
trained model beats plain indicator rules.

Short answer: it did not. The XGBoost model lost 61.6% over the test window.
The five-line RSI baseline it was supposed to beat lost 19.4%. A rule-based
trend follower written afterward lost 16.2% on the same window, and made money
only in the one period that trended up.

This repo is the strategies and the numbers. It is a record of a negative
result, not a bot anyone should run.

## Results

All runs: 1h candles, five pairs (BTC, ETH, SOL, BNB, ADA against USDT),
Binance data, 1000 USDT starting balance, dry run, max 3 open positions,
100 USDT per position.

| Strategy | Window | Trades | Return | Win rate | Max drawdown |
|---|---|---|---|---|---|
| XGBoost (FreqAI) | 2024-06 to 2026-01 | 2,380 | -61.6% | 42.3% | 61.7% |
| XGBoost (FreqAI) | 2024-08 to 2024-10 | 241 | -8.3% | 39.8% | 8.3% |
| RSI baseline | 2024-06 to 2026-01 | 266 | -19.4% | 57.9% | 25.0% |
| Trend follow | 2024-06 to 2026-01 | 304 | -16.2% | 21.1% | 18.4% |
| Trend follow | 2023-10 to 2024-04 | 127 | +7.9% | 33.9% | 3.5% |
| Trend follow | 2024-04 to 2026-01 | 330 | -17.1% | 20.9% | 18.6% |

The one profitable row is the 2023-10 to 2024-04 window, which was a sustained
uptrend. A trend follower making money in a trending market is close to
tautological, so that result says little about the strategy.

## What went wrong with the model

The XGBoost run is the interesting failure. It predicted 24-candle forward
returns from about 40 engineered features (RSI, MFI, ADX, moving averages,
Bollinger width, rate of change, relative volume, MACD, hour and day), retrained
every 7 days on a 30-day window.

Three problems, in rough order of how much damage they did:

**It traded constantly.** 2,380 trades in 19 months across five pairs. At
roughly 0.1% per side, fees alone account for a large share of the loss before
any directional error. The entry condition was a predicted move above 2%, which
on hourly crypto data fires often.

**It had no regime awareness.** The model was trained to predict returns, not
to know when it should stay out. `do_predict` filters points outside the
training distribution, but that is a data-similarity check, not a market-
direction check. Most of 2024 to 2025 in this pair set trended down, and the
model kept opening longs into it.

**The label leaked in a subtle way.** `set_freqai_targets` computes the target
with `.shift(-24).pct_change(24)`, which chains two operations that both look
forward. The intent was the return over the next 24 candles. Worth rechecking
before anyone reuses this file, since a mislabeled target would explain a model
that looks fine in training and fails in backtest.

The 2-month window (-8.3%) was run to check whether the long window was hiding
one catastrophic stretch. It was not. The loss is steady.

## What the rule-based follow-up changed

`TrendFollowStrategy` was written against those three findings:

- A 200 EMA regime gate, so entries only happen above the long-term trend
- Entry needs a fresh EMA20 over EMA50 crossover plus MACD and RSI agreement,
  which cut trade count from 2,380 to a few hundred
- ROI cap raised to 50% and a trailing stop from +6%, so winners are not closed
  early, with losers cut at -8%

It still lost money on the full window. The win rate dropped to 21%, which is
expected for a trend follower (many small losses, few large wins), but the wins
were not large enough to cover the losses in a market that mostly chopped
sideways or fell.

## Files

```
strategies/
  RsiStrategy.py           RSI mean reversion, the control
  TrendFollowStrategy.py   EMA regime gate plus MACD and RSI filters
  XGBoostStrategy.py       FreqAI feature engineering and targets
config.example.json        Freqtrade config with all credentials blank
```

## Running it

Requires [Freqtrade](https://www.freqtrade.io/) with the FreqAI extra, and
TA-Lib.

```bash
pip install "freqtrade[freqai]"
freqtrade create-userdir --userdir user_data
cp config.example.json user_data/config.json
cp strategies/*.py user_data/strategies/
```

Download data for the pairs you want:

```bash
freqtrade download-data --config user_data/config.json --timeframe 1h --days 600
```

Then backtest:

```bash
freqtrade backtesting --config user_data/config.json --strategy RsiStrategy --timerange 20240601-20260101
freqtrade backtesting --config user_data/config.json --strategy TrendFollowStrategy --timerange 20240601-20260101
freqtrade backtesting --config user_data/config.json --strategy XGBoostStrategy --freqaimodel XGBoostRegressor --timerange 20240601-20260101
```

The XGBoost run trains a model per pair per retrain window, so it takes
considerably longer than the other two.

## Notes

`config.example.json` has `dry_run` set to true and every credential field
blank. Fill in `exchange.key` and `exchange.secret` only if you intend to
connect to a real account, and do not commit that file afterward. The
`.gitignore` excludes `user_data/` for this reason.

Nothing here has been run against a live account, and the results above are
backtests. Backtested returns on 19 months of five pairs are a small sample,
and the fee model is an approximation. Treat the numbers as a record of what
these specific rules did on this specific data.
