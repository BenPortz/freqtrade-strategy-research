# Freqtrade Strategy Research

A crypto trading bot built in Python with Freqtrade, FreqAI, XGBoost and TA-Lib, to test whether a machine-learning strategy could outperform simpler rule-based approaches while keeping downside risk limited.

I stress-tested the strategies on a heavily negative market period first, then evaluated them in favorable conditions. The ML model used roughly 40 technical and market features. The baseline used RSI, a momentum indicator that flags overbought and oversold conditions. XGBoost underperformed the simpler strategies, which led me to look at overtrading, transaction costs, market-regime filtering, and how the model's target was constructed.

## Experiment design

**Question.** Does a trained model beat plain indicator rules after fees, without taking on more downside?

**Order of evaluation.** The main window, 2024-06 to 2026-01, was a sustained downturn for the five pairs tested. Every strategy ran there first. A strategy that survives a bad market earns a look at a good one, so the trend follower was then run on the 2023-10 to 2024-04 uptrend as well.

**Risk limits, applied to every strategy.** A hard stop on each trade (5% for XGBoost, 8% for the trend follower, 10% for RSI), at most 3 open positions, 100 USDT per position from a 1000 USDT dry-run wallet. No strategy was allowed to average down or run without a stop.

**Setup.** 1h candles, BTC, ETH, SOL, BNB and ADA against USDT, Binance data.

**Strategies.**

- `RsiStrategy`, the baseline. Enter below RSI 30, exit above RSI 70. About five lines of logic.
- `XGBoostStrategy`, the ML candidate. FreqAI trains an XGBoost regressor on about 40 features (RSI, MFI, ADX, moving averages, Bollinger width, rate of change, relative volume, MACD, hour and day of week) to predict the 24-candle forward return. Retrained every 7 days on a 30-day window.
- `TrendFollowStrategy`, written after the XGBoost results. A 200 EMA regime gate, an EMA20 over EMA50 entry trigger, MACD and RSI confirmation, and a trailing stop.

## Results

| Strategy | Window | Market | Trades | Return | Max drawdown |
|---|---|---|---|---|---|
| RSI baseline | 2024-06 to 2026-01 | down | 266 | -19.4% | 25.0% |
| XGBoost | 2024-06 to 2026-01 | down | 2,380 | -61.6% | 61.7% |
| XGBoost | 2024-08 to 2024-10 | down | 241 | -8.3% | 8.3% |
| Trend follow | 2024-06 to 2026-01 | down | 304 | -16.2% | 18.4% |
| Trend follow | 2024-04 to 2026-01 | down | 330 | -17.1% | 18.6% |
| Trend follow | 2023-10 to 2024-04 | up | 127 | +7.9% | 3.5% |

The model lost three times as much as the five-line baseline on the same data. The shorter XGBoost window checks whether one bad stretch explained the long run; it did not, the loss was steady throughout. The trend follower held drawdown under 19% in the downturn and was the only strategy tested on the uptrend, where it returned 7.9% with a 3.5% drawdown.

## What the underperformance pointed to

**Overtrading.** 2,380 trades in 19 months across five pairs. The entry threshold, a predicted move above 2%, fires often on hourly data.

**Transaction costs.** At roughly 0.1% per side, fees on that many trades account for a large share of the loss before any directional error is counted.

**Regime filtering.** The model predicted returns and had no signal for whether to participate at all. FreqAI's `do_predict` flag only checks that a data point resembles the training distribution. It says nothing about market direction, and most of the window trended down.

**Target construction.** `set_freqai_targets` computes the label with `.shift(-24).pct_change(24)`, chaining two forward-looking operations. The intent was the 24-candle forward return. This needs rechecking before the model is trained again, since a mislabeled target would explain a model that fits in training and fails in backtest.

## What changed in the trend follower

The trend follower was built against the first three findings:

- A 200 EMA regime gate, so it only opens longs above the long-term trend
- Entry needs a fresh EMA20 over EMA50 crossover plus MACD and RSI agreement, which cut trades from 2,380 to a few hundred
- ROI cap at 50% with a trailing stop from +6%, and losers cut at 8%

It still lost 16.2% in the downturn. The win rate fell to 21%, which is normal for trend following, but the wins did not cover the losses in a market that mostly chopped or fell. In the uptrend it made money with low drawdown, which is what a trend follower should do and is weak evidence on its own.

## Files

```
strategies/
  RsiStrategy.py           RSI mean reversion, the baseline
  TrendFollowStrategy.py   EMA regime gate plus MACD and RSI filters
  XGBoostStrategy.py       FreqAI feature engineering and targets
config.example.json        Freqtrade config, all credentials blank
```

## Running it

Requires [Freqtrade](https://www.freqtrade.io/) with the FreqAI extra, and TA-Lib.

```bash
pip install "freqtrade[freqai]"
freqtrade create-userdir --userdir user_data
cp config.example.json user_data/config.json
cp strategies/*.py user_data/strategies/
freqtrade download-data --config user_data/config.json --timeframe 1h --days 600
```

Backtest:

```bash
freqtrade backtesting --config user_data/config.json --strategy RsiStrategy --timerange 20240601-20260101
freqtrade backtesting --config user_data/config.json --strategy TrendFollowStrategy --timerange 20240601-20260101
freqtrade backtesting --config user_data/config.json --strategy XGBoostStrategy --freqaimodel XGBoostRegressor --timerange 20240601-20260101
```

The XGBoost run trains a model per pair per retrain window and takes much longer than the other two.

## Notes

`config.example.json` has `dry_run: true` and every credential field blank. `.gitignore` excludes `user_data/` so a filled-in config does not get committed.

These are backtests over 19 months and five pairs, with an approximate fee model. Nothing here has been run against a live account.
