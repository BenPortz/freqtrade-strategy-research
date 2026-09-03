# Freqtrade Strategy Research

Backtests of three crypto strategies on the same data, testing whether a trained model beats plain indicator rules.

It did not. The XGBoost model lost 61.6%. The five-line RSI baseline it was measured against lost 19.4%. A rule-based trend follower written afterward lost 16.2% on the same window.

These are backtests of a negative result. Nothing here should be run against a live account.

## Results

1h candles, five pairs (BTC, ETH, SOL, BNB, ADA against USDT), Binance data, 1000 USDT start, dry run, max 3 open positions, 100 USDT per position.

| Strategy | Window | Trades | Return | Win rate | Max drawdown |
|---|---|---|---|---|---|
| XGBoost (FreqAI) | 2024-06 to 2026-01 | 2,380 | -61.6% | 42.3% | 61.7% |
| XGBoost (FreqAI) | 2024-08 to 2024-10 | 241 | -8.3% | 39.8% | 8.3% |
| RSI baseline | 2024-06 to 2026-01 | 266 | -19.4% | 57.9% | 25.0% |
| Trend follow | 2024-06 to 2026-01 | 304 | -16.2% | 21.1% | 18.4% |
| Trend follow | 2023-10 to 2024-04 | 127 | +7.9% | 33.9% | 3.5% |
| Trend follow | 2024-04 to 2026-01 | 330 | -17.1% | 20.9% | 18.6% |

The only profitable window, 2023-10 to 2024-04, was a sustained uptrend. A trend follower should make money there, so it is weak evidence.

## Why the model lost

XGBoost predicted 24-candle forward returns from about 40 features (RSI, MFI, ADX, moving averages, Bollinger width, rate of change, relative volume, MACD, hour, day), retrained every 7 days on a 30-day window.

**Overtrading.** 2,380 trades in 19 months across five pairs. At roughly 0.1% per side, fees account for a large share of the loss before any directional error. The entry threshold was a predicted move above 2%, which fires often on hourly data.

**No regime filter.** The model predicted returns, not whether to participate. `do_predict` only flags points outside the training distribution, which is a data-similarity check rather than a market-direction one. Most of the window trended down and the model kept opening longs.

**Possible label bug.** `set_freqai_targets` computes the target with `.shift(-24).pct_change(24)`, chaining two forward-looking operations. The intent was the 24-candle forward return. This needs rechecking before the file is reused.

The 2-month run (-8.3%) tested whether the long window hid one bad stretch. It did not. The loss is steady.

## What the rule-based version changed

`TrendFollowStrategy` addresses the first two findings:

- 200 EMA regime gate, so entries only fire above the long-term trend
- Entry requires a fresh EMA20 over EMA50 crossover plus MACD and RSI agreement, cutting trades from 2,380 to a few hundred
- ROI cap at 50% with a trailing stop from +6%, losers cut at -8%

It still lost 16.2%. Win rate fell to 21%, which is normal for trend following, but the wins did not cover the losses in a market that mostly chopped or fell.

## Files

```
strategies/
  RsiStrategy.py           RSI mean reversion, the control
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

The numbers above are backtests over 19 months and five pairs. Small sample, approximate fee model.
