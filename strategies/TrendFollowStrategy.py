import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from pandas import DataFrame

from freqtrade.strategy import IStrategy


class TrendFollowStrategy(IStrategy):
    """
    Rule-based multi-filter trend following.

    Written after the XGBoost run lost 61% over the same period. It addresses
    three problems from that run:

    1. It bought into downtrends. A 200 EMA regime gate now blocks entries
       below the long-term trend.

    2. It took 2,380 trades in 19 months and paid fees on all of them. Entries
       now need a fresh EMA crossover plus three confirming filters, which
       brings the count to a few hundred.

    3. It capped winners and held losers. The ROI cap is high enough to rarely
       close early, a trailing stop handles the exit from +6%, and losers are
       cut at -8%.

    Filters:
      EMA200            regime gate, only long above it
      EMA20 over EMA50  entry trigger
      MACD histogram    momentum confirmation, must be positive
      RSI below 70      avoids entering an overbought spike

    Backtest results are in the README. Two of three windows lost money.
    """

    timeframe = "1h"
    can_short = False

    # Set high so it rarely closes a winner early. The trailing stop and the
    # exit signal do most of the work.
    minimal_roi = {"0": 0.50}

    stoploss = -0.08

    # Start trailing once a trade is up 6%, then give back at most 3%.
    trailing_stop = True
    trailing_stop_positive = 0.03
    trailing_stop_positive_offset = 0.06
    trailing_only_offset_is_reached = True

    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    # The EMA200 regime filter needs 200 candles of history.
    startup_candle_count = 200

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Trend and regime
        dataframe["ema_fast"] = ta.EMA(dataframe, timeperiod=20)
        dataframe["ema_slow"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["ema_trend"] = ta.EMA(dataframe, timeperiod=200)

        # Momentum confirmation
        macd = ta.MACD(dataframe)
        dataframe["macd"] = macd["macd"]
        dataframe["macdsignal"] = macd["macdsignal"]
        dataframe["macdhist"] = macd["macdhist"]

        # Overbought guard
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                # Regime: broader trend is up
                (dataframe["close"] > dataframe["ema_trend"])
                # Trigger: fast EMA just crossed above slow EMA
                & (qtpylib.crossed_above(dataframe["ema_fast"], dataframe["ema_slow"]))
                # Momentum agrees
                & (dataframe["macdhist"] > 0)
                # Not already overbought
                & (dataframe["rsi"] < 70)
                & (dataframe["volume"] > 0)
            ),
            "enter_long",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                # Trend turned: fast EMA crossed back below slow EMA
                (qtpylib.crossed_below(dataframe["ema_fast"], dataframe["ema_slow"]))
                # Or price lost the long-term trend entirely
                | (dataframe["close"] < dataframe["ema_trend"])
            ),
            "exit_long",
        ] = 1
        return dataframe
