import talib.abstract as ta
from pandas import DataFrame

from freqtrade.strategy import IStrategy


class RsiStrategy(IStrategy):
    """
    Rule-based baseline: RSI mean reversion.

    Enter when RSI drops below 30, exit when it rises above 70. About five
    lines of logic, with a 10% stop on every trade.

    This is the baseline the XGBoost model had to beat after fees. On the
    2024-06 to 2026-01 downturn it lost 19.4% against 61.6% for the model.
    """

    timeframe = "1h"
    can_short = False

    # Give trades room and let the RSI exit do the work, but cap the downside.
    minimal_roi = {"0": 0.10}
    stoploss = -0.10
    trailing_stop = False

    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    startup_candle_count = 20

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe["rsi"] < 30)
                & (dataframe["volume"] > 0)
            ),
            "enter_long",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["rsi"] > 70),
            "exit_long",
        ] = 1
        return dataframe
