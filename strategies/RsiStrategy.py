import talib.abstract as ta
from pandas import DataFrame

from freqtrade.strategy import IStrategy


class RsiStrategy(IStrategy):
    """
    Rule-based baseline: RSI mean reversion.

    Enter when RSI drops below 30, exit when it rises above 70. No machine
    learning, about five lines of actual logic.

    This is the control for the experiment. The XGBoost strategy has to beat
    this after fees to justify the extra complexity of a trained model. Over
    the main test window it did not: this lost 19.4% and the model lost 61.6%.
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
