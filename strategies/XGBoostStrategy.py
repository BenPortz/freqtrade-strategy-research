import logging
from functools import reduce
from typing import Optional

import pandas as pd
import talib.abstract as ta
from pandas import DataFrame

from freqtrade.strategy import IStrategy, merge_informative_pair


logger = logging.getLogger(__name__)


class XGBoostStrategy(IStrategy):
    """
    FreqAI strategy using XGBoost to predict 24-candle forward returns.

    The model trains on technical indicators and predicts two things:
    do_predict, which is FreqAI's own flag for whether the current point is
    inside the training distribution, and &-price_change, the expected percent
    move over the next 24 candles. A trade opens when the model predicts a rise
    above 2% and the dissimilarity filter passes.

    On the 2024-06 to 2026-01 downturn it lost 61.6% across 2,380 trades,
    against 19.4% for the RSI baseline. The README covers what that pointed
    to: overtrading, transaction costs, regime filtering and target
    construction.
    """

    # Strategy settings
    minimal_roi = {"0": 0.1, "60": 0.05, "120": 0.02, "240": -1}
    stoploss = -0.05
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.03
    trailing_only_offset_is_reached = True

    timeframe = "1h"
    can_short = False
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    process_only_new_candles = True
    startup_candle_count = 50

    def feature_engineering_expand_all(
        self, dataframe: DataFrame, period: int, metadata: dict, **kwargs
    ) -> DataFrame:
        """
        Features computed at every timeframe/period combination defined in config.
        These are automatically shifted and expanded by FreqAI.
        """
        dataframe["%-rsi"] = ta.RSI(dataframe, timeperiod=period)
        dataframe["%-mfi"] = ta.MFI(dataframe, timeperiod=period)
        dataframe["%-adx"] = ta.ADX(dataframe, timeperiod=period)
        dataframe["%-sma"] = ta.SMA(dataframe, timeperiod=period)
        dataframe["%-ema"] = ta.EMA(dataframe, timeperiod=period)

        bollinger = ta.BBANDS(dataframe, timeperiod=period)
        dataframe["%-bb_upperband"] = bollinger["upperband"]
        dataframe["%-bb_middleband"] = bollinger["middleband"]
        dataframe["%-bb_lowerband"] = bollinger["lowerband"]
        dataframe["%-bb_width"] = (
            dataframe["%-bb_upperband"] - dataframe["%-bb_lowerband"]
        ) / dataframe["%-bb_middleband"]
        dataframe["%-close-bb_lower"] = (
            dataframe["close"] / dataframe["%-bb_lowerband"]
        )

        dataframe["%-roc"] = ta.ROC(dataframe, timeperiod=period)
        dataframe["%-relative_volume"] = (
            dataframe["volume"] / dataframe["volume"].rolling(period).mean()
        )

        return dataframe

    def feature_engineering_expand_basic(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ) -> DataFrame:
        """
        Features computed once per timeframe (no period expansion).
        """
        dataframe["%-pct-change"] = dataframe["close"].pct_change()
        dataframe["%-raw_volume"] = dataframe["volume"]
        dataframe["%-raw_price"] = dataframe["close"]

        macd = ta.MACD(dataframe)
        dataframe["%-macd"] = macd["macd"]
        dataframe["%-macdsignal"] = macd["macdsignal"]
        dataframe["%-macdhist"] = macd["macdhist"]

        dataframe["%-day_of_week"] = pd.to_datetime(
            dataframe["date"]
        ).dt.dayofweek
        dataframe["%-hour_of_day"] = pd.to_datetime(
            dataframe["date"]
        ).dt.hour

        return dataframe

    def feature_engineering_standard(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ) -> DataFrame:
        """
        Target label: the % price change over the next 24 candles.
        FreqAI will train the model to predict this value.
        """
        dataframe["&-price_change"] = (
            dataframe["close"]
            .shift(-self.freqai_info["feature_parameters"]["label_period_candles"])
            .pct_change(self.freqai_info["feature_parameters"]["label_period_candles"])
        )
        return dataframe

    def set_freqai_targets(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ) -> DataFrame:
        """
        Define what the model should predict.
        &-price_change: the % return over the next 24 candles.
        """
        dataframe["&-price_change"] = (
            dataframe["close"]
            .shift(-self.freqai_info["feature_parameters"]["label_period_candles"])
            .pct_change(self.freqai_info["feature_parameters"]["label_period_candles"])
        )
        return dataframe

    def populate_indicators(
        self, dataframe: DataFrame, metadata: dict
    ) -> DataFrame:
        # Regular indicators available in entry/exit signals (not FreqAI features)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe = self.freqai.start(dataframe, metadata, self)
        return dataframe

    def populate_entry_trend(
        self, dataframe: DataFrame, metadata: dict
    ) -> DataFrame:
        dataframe.loc[
            (
                # Model is confident (DI filter passed)
                (dataframe["do_predict"] == 1)
                # Model predicts price will rise more than 2%
                & (dataframe["&-price_change"] > 0.02)
                # RSI not overbought
                & (dataframe["rsi"] < 70)
                & (dataframe["volume"] > 0)
            ),
            "enter_long",
        ] = 1

        return dataframe

    def populate_exit_trend(
        self, dataframe: DataFrame, metadata: dict
    ) -> DataFrame:
        dataframe.loc[
            (
                (dataframe["do_predict"] == 1)
                # Model predicts price will fall
                & (dataframe["&-price_change"] < -0.01)
            ),
            "exit_long",
        ] = 1

        return dataframe
