import pandas as pd
import numpy as np

class TimeSeriesProcessing:
    def __init__(self):
        pass

    def lag_column(self, df: pd.DataFrame, num_lags: int, column: str) -> pd.DataFrame:
        assert num_lags > 0, "num_lags must be at least 1"
        assert df.isnull().sum().sum() == 0, "cannot have nan's in df"
        
        lagged_columns = pd.DataFrame()
        for lag in range(1, num_lags+1):
            lagged_columns[column+f"_lagged_{lag}"] = df[column].shift(lag)

        lagged_columns = lagged_columns.dropna()
        new_df = df.iloc[num_lags:]
        
        return pd.concat([new_df, lagged_columns], axis=1)
