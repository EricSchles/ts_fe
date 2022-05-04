import pandas as pd
import numpy as np

def lag_column(df: pd.DataFrame, num_lags: int, column: str) -> pd.DataFrame:
    assert num_lags > 0, "num_lags must be at least 1"
    assert df.isnull().sum().sum() == 0, "cannot have nan's in df"

    lagged_columns = pd.DataFrame()
    for lag in range(1, num_lags+1):
        lagged_columns[column+f"_lagged_{lag}"] = df[column].shift(lag)

    lagged_columns = lagged_columns.dropna()
    new_df = df.iloc[num_lags:]

    return pd.concat([new_df, lagged_columns], axis=1)

def timeseries_train_test_split(X, y, position=0.5):
    if isinstance(position, int):
        assert position < len(y) and position > 1, "if position is int, must be in [2, len(y) - 1]"
    if isinstance(position, float):
        assert position < 1.0 and position > 0.0, "if position is a float, must be between (0.0, 1.0)"
        
    if isinstance(position, float):
        split_point = int(len(y) * position)
    if isinstance(position, int):
        split_point = position
    if position == "auto":
        position = np.random.uniform(0.45, 0.75)
        split_point = int(len(y) * position)

    X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
    y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]
    return (
        X_train, X_test,
        y_train, y_test
    )
    
