import pandas as pd
from abc import ABC, abstractmethod


class BaseFeatureEngineer(ABC):
    @abstractmethod
    def fit(self, X: pd.DataFrame) -> "BaseFeatureEngineer":
        """Fit the feature engineer to the data."""
        pass

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the input features."""
        pass

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit the feature engineer to the data and transform the input features."""
        return self.fit(X).transform(X)


class OneHotEncoder(BaseFeatureEngineer):
    def __init__(self, columns: list):
        self.columns = columns
        self.categories_ = {}
        self.encoded_columns_ = []

    def fit(self, X: pd.DataFrame) -> "OneHotEncoder":
        # 記錄每個欄位的類別
        for col in self.columns:
            if col in X.columns:
                self.categories_[col] = X[col].unique().tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.categories_:
            raise RuntimeError(
                "The feature engineer must be fitted before calling transform."
            )
        
        X_encoded = X.copy()
        
        # 為每個類別欄位創建 one-hot 編碼
        for col in self.columns:
            if col in X_encoded.columns:
                for category in self.categories_.get(col, []):
                    X_encoded[f"{col}_{category}"] = (X_encoded[col] == category).astype(int)
                X_encoded = X_encoded.drop(columns=[col])
        
        return X_encoded
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)

class TimeSeriesTransformer(BaseFeatureEngineer):
    """
    Transform tabular data into time series format using sliding windows.
    
    Args:
        window_size: Number of time steps to look back
        forecast_horizon: Number of time steps to predict ahead
        target_column: Name of the target column to predict
        feature_columns: List of feature column names (if None, use all except target)
        stride: Step size for sliding window (default: 1)
    """
    def __init__(
        self, 
        window_size: int,
        forecast_horizon: int = 1,
        target_column: str = "Close",
        feature_columns: list = None,
        stride: int = 1
    ):
        self.window_size = window_size
        self.forecast_horizon = forecast_horizon
        self.target_column = target_column
        self.feature_columns = feature_columns
        self.stride = stride
        self.fitted_feature_columns_ = None
        
    def fit(self, X: pd.DataFrame) -> "TimeSeriesTransformer":
        """Fit the transformer by recording feature columns."""
        if self.feature_columns is None:
            # 使用所有欄位除了目標欄位
            self.fitted_feature_columns_ = [
                col for col in X.columns if col != self.target_column
            ]
        else:
            self.fitted_feature_columns_ = self.feature_columns
            
        # 驗證欄位存在
        if self.target_column not in X.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in data")
            
        for col in self.fitted_feature_columns_:
            if col not in X.columns:
                raise ValueError(f"Feature column '{col}' not found in data")
                
        return self
    
    def transform(self, X: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Transform data into time series format.
        
        Returns:
            X_sequences: DataFrame with shape (n_samples, window_size * n_features)
            y_sequences: DataFrame with shape (n_samples, forecast_horizon)
        """
        if self.fitted_feature_columns_ is None:
            raise RuntimeError(
                "The feature engineer must be fitted before calling transform."
            )
        
        X_sequences = []
        y_sequences = []
        
        # 創建滑動窗口
        for i in range(0, len(X) - self.window_size - self.forecast_horizon + 1, self.stride):
            # 提取特徵序列
            X_window = X.iloc[i:i + self.window_size][self.fitted_feature_columns_].values.flatten()
            
            # 提取目標序列
            y_window = X.iloc[
                i + self.window_size:i + self.window_size + self.forecast_horizon
            ][self.target_column].values
            
            X_sequences.append(X_window)
            y_sequences.append(y_window)
        
        # 轉換為 DataFrame
        feature_names = [
            f"{col}_t-{self.window_size - t}"
            for t in range(self.window_size)
            for col in self.fitted_feature_columns_
        ]
        
        target_names = [
            f"{self.target_column}_t+{t+1}"
            for t in range(self.forecast_horizon)
        ]
        
        X_sequences_df = pd.DataFrame(X_sequences, columns=feature_names)
        y_sequences_df = pd.DataFrame(y_sequences, columns=target_names)
        
        return X_sequences_df, y_sequences_df
    
    def fit_transform(self, X: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Fit the transformer and transform the data."""
        return self.fit(X).transform(X)

class FeatureEngineerPipeline(BaseFeatureEngineer):
    def __init__(self, steps: list):
        self.steps = steps

    def fit(self, X: pd.DataFrame) -> "FeatureEngineerPipeline":
        X_temp = X.copy()
        for step in self.steps:
            step.fit(X_temp)
            X_temp = step.transform(X_temp)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        for step in self.steps:
            X = step.transform(X)
        return X
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        for step in self.steps:
            X = step.fit_transform(X)
        return X
