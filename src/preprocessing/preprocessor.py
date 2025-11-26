import pandas as pd
from abc import ABC, abstractmethod
from sklearn.preprocessing import MinMaxScaler as SklearnMinMaxScaler


class BasePreprocessor(ABC):
    @abstractmethod
    def fit(self, X: pd.DataFrame) -> "BasePreprocessor":
        """Fit the preprocessor to the data."""
        pass

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the data using the fitted preprocessor."""
        pass

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit the preprocessor and transform the data."""
        return self.fit(X).transform(X)


class MissingValueHandler(BasePreprocessor):
    def __init__(self, strategy: str = "mean", fill_value=None):
        self.strategy = strategy
        self.fill_values = {}
        self.fill_value = fill_value  # 新增屬性

    def fit(self, X: pd.DataFrame) -> "MissingValueHandler":
        # 只處理數值型欄位
        numeric_columns = X.select_dtypes(include=["number"]).columns

        for column in numeric_columns:
            if X[column].isnull().any():
                if self.strategy == "mean":
                    self.fill_values[column] = X[column].mean()
                elif self.strategy == "median":
                    self.fill_values[column] = X[column].median()
                elif self.strategy == "mode":
                    self.fill_values[column] = X[column].mode()[0]
                elif self.strategy == "constant":
                    self.fill_values[column] = self.fill_value

        # 處理類別型欄位（用 mode 或 'Unknown'）
        categorical_columns = X.select_dtypes(include=["object", "category"]).columns
        for column in categorical_columns:
            if X[column].isnull().any():
                if X[column].mode().empty:
                    self.fill_values[column] = "Unknown"
                else:
                    self.fill_values[column] = X[column].mode()[0]

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for column, fill_value in self.fill_values.items():
            if column in X.columns:
                X[column] = X[column].fillna(fill_value)
        return X

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)


class OutlierHandler(BasePreprocessor):
    def __init__(self, method: str = "zscore", threshold: float = 3.0):
        self.method = method
        self.threshold = threshold

    def fit(self, X: pd.DataFrame) -> "OutlierHandler":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_cleaned = X.copy()
        numeric_columns = X.select_dtypes(include=["number"]).columns
        
        for column in numeric_columns:  # 只處理數值型欄位
            if self.method == "zscore":
                mean = X[column].mean()
                std = X[column].std()
                z_scores = (X[column] - mean) / std
                mask = z_scores.abs() <= self.threshold
                X_cleaned = X_cleaned[mask]
            elif self.method == "iqr":
                Q1 = X[column].quantile(0.25)
                Q3 = X[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - self.threshold * IQR
                upper_bound = Q3 + self.threshold * IQR
                mask = (X[column] >= lower_bound) & (X[column] <= upper_bound)
                X_cleaned = X_cleaned[mask]
            else:
                raise ValueError(f"Unknown method: {self.method}")
        return X_cleaned

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)


class DropColumnsPreprocessor(BasePreprocessor):
    def __init__(self, columns_to_drop: list[str]):
        self.columns_to_drop = columns_to_drop

    def fit(self, X: pd.DataFrame) -> "DropColumnsPreprocessor":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.drop(columns=self.columns_to_drop, errors="ignore")

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)

class MinMaxScaler(BasePreprocessor):
    def __init__(self, feature_range: tuple = (0, 1)):
        """
        Initialize MinMaxScaler using sklearn's MinMaxScaler.
        
        Args:
            feature_range: Desired range of transformed data (min, max)
        """
        self.feature_range = feature_range
        self.scaler = SklearnMinMaxScaler(feature_range=feature_range)
        self.numeric_columns = None

    def fit(self, X: pd.DataFrame) -> "MinMaxScaler":
        # 只處理數值型欄位
        self.numeric_columns = X.select_dtypes(include=["number"]).columns.tolist()
        
        if self.numeric_columns:
            self.scaler.fit(X[self.numeric_columns])
        
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.numeric_columns is None:
            raise RuntimeError(
                "The scaler must be fitted before calling transform."
            )
        
        X_scaled = X.copy()
        
        if self.numeric_columns:
            X_scaled[self.numeric_columns] = self.scaler.transform(
                X[self.numeric_columns]
            )
        
        return X_scaled

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)

class PreprocessingPipeline(BasePreprocessor):
    def __init__(self, steps: list[BasePreprocessor]):
        self.steps = steps

    def fit(self, X: pd.DataFrame) -> "PreprocessingPipeline":
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
