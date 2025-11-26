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
