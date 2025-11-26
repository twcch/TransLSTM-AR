import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional


class BaseModel(ABC):
    """Base class for all models."""
    
    @abstractmethod
    def fit(self, X_train: pd.DataFrame, y_train: pd.DataFrame, 
            X_val: Optional[pd.DataFrame] = None, 
            y_val: Optional[pd.DataFrame] = None) -> "BaseModel":
        """Train the model."""
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Save the model."""
        pass
    
    @abstractmethod
    def load(self, path: str) -> "BaseModel":
        """Load the model."""
        pass