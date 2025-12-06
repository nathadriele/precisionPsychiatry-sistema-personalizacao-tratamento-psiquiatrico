"""
Pré-processadores de dados avançados para Medicina de Precisão em Psiquiatria.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BasePreprocessor(ABC):    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.fitted = False
    
    @abstractmethod
    def fit(self, X: np.ndarray) -> 'BasePreprocessor':
        pass
    
    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        pass
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
    
    def log(self, message: str):
        if self.verbose:
            logger.info(message)


class MissingValueHandler(BasePreprocessor):
    """Tratamento de valores faltantes."""
    
    def __init__(
        self,
        strategy: str = "mean",
        threshold: float = 0.5,
        verbose: bool = True
    ):
        super().__init__(verbose)
        self.strategy = strategy
        self.threshold = threshold
        self.fill_values = {}
    
    def fit(self, X: pd.DataFrame) -> 'MissingValueHandler':
        self.log("Analisando valores faltantes...")
        
        # Remover colunas com muitos missing values
        missing_pct = X.isnull().sum() / len(X)
        cols_to_drop = missing_pct[missing_pct > self.threshold].index.tolist()
        
        if cols_to_drop:
            self.log(f"Colunas a dropar (>{self.threshold*100:.0f}% missing): {cols_to_drop}")
            X = X.drop(columns=cols_to_drop)
        
        # Calcular valores de preenchimento
        if self.strategy == "mean":
            self.fill_values = X.mean()
        elif self.strategy == "median":
            self.fill_values = X.median()
        elif self.strategy == "mode":
            self.fill_values = X.mode().iloc[0]
        
        self.fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise ValueError("Preprocessor não foi ajustado. Chame fit() primeiro.")
        
        X = X.copy()
        
        if self.strategy == "drop":
            X = X.dropna()
        elif self.strategy == "forward_fill":
            X = X.fillna(method='ffill')
        else:
            X = X.fillna(self.fill_values)
        
        return X


class OutlierRemover(BasePreprocessor):
    """Remoção de outliers."""
    
    def __init__(
        self,
        method: str = "iqr",
        threshold: float = 3.0,
        verbose: bool = True
    ):
        super().__init__(verbose)
        self.method = method
        self.threshold = threshold
        self.bounds = {}
    
    def fit(self, X: pd.DataFrame) -> 'OutlierRemover':
        self.log(f"Calculando bounds de outliers usando {self.method}...")
        
        if self.method == "iqr":
            Q1 = X.quantile(0.25)
            Q3 = X.quantile(0.75)
            IQR = Q3 - Q1
            self.bounds = {
                'lower': Q1 - self.threshold * IQR,
                'upper': Q3 + self.threshold * IQR
            }
        elif self.method == "zscore":
            mean = X.mean()
            std = X.std()
            self.bounds = {
                'mean': mean,
                'std': std,
                'threshold': self.threshold
            }
        
        self.fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise ValueError("Preprocessor não foi ajustado.")
        
        X = X.copy()
        n_before = len(X)
        
        if self.method == "iqr":
            mask = ((X >= self.bounds['lower']) & (X <= self.bounds['upper'])).all(axis=1)
            X = X[mask]
        elif self.method == "zscore":
            mask = (np.abs((X - self.bounds['mean']) / self.bounds['std']) < self.threshold).all(axis=1)
            X = X[mask]
        
        n_removed = n_before - len(X)
        if n_removed > 0:
            self.log(f"Removidos {n_removed} outliers ({n_removed/n_before*100:.1f}%)")
        
        return X


class NormalizeScaler(BasePreprocessor):
    """Normalização de features."""
    
    def __init__(
        self,
        method: str = "standard",
        verbose: bool = True
    ):
        super().__init__(verbose)
        self.method = method
        self.scaler = None
    
    def fit(self, X: pd.DataFrame) -> 'NormalizeScaler':
        self.log(f"Ajustando scaler {self.method}...")
        
        if self.method == "standard":
            self.scaler = StandardScaler()
        elif self.method == "robust":
            self.scaler = RobustScaler()
        elif self.method == "minmax":
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Método desconhecido: {self.method}")
        
        self.scaler.fit(X)
        self.fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Scaler não foi ajustado.")
        
        return self.scaler.transform(X)


class FeatureSelector(BasePreprocessor):
    
    def __init__(
        self,
        method: str = "variance",
        threshold: float = 0.01,
        verbose: bool = True
    ):
        super().__init__(verbose)
        self.method = method
        self.threshold = threshold
        self.selected_features = None
    
    def fit(self, X: pd.DataFrame) -> 'FeatureSelector':
        """Ajustar seletor."""
        self.log(f"Selecionando features usando {self.method}...")
        
        if self.method == "variance":
            variances = X.var()
            self.selected_features = variances[variances > self.threshold].index.tolist()
        elif self.method == "correlation":
            corr_matrix = X.corr().abs()
            upper = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            to_drop = [col for col in upper.columns if any(upper[col] > self.threshold)]
            self.selected_features = [col for col in X.columns if col not in to_drop]
        
        self.log(f"Features selecionadas: {len(self.selected_features)}/{len(X.columns)}")
        self.fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise ValueError("Seletor não foi ajustado.")
        
        return X[self.selected_features]


class DimensionalityReducer(BasePreprocessor):
    """Redução de dimensionalidade."""
    
    def __init__(
        self,
        method: str = "pca",
        n_components: int = None,
        variance_ratio: float = 0.95,
        verbose: bool = True
    ):
        super().__init__(verbose)
        self.method = method
        self.n_components = n_components
        self.variance_ratio = variance_ratio
        self.reducer = None
    
    def fit(self, X: np.ndarray) -> 'DimensionalityReducer':
        self.log(f"Ajustando {self.method}...")
        
        if self.method == "pca":
            if self.n_components is None:
                self.reducer = PCA()
                self.reducer.fit(X)
                cumsum = np.cumsum(self.reducer.explained_variance_ratio_)
                self.n_components = np.argmax(cumsum >= self.variance_ratio) + 1
                self.log(f"n_components calculado: {self.n_components}")
            
            self.reducer = PCA(n_components=self.n_components)
            self.reducer.fit(X)
        
        self.fitted = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Redutor não foi ajustado.")
        
        return self.reducer.transform(X)


class PreprocessingPipeline:
    """Pipeline de pré-processamento."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.steps = []
        self.fitted = False
    
    def add_step(self, name: str, preprocessor: BasePreprocessor) -> 'PreprocessingPipeline':
        self.steps.append((name, preprocessor))
        return self
    
    def fit(self, X: pd.DataFrame) -> 'PreprocessingPipeline':
        self.log("Ajustando pipeline...")
        
        for name, preprocessor in self.steps:
            self.log(f"  - Ajustando {name}...")
            if isinstance(X, pd.DataFrame):
                preprocessor.fit(X)
            else:
                preprocessor.fit(X)
        
        self.fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> Union[pd.DataFrame, np.ndarray]:
        if not self.fitted:
            raise ValueError("Pipeline não foi ajustado.")
        
        self.log("Transformando dados...")
        
        for name, preprocessor in self.steps:
            X = preprocessor.transform(X)
        
        return X
    
    def fit_transform(self, X: pd.DataFrame) -> Union[pd.DataFrame, np.ndarray]:
        return self.fit(X).transform(X)
    
    def log(self, message: str):
        if self.verbose:
            logger.info(message)


def build_default_pipeline() -> PreprocessingPipeline:
    pipeline = PreprocessingPipeline()
    
    pipeline.add_step("missing_values", MissingValueHandler(strategy="mean"))
    pipeline.add_step("outliers", OutlierRemover(method="iqr", threshold=3.0))
    pipeline.add_step("normalization", NormalizeScaler(method="standard"))
    
    return pipeline


def preprocess_data(
    X: pd.DataFrame,
    pipeline: Optional[PreprocessingPipeline] = None,
    fit: bool = True
) -> Union[pd.DataFrame, np.ndarray]:
    if pipeline is None:
        pipeline = build_default_pipeline()
    
    if fit:
        return pipeline.fit_transform(X)
    else:
        return pipeline.transform(X)