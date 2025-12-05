"""
Classe base para todos os modelos de ML.
Define interface consistente e funcionalidades comuns.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
import logging

logger = logging.getLogger(__name__)


class BaseModel(ABC, BaseEstimator, ClassifierMixin):
    """Classe base abstrata para modelos de ML."""
    
    def __init__(self, random_state: int = 42, verbose: bool = True):
        """
        Inicializar modelo.
        
        Args:
            random_state: Seed para reprodutibilidade
            verbose: Verbosidade
        """
        self.random_state = random_state
        self.verbose = verbose
        self.model = None
        self.is_fitted = False
        self.feature_names = None
        self.n_features = None
        self.classes_ = np.array([0, 1])
    
    @abstractmethod
    def _build_model(self, **kwargs):
        """Construir modelo específico."""
        pass
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'BaseModel':
        """
        Treinar modelo.
        
        Args:
            X: Features
            y: Target
            **kwargs: Argumentos adicionais
        
        Returns:
            Self (para encadeamento)
        """
        self.log(f"Treinando {self.__class__.__name__}...")
        
        # Validar input
        if X is None or y is None:
            raise ValueError("X e y não podem ser None")
        
        if len(X) != len(y):
            raise ValueError(f"X e y têm tamanhos diferentes: {len(X)} vs {len(y)}")
        
        # Armazenar informações
        self.n_features = X.shape[1]
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        
        # Construir e treinar modelo
        self._build_model(**kwargs)
        self.model.fit(X, y)
        
        self.is_fitted = True
        self.log("✓ Modelo treinado com sucesso")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Fazer predições.
        
        Args:
            X: Features
        
        Returns:
            Array de predições
        """
        self._check_is_fitted()
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predições com probabilidades.
        
        Args:
            X: Features
        
        Returns:
            Array de probabilidades (n_samples, 2)
        """
        self._check_is_fitted()
        
        if not hasattr(self.model, 'predict_proba'):
            raise ValueError(f"{self.__class__.__name__} não suporta predict_proba")
        
        return self.model.predict_proba(X)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calcular score (acurácia).
        
        Args:
            X: Features
            y: Target
        
        Returns:
            Score
        """
        self._check_is_fitted()
        return self.model.score(X, y)
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Obter importância de features.
        
        Returns:
            Dicionário feature → importância
        """
        if not hasattr(self.model, 'feature_importances_'):
            return None
        
        importances = self.model.feature_importances_
        
        if self.feature_names:
            return dict(zip(self.feature_names, importances))
        else:
            return {f"feature_{i}": imp for i, imp in enumerate(importances)}
    
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """
        Obter parâmetros do modelo.
        
        Args:
            deep: Se True, retorna parâmetros dos sub-modelos
        
        Returns:
            Dicionário de parâmetros
        """
        params = {
            'random_state': self.random_state,
            'verbose': self.verbose
        }
        
        if self.model and hasattr(self.model, 'get_params'):
            if deep:
                params.update(self.model.get_params(deep=True))
        
        return params
    
    def set_params(self, **params) -> 'BaseModel':
        """
        Definir parâmetros do modelo.
        
        Args:
            **params: Parâmetros
        
        Returns:
            Self
        """
        for key, value in params.items():
            if key in ['random_state', 'verbose']:
                setattr(self, key, value)
            elif self.model and hasattr(self.model, 'set_params'):
                self.model.set_params(**{key: value})
        
        return self
    
    def _check_is_fitted(self):
        """Verificar se modelo foi treinado."""
        if not self.is_fitted:
            raise ValueError(f"{self.__class__.__name__} não foi treinado")
    
    def log(self, message: str):
        """Log condicional."""
        if self.verbose:
            logger.info(message)


class BinaryClassifierMixin:
    """Mixin para classificadores binários."""
    
    def get_classes(self) -> np.ndarray:
        """Obter classes do modelo."""
        return self.classes_
    
    def is_binary(self) -> bool:
        """Verificar se é classificação binária."""
        return len(self.classes_) == 2


class ProbabilisticMixin:
    """Mixin para modelos probabilísticos."""
    
    def get_probabilities(self, X: np.ndarray) -> np.ndarray:
        """Obter probabilidades."""
        return self.predict_proba(X)
    
    def get_log_probabilities(self, X: np.ndarray) -> np.ndarray:
        """Obter log-probabilidades."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            proba = self.predict_proba(X)
            return np.log(proba + 1e-15)


class InterpretableMixin:
    """Mixin para modelos interpretáveis."""
    
    def explain_prediction(self, X: np.ndarray, sample_idx: int = 0) -> Dict:
        """
        Explicar predição individual.
        
        Args:
            X: Features
            sample_idx: Índice da amostra
        
        Returns:
            Dicionário com explicação
        """
        importance = self.get_feature_importance()
        
        if importance is None:
            return {"message": "Modelo não suporta feature importance"}
        
        # Ordenar por importância
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        
        return {
            "top_features": sorted_features[:10],
            "feature_count": len(importance),
            "sample_index": sample_idx
        }