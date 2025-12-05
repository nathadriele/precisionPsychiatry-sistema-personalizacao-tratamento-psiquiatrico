"""
Explicabilidade e interpretabilidade de modelos.
Integra SHAP, LIME, e outras técnicas.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class SHAPExplainer:
    """Explicabilidade usando SHAP."""
    
    def __init__(self, model, X_background: np.ndarray = None):
        """
        Inicializar explicador SHAP.
        
        Args:
            model: Modelo treinado
            X_background: Dados de background para SHAP
        """
        self.model = model
        self.X_background = X_background
        self.explainer = None
    
    def fit(self):
        """Ajustar explicador SHAP."""
        try:
            import shap
            
            # Usar TreeExplainer se possível (mais rápido)
            if hasattr(self.model, 'predict_proba') and hasattr(self.model, 'predict'):
                self.explainer = shap.TreeExplainer(self.model)
            else:
                # Fallback para KernelExplainer
                self.explainer = shap.KernelExplainer(
                    self.model.predict_proba,
                    self.X_background
                )
            
            logging.info("✓ Explicador SHAP inicializado")
            
        except ImportError:
            logging.warning("SHAP não instalado. Pulando inicialização.")
    
    def explain_instance(
        self,
        X_instance: np.ndarray,
        feature_names: Optional[List[str]] = None,
        top_n: int = 10
    ) -> Dict[str, Any]:
        """
        Explicar predição individual.
        
        Args:
            X_instance: Instância a explicar
            feature_names: Nomes das features
            top_n: Número de top features
        
        Returns:
            Dicionário com explicação
        """
        if self.explainer is None:
            return {"error": "Explicador não foi inicializado"}
        
        try:
            import shap
            
            # Calcular SHAP values
            shap_values = self.explainer.shap_values(X_instance)
            
            # Lidar com múltiplas classes
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Classe positiva
            
            # Valores absolutos para ranking
            abs_shap = np.abs(shap_values).flatten()
            top_indices = np.argsort(abs_shap)[-top_n:][::-1]
            
            # Construir explicação
            explanation = {
                "instance_index": 0,
                "top_features": [],
                "shap_values": {}
            }
            
            for idx in top_indices:
                feature_name = feature_names[idx] if feature_names else f"feature_{idx}"
                shap_val = float(shap_values[idx])
                
                explanation["top_features"].append({
                    "feature": feature_name,
                    "shap_value": shap_val,
                    "abs_shap_value": abs(shap_val)
                })
                
                explanation["shap_values"][feature_name] = shap_val
            
            return explanation
            
        except Exception as e:
            logging.error(f"Erro ao explicar com SHAP: {e}")
            return {"error": str(e)}
    
    def explain_dataset(
        self,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Calcular SHAP values para todo dataset.
        
        Args:
            X: Dataset
            feature_names: Nomes das features
        
        Returns:
            Dicionário com SHAP values médios
        """
        try:
            import shap
            
            shap_values = self.explainer.shap_values(X)
            
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            
            # SHAP value médio por feature
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            
            # Ranking
            ranking = np.argsort(mean_abs_shap)[::-1]
            
            importance = {}
            for idx in ranking:
                feature_name = feature_names[idx] if feature_names else f"feature_{idx}"
                importance[feature_name] = float(mean_abs_shap[idx])
            
            return importance
            
        except Exception as e:
            logging.error(f"Erro ao calcular SHAP dataset: {e}")
            return {}


class FeatureImportanceExplainer:
    """Explicabilidade usando feature importance."""
    
    @staticmethod
    def get_importance(model, feature_names: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Obter importância de features do modelo.
        
        Args:
            model: Modelo treinado
            feature_names: Nomes das features
        
        Returns:
            Dicionário de importâncias
        """
        if not hasattr(model, 'feature_importances_'):
            return {}
        
        importances = model.feature_importances_
        
        if feature_names:
            return dict(zip(feature_names, importances))
        else:
            return {f"feature_{i}": imp for i, imp in enumerate(importances)}
    
    @staticmethod
    def get_top_features(
        model,
        feature_names: Optional[List[str]] = None,
        top_n: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Obter top N features.
        
        Args:
            model: Modelo
            feature_names: Nomes das features
            top_n: Número de top features
        
        Returns:
            Lista de (feature, importância)
        """
        importance = FeatureImportanceExplainer.get_importance(model, feature_names)
        
        if not importance:
            return []
        
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        return sorted_features[:top_n]


class PermutationExplainer:
    """Explicabilidade usando permutação de features."""
    
    @staticmethod
    def get_importance(
        model,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        n_repeats: int = 10
    ) -> Dict[str, float]:
        """
        Calcular importância por permutação.
        
        Args:
            model: Modelo treinado
            X: Features
            y: Target
            feature_names: Nomes das features
            n_repeats: Número de repetições
        
        Returns:
            Dicionário de importâncias
        """
        from sklearn.inspection import permutation_importance
        
        result = permutation_importance(
            model,
            X,
            y,
            n_repeats=n_repeats,
            random_state=42
        )
        
        importances = {}
        for i, imp in enumerate(result.importances_mean):
            feature_name = feature_names[i] if feature_names else f"feature_{i}"
            importances[feature_name] = float(imp)
        
        return importances


class PartialDependenceExplainer:
    """Explicabilidade usando partial dependence plots."""
    
    @staticmethod
    def get_partial_dependence(
        model,
        X: pd.DataFrame,
        feature: str,
        grid_resolution: int = 50
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcular partial dependence.
        
        Args:
            model: Modelo treinado
            X: Features
            feature: Feature a analisar
            grid_resolution: Resolução do grid
        
        Returns:
            Tupla (valores_grid, dependências)
        """
        from sklearn.inspection import partial_dependence
        
        feature_idx = list(X.columns).index(feature)
        
        pd_result = partial_dependence(
            model,
            X.values,
            features=[feature_idx],
            grid_resolution=grid_resolution
        )
        
        return pd_result['grid_values'][0], pd_result['average'][0]


class ComprehensiveExplainer:
    """Explicador completo que combina múltiplas técnicas."""
    
    def __init__(self, model, X_train: np.ndarray = None, feature_names: List[str] = None):
        """
        Inicializar.
        
        Args:
            model: Modelo treinado
            X_train: Dados de treinamento (para SHAP)
            feature_names: Nomes das features
        """
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names
        
        self.shap_explainer = SHAPExplainer(model, X_train)
        try:
            self.shap_explainer.fit()
            self.has_shap = True
        except:
            self.has_shap = False
    
    def explain_prediction(
        self,
        X_instance: np.ndarray,
        top_n: int = 10
    ) -> Dict[str, Any]:
        """
        Explicação completa de uma predição.
        
        Args:
            X_instance: Instância
            top_n: Top features
        
        Returns:
            Explicação completa
        """
        explanation = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "methods": {}
        }
        
        # Feature Importance
        fi_importance = FeatureImportanceExplainer.get_top_features(
            self.model,
            self.feature_names,
            top_n
        )
        explanation["methods"]["feature_importance"] = [
            {"feature": f, "importance": float(i)} for f, i in fi_importance
        ]
        
        # SHAP
        if self.has_shap:
            shap_exp = self.shap_explainer.explain_instance(
                X_instance.reshape(1, -1),
                self.feature_names,
                top_n
            )
            explanation["methods"]["shap"] = shap_exp
        
        return explanation


def explain_model(
    model,
    X_train: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None,
    X_instance: Optional[np.ndarray] = None,
    method: str = 'comprehensive'
) -> Dict[str, Any]:
    """
    Explicar modelo facilmente.
    
    Args:
        model: Modelo treinado
        X_train: Dados de treinamento
        feature_names: Nomes das features
        X_instance: Instância para explicar
        method: 'comprehensive', 'shap', 'feature_importance', 'permutation'
    
    Returns:
        Explicação
    """
    if method == 'comprehensive':
        explainer = ComprehensiveExplainer(model, X_train, feature_names)
        return explainer.explain_prediction(X_instance)
    
    elif method == 'shap':
        explainer = SHAPExplainer(model, X_train)
        explainer.fit()
        return explainer.explain_instance(X_instance, feature_names)
    
    elif method == 'feature_importance':
        return FeatureImportanceExplainer.get_importance(model, feature_names)
    
    elif method == 'permutation':
        return PermutationExplainer.get_importance(model, X_train, None, feature_names)
    
    else:
        raise ValueError(f"Método desconhecido: {method}")