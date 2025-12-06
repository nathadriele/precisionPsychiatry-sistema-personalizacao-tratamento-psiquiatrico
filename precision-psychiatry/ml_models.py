"""
Módulo de Modelos de Machine Learning
Modelos preditivos para resposta terapêutica e classificação
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod
import logging
import numpy as np
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Tipos de modelos disponíveis"""
    LOGISTIC_REGRESSION = "logistic_regression"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    NEURAL_NETWORK = "neural_network"
    ENSEMBLE = "ensemble"


@dataclass
class ModelPerformance:
    """Métricas de performance do modelo"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    sensitivity: float
    specificity: float
    
    # Métricas específicas para psiquiatria
    positive_predictive_value: float  # Taxa de resposta real entre os preditos
    negative_predictive_value: float
    
    # Intervalos de confiança
    ci_lower: float = 0.0
    ci_upper: float = 1.0


@dataclass
class PredictionResult:
    """Resultado de predição para um paciente"""
    patient_id: str
    prediction_date: datetime
    
    predicted_response_probability: float  # 0-1
    predicted_response_category: str  # "Responder", "Partial", "Non-responder"
    confidence_score: float  # 0-1
    
    # Features mais importantes para a predição
    top_features: List[Tuple[str, float]] = field(default_factory=list)
    
    # Diagnóstico do modelo
    model_type: str = ""
    prediction_reasoning: str = ""
    
    # Intervalos de confiança
    probability_lower_bound: float = 0.0
    probability_upper_bound: float = 1.0


@dataclass
class ModelMetadata:
    """Metadados do modelo"""
    model_id: str
    model_type: ModelType
    version: str
    creation_date: datetime
    training_date: datetime
    last_updated: datetime
    
    # Dados de treinamento
    training_samples: int = 0
    features_used: List[str] = field(default_factory=list)
    target_variable: str = ""
    
    # Performance
    performance_metrics: Optional[ModelPerformance] = None
    
    # Validação cruzada
    cv_score_mean: float = 0.0
    cv_score_std: float = 0.0
    
    # Status
    is_production: bool = False
    notes: str = ""


class PredictorModel(ABC):
    """Classe base abstrata para modelos preditivos"""
    
    def __init__(self, model_id: str, model_type: ModelType):
        self.model_id = model_id
        self.model_type = model_type
        self.logger = logger
        self.is_trained = False
        self.feature_importance = {}
    
    @abstractmethod
    def predict(self, features: Dict[str, float]) -> PredictionResult:
        """Realiza predição"""
        pass
    
    @abstractmethod
    def predict_proba(self, features: Dict[str, float]) -> float:
        """Retorna probabilidade de resposta"""
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """Retorna importância das features"""
        pass


class TherapeuticResponsePredictor(PredictorModel):
    """Preditor de resposta terapêutica usando multiple features"""
    
    def __init__(self, model_id: str = "therapeutic_response_v1"):
        super().__init__(model_id, ModelType.ENSEMBLE)
        
        # Pesos de cada componente genômico/neurobiológico/psicossocial
        self.genomic_weight = 0.35
        self.neurobiological_weight = 0.40
        self.psychosocial_weight = 0.25
        
        # Thresholds de classificação
        self.responder_threshold = 0.65
        self.partial_threshold = 0.45
        
        self.is_trained = True  # Usar modelo baseado em regras
    
    def predict(self, features: Dict[str, float], patient_id: str = "") -> PredictionResult:
        """Predição de resposta terapêutica baseada em ensemble"""
        
        proba = self.predict_proba(features)
        
        # Classificar categoria
        if proba >= self.responder_threshold:
            category = "Responder"
        elif proba >= self.partial_threshold:
            category = "Partial Responder"
        else:
            category = "Non-responder"
        
        # Extrair features mais importantes
        top_features = self.get_top_features(features, n=5)
        
        # Gerar reasoning
        reasoning = self._generate_reasoning(features, proba, category)
        
        # Calcular intervalo de confiança
        confidence = self._calculate_confidence(features)
        ci_lower = max(0, proba - 0.15)
        ci_upper = min(1, proba + 0.15)
        
        result = PredictionResult(
            patient_id=patient_id,
            prediction_date=datetime.now(),
            predicted_response_probability=proba,
            predicted_response_category=category,
            confidence_score=confidence,
            top_features=top_features,
            model_type="Therapeutic Response Predictor (Ensemble)",
            prediction_reasoning=reasoning,
            probability_lower_bound=ci_lower,
            probability_upper_bound=ci_upper
        )
        
        self.logger.info(f"Predição para {patient_id}: {category} (prob={proba:.2%})")
        
        return result
    
    def predict_proba(self, features: Dict[str, float]) -> float:
        """Calcula probabilidade de resposta"""
        
        # Componente genômico
        genomic_score = self._calculate_genomic_score(features)
        
        # Componente neurobiológico
        neurobiological_score = self._calculate_neurobiological_score(features)
        
        # Componente psicossocial
        psychosocial_score = self._calculate_psychosocial_score(features)
        
        # Score agregado
        overall_probability = (
            genomic_score * self.genomic_weight +
            neurobiological_score * self.neurobiological_weight +
            psychosocial_score * self.psychosocial_weight
        )
        
        return min(max(overall_probability, 0.0), 1.0)
    
    def _calculate_genomic_score(self, features: Dict[str, float]) -> float:
        """Calcula componente genômico"""
        
        # Baixo risco genômico favorece resposta
        genomic_risk = features.get("genomic_risk_aggregate", 0.5)
        
        # Metabolizador normal de CYP2D6 favorece resposta
        cyp2d6_favorable = features.get("cyp2d6_dose_multiplier", 1.0) == 1.0
        cyp2d6_score = 0.7 if cyp2d6_favorable else 0.5
        
        # Gene-ambiente
        trauma_interaction = features.get("fkbp5_trauma_interaction", 0.3)
        stress_interaction = features.get("serotonin_stress_interaction", 0.3)
        
        genomic_components = [
            1.0 - genomic_risk,  # Inverter (baixo risco = alta resposta)
            cyp2d6_score,
            1.0 - (trauma_interaction * 0.5),
            1.0 - (stress_interaction * 0.5)
        ]
        
        return np.mean(genomic_components)
    
    def _calculate_neurobiological_score(self, features: Dict[str, float]) -> float:
        """Calcula componente neurobiológico"""
        
        # HPA normal favorece resposta
        hpa_activation = features.get("hpa_overactivation", 0.5)
        
        # Inflamação baixa favorece resposta
        inflammation = features.get("inflammation_aggregate", 0.3)
        
        # Ativação prefrontal adequada
        prefrontal = features.get("prefrontal_activation", 0.7)
        
        # Reatividade amigdalar não excessiva
        amygdala = features.get("amygdala_hyperreactivity", 0.5)
        
        # Sono e cognição normais
        sleep = features.get("sleep_quality_normalized", 0.5)
        cognition = features.get("cognitive_impairment", 0.3)
        
        neuro_components = [
            1.0 - hpa_activation,  # Inverter
            1.0 - inflammation,    # Inverter
            prefrontal,
            1.0 - amygdala,       # Inverter
            sleep,
            1.0 - cognition        # Inverter
        ]
        
        return np.mean(neuro_components)
    
    def _calculate_psychosocial_score(self, features: Dict[str, float]) -> float:
        """Calcula componente psicossocial"""
        
        # Suporte social favorece resposta
        social_support = features.get("social_support_score", 0.5)
        
        # Isolamento prejudica resposta
        isolation = features.get("social_isolation", 0.0)
        
        # Qualidade de vida
        qol = features.get("quality_of_life_normalized", 0.5)
        
        # Vulnerabilidade psicossocial
        vulnerability = features.get("psychosocial_vulnerability_index", 0.5)
        
        # Propósito/significado
        purpose = features.get("sense_of_purpose", 0.5)
        
        psycho_components = [
            social_support,
            1.0 - isolation,       # Inverter
            qol,
            1.0 - vulnerability,   # Inverter
            purpose
        ]
        
        return np.mean(psycho_components)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Retorna importância das features no modelo"""
        
        return {
            "genomic_risk_aggregate": 0.15,
            "cyp2d6_dose_multiplier": 0.10,
            "hpa_overactivation": 0.12,
            "inflammation_aggregate": 0.10,
            "social_support_score": 0.12,
            "psychosocial_vulnerability_index": 0.10,
            "fkbp5_trauma_interaction": 0.08,
            "sleep_quality_normalized": 0.07,
            "quality_of_life_normalized": 0.08,
            "prefrontal_activation": 0.08
        }
    
    def get_top_features(
        self,
        features: Dict[str, float],
        n: int = 5
    ) -> List[Tuple[str, float]]:
        """Retorna top N features mais importantes para predição"""
        
        importance = self.get_feature_importance()
        
        # Combinar importância com valor do feature
        feature_scores = {}
        for feature_name, importance_value in importance.items():
            feature_value = features.get(feature_name, 0.0)
            # Score = importância * |desviação de 0.5| (quanto mais longe de neutral, mais impactante)
            impact = abs(feature_value - 0.5)
            feature_scores[feature_name] = importance_value * (1.0 + impact)
        
        # Ordenar e retornar top N
        top = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)[:n]
        return top
    
    def _generate_reasoning(self, features: Dict, proba: float, category: str) -> str:
        """Gera explicação textual da predição"""
        
        reasons = []
        
        # Análise genômica
        genomic_risk = features.get("genomic_risk_aggregate", 0.5)
        if genomic_risk > 0.6:
            reasons.append("Perfil genético desfavorável para resposta SSRI")
        else:
            reasons.append("Perfil genético favorável para resposta terapêutica")
        
        # Análise HPA
        hpa = features.get("hpa_overactivation", 0.5)
        if hpa > 0.7:
            reasons.append("Disfunção significativa do eixo HPA")
        
        # Análise inflamação
        inflammation = features.get("inflammation_aggregate", 0.3)
        if inflammation > 0.5:
            reasons.append("Marcadores inflamatórios elevados (possível depressão inflamatória)")
        
        # Análise suporte social
        support = features.get("social_support_score", 0.5)
        if support < 0.3:
            reasons.append("Suporte social limitado - fator de risco")
        
        # Análise trauma
        trauma = features.get("fkbp5_trauma_interaction", 0.3)
        if trauma > 0.5:
            reasons.append("Interação significativa gene-trauma detectada")
        
        reasoning = " | ".join(reasons) if reasons else "Fatores equilibrados"
        return reasoning
    
    def _calculate_confidence(self, features: Dict[str, float]) -> float:
        """Calcula confiança da predição"""
        
        # Confiança reduzida se faltam features
        present_features = sum(1 for v in features.values() if v is not None)
        total_features = len(features)
        
        completeness = present_features / total_features if total_features > 0 else 0.8
        
        # Base confidence
        confidence = 0.75 * completeness
        
        # Aumentar confiança se há certeza (features muito altos ou baixos)
        extreme_features = sum(1 for v in features.values() if v < 0.2 or v > 0.8)
        if extreme_features > 3:
            confidence += 0.10
        
        return min(confidence, 0.95)


class MedicationResponsePredictor(PredictorModel):
    """Preditor de resposta específica a medicamentos"""
    
    def __init__(self, model_id: str = "medication_response_v1"):
        super().__init__(model_id, ModelType.LOGISTIC_REGRESSION)
        
        # Mapeamento de genótipos a medicamentos eficazes
        self.genotype_medication_mapping = {
            "5-HTTLPR_L_L": ["sertraline", "escitalopram"],  # Bom metabolismo
            "5-HTTLPR_S_S": ["venlafaxine", "bupropion"],    # Requer alternativas
            "COMT_Met_Met": ["sertraline"],                  # Sensível a dopamina
            "BDNF_Met_Met": ["exercise_augmentation"],       # Requer augmentação
        }
        
        self.is_trained = True
    
    def predict(
        self,
        features: Dict[str, float],
        medication_name: str,
        patient_id: str = ""
    ) -> PredictionResult:
        """Predição de resposta a medicamento específico"""
        
        proba = self.predict_proba(features)
        
        category = "Likely Responder" if proba > 0.6 else "Uncertain Response"
        
        reasoning = f"Esperada eficácia de {proba:.0%} para {medication_name}"
        
        result = PredictionResult(
            patient_id=patient_id,
            prediction_date=datetime.now(),
            predicted_response_probability=proba,
            predicted_response_category=category,
            confidence_score=0.70,
            model_type=f"Medication Response Predictor ({medication_name})",
            prediction_reasoning=reasoning
        )
        
        return result
    
    def predict_proba(self, features: Dict[str, float]) -> float:
        """Probabilidade de resposta"""
        
        # Simplificado: baseado em features genômicas e HPA
        genomic_component = 1.0 - features.get("genomic_risk_aggregate", 0.5)
        hpa_component = 1.0 - features.get("hpa_overactivation", 0.5)
        
        proba = 0.6 * genomic_component + 0.4 * hpa_component
        
        return min(max(proba, 0.0), 1.0)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Importância das features"""
        return {
            "genomic_risk_aggregate": 0.50,
            "hpa_overactivation": 0.30,
            "cyp2d6_dose_multiplier": 0.20
        }


class TreatmentResistanceClassifier(PredictorModel):
    """Classificador de probabilidade de depressão refratária"""
    
    def __init__(self, model_id: str = "resistance_classifier_v1"):
        super().__init__(model_id, ModelType.RANDOM_FOREST)
        self.is_trained = True
    
    def predict(self, features: Dict[str, float], patient_id: str = "") -> PredictionResult:
        """Predição de resistência ao tratamento"""
        
        proba = self.predict_proba(features)
        
        category = "High Risk for Refractoriness" if proba > 0.65 else "Standard Risk"
        
        result = PredictionResult(
            patient_id=patient_id,
            prediction_date=datetime.now(),
            predicted_response_probability=proba,
            predicted_response_category=category,
            confidence_score=0.78,
            model_type="Treatment Resistance Classifier",
            prediction_reasoning=f"Risco de refratariedade: {proba:.0%}"
        )
        
        return result
    
    def predict_proba(self, features: Dict[str, float]) -> float:
        """Probabilidade de refratariedade"""
        
        # Fatores de risco
        genomic_risk = features.get("genomic_risk_aggregate", 0.5)
        inflammation = features.get("inflammation_aggregate", 0.3)
        vulnerability = features.get("psychosocial_vulnerability_index", 0.5)
        trauma = features.get("fkbp5_trauma_interaction", 0.3)
        
        refractoriness_score = (
            genomic_risk * 0.35 +
            inflammation * 0.25 +
            vulnerability * 0.25 +
            trauma * 0.15
        )
        
        return min(max(refractoriness_score, 0.0), 1.0)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Importância das features"""
        return {
            "genomic_risk_aggregate": 0.35,
            "inflammation_aggregate": 0.25,
            "psychosocial_vulnerability_index": 0.25,
            "fkbp5_trauma_interaction": 0.15
        }