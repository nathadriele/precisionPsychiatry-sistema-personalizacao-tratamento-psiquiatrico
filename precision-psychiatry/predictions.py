"""
Módulo de Orquestração de Predições
Integra feature engineering, modelos ML e gera predições clínicas
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime
import logging

from ml_models import (
    PredictionResult,
    TherapeuticResponsePredictor,
    MedicationResponsePredictor,
    TreatmentResistanceClassifier
)
from feature_engineering import FeatureEngineer, FeatureSet

logger = logging.getLogger(__name__)


@dataclass
class ComprehensivePrediction:
    """Predição clínica abrangente para um paciente"""
    patient_id: str
    prediction_date: datetime
    
    # Predições de modelo
    therapeutic_response: PredictionResult
    treatment_resistance: PredictionResult
    medication_responses: List[PredictionResult] = field(default_factory=list)
    
    # Features utilizadas
    feature_set: Optional[FeatureSet] = None
    
    # Síntese clínica
    clinical_summary: str = ""
    recommendations: List[str] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)
    protective_factors: List[str] = field(default_factory=list)
    
    # Confiança geral
    overall_confidence: float = 0.0
    
    # Data completude
    data_completeness_percentage: float = 0.0


class PredictionOrchestrator:
    """Orquestra o pipeline completo de predições"""
    
    def __init__(self):
        self.logger = logger
        
        # Inicializar feature engineer
        self.feature_engineer = FeatureEngineer()
        
        # Inicializar modelos
        self.therapeutic_response_model = TherapeuticResponsePredictor()
        self.medication_response_model = MedicationResponsePredictor()
        self.resistance_classifier = TreatmentResistanceClassifier()
        
        # Medicações suportadas
        self.available_medications = [
            "sertraline",
            "escitalopram",
            "venlafaxine",
            "bupropion",
            "aripiprazole"
        ]
    
    def generate_predictions(
        self,
        patient_id: str,
        genetic_profile: Optional[Dict] = None,
        neurobiological_markers: Optional[Dict] = None,
        psychosocial_factors: Optional[Dict] = None,
        clinical_assessment: Optional[Dict] = None
    ) -> ComprehensivePrediction:
        """
        Gera predições clínicas abrangentes para um paciente
        """
        
        self.logger.info(f"Iniciando pipeline de predições para {patient_id}")
        
        # Etapa 1: Feature Engineering
        self.logger.info("Etapa 1: Engenharia de features")
        feature_set = self.feature_engineer.engineer_features(
            patient_id=patient_id,
            genetic_profile=genetic_profile,
            neurobiological_markers=neurobiological_markers,
            psychosocial_factors=psychosocial_factors,
            clinical_assessment=clinical_assessment
        )
        
        # Calcular completude de dados
        data_completeness = 100.0 * (1.0 - len(feature_set.missing_values) / 4.0)
        
        # Etapa 2: Predições de Modelos
        self.logger.info("Etapa 2: Executando modelos preditivos")
        
        # Predição de resposta terapêutica geral
        therapeutic_response = self.therapeutic_response_model.predict(
            features=feature_set.feature_vector,
            patient_id=patient_id
        )
        
        # Predição de resistência ao tratamento
        treatment_resistance = self.resistance_classifier.predict(
            features=feature_set.feature_vector,
            patient_id=patient_id
        )
        
        # Predições específicas por medicamento
        medication_predictions = []
        for med in self.available_medications:
            med_prediction = self.medication_response_model.predict(
                features=feature_set.feature_vector,
                medication_name=med,
                patient_id=patient_id
            )
            medication_predictions.append(med_prediction)
        
        # Etapa 3: Síntese Clínica
        self.logger.info("Etapa 3: Síntese clínica")
        
        risk_factors = self._identify_risk_factors(feature_set.feature_vector)
        protective_factors = self._identify_protective_factors(feature_set.feature_vector)
        recommendations = self._generate_recommendations(
            therapeutic_response,
            treatment_resistance,
            medication_predictions,
            risk_factors
        )
        clinical_summary = self._generate_clinical_summary(
            therapeutic_response,
            treatment_resistance,
            risk_factors,
            protective_factors
        )
        
        # Calcular confiança geral
        overall_confidence = (
            therapeutic_response.confidence_score * 0.4 +
            treatment_resistance.confidence_score * 0.3 +
            (sum(m.confidence_score for m in medication_predictions) / len(medication_predictions) if medication_predictions else 0.5) * 0.3
        )
        
        # Etapa 4: Compilar Resultado Final
        prediction = ComprehensivePrediction(
            patient_id=patient_id,
            prediction_date=datetime.now(),
            therapeutic_response=therapeutic_response,
            treatment_resistance=treatment_resistance,
            medication_responses=medication_predictions,
            feature_set=feature_set,
            clinical_summary=clinical_summary,
            recommendations=recommendations,
            risk_factors=risk_factors,
            protective_factors=protective_factors,
            overall_confidence=overall_confidence,
            data_completeness_percentage=data_completeness
        )
        
        self.logger.info(f"Predições concluídas para {patient_id}. "
                        f"Confiança geral: {overall_confidence:.0%}")
        
        return prediction
    
    def _identify_risk_factors(self, features: Dict[str, float]) -> List[str]:
        """Identifica fatores de risco associados aos features"""
        
        risk_factors = []
        
        # Fatores genômicos
        genomic_risk = features.get("genomic_risk_aggregate", 0.5)
        if genomic_risk > 0.65:
            risk_factors.append("Perfil genômico desfavorável para resposta a SSRI")
        
        # Disfunção HPA
        hpa = features.get("hpa_overactivation", 0.5)
        if hpa > 0.70:
            risk_factors.append("Disfunção significativa do eixo HPA (possível depressão melancólica)")
        
        # Inflamação
        inflammation = features.get("inflammation_aggregate", 0.3)
        if inflammation > 0.55:
            risk_factors.append("Marcadores inflamatórios elevados (depressão inflamatória)")
        
        # Interação trauma-gene
        trauma_interaction = features.get("fkbp5_trauma_interaction", 0.3)
        if trauma_interaction > 0.55:
            risk_factors.append("Interação significativa gene-ambiente com histórico de trauma")
        
        # Fatores psicossociais
        vulnerability = features.get("psychosocial_vulnerability_index", 0.5)
        if vulnerability > 0.65:
            risk_factors.append("Vulnerabilidade psicossocial significativa")
        
        isolation = features.get("social_isolation", 0.0)
        if isolation > 0.5:
            risk_factors.append("Isolamento social detectado")
        
        # Qualidade de vida
        qol = features.get("quality_of_life_normalized", 0.5)
        if qol < 0.35:
            risk_factors.append("Qualidade de vida significativamente reduzida")
        
        # Problemas cognitivos
        cognitive = features.get("cognitive_impairment", 0.3)
        if cognitive > 0.60:
            risk_factors.append("Deficiência cognitiva detectada")
        
        return risk_factors or ["Nenhum fator de risco maior identificado"]
    
    def _identify_protective_factors(self, features: Dict[str, float]) -> List[str]:
        """Identifica fatores protetores"""
        
        protective_factors = []
        
        # Suporte social
        support = features.get("social_support_score", 0.5)
        if support > 0.65:
            protective_factors.append("Suporte social forte")
        
        # Qualidade de vida
        qol = features.get("quality_of_life_normalized", 0.5)
        if qol > 0.65:
            protective_factors.append("Qualidade de vida relativamente preservada")
        
        # Propósito
        purpose = features.get("sense_of_purpose", 0.5)
        if purpose > 0.65:
            protective_factors.append("Forte senso de propósito e significado")
        
        # Engajamento em atividades
        activities = features.get("activities_engagement", 0.5)
        if activities > 0.60:
            protective_factors.append("Bom engajamento em atividades")
        
        # Perfil genômico favorável
        genomic_risk = features.get("genomic_risk_aggregate", 0.5)
        if genomic_risk < 0.40:
            protective_factors.append("Perfil genômico favorável para resposta terapêutica")
        
        # HPA normal
        hpa = features.get("hpa_overactivation", 0.5)
        if hpa < 0.40:
            protective_factors.append("Funcionamento normal do eixo HPA")
        
        # Sono adequado
        sleep = features.get("sleep_quality_normalized", 0.5)
        if sleep > 0.65:
            protective_factors.append("Qualidade do sono preservada")
        
        return protective_factors or ["Avaliação necessária de fatores protetores"]
    
    def _generate_recommendations(
        self,
        therapeutic_response: PredictionResult,
        treatment_resistance: PredictionResult,
        medication_predictions: List[PredictionResult],
        risk_factors: List[str]
    ) -> List[str]:
        """Gera recomendações clínicas personalizadas"""
        
        recommendations = []
        
        # Recomendações baseadas em resposta terapêutica
        response_prob = therapeutic_response.predicted_response_probability
        
        if response_prob > 0.75:
            recommendations.append("Prognóstico favorável para resposta ao tratamento farmacológico")
        elif response_prob > 0.55:
            recommendations.append("Resposta moderada esperada - considerar combinação de terapias")
        else:
            recommendations.append("Prognóstico desafiador - considerar abordagem multimodal agressiva")
        
        # Recomendações baseadas em resistência
        resistance_prob = treatment_resistance.predicted_response_probability
        
        if resistance_prob > 0.70:
            recommendations.append("ALTO RISCO DE REFRATARIEDADE:")
            recommendations.append("  - Considerar referência para especialista em depressão refratária")
            recommendations.append("  - Avaliar para terapias neuromoduladores (TMS, TEC)")
            recommendations.append("  - Considerar combinações de medicamentos desde o início")
        elif resistance_prob > 0.50:
            recommendations.append("Risco moderado de refratariedade - iniciar com doses ótimas")
        
        # Recomendações por medicamento
        if medication_predictions:
            best_medications = sorted(
                medication_predictions,
                key=lambda x: x.predicted_response_probability,
                reverse=True
            )[:3]
            
            recommendations.append("\nMedicamentos recomendados por probabilidade de resposta:")
            for i, med_pred in enumerate(best_medications, 1):
                med_name = med_pred.model_type.split("(")[1].rstrip(")")
                recommendations.append(
                    f"  {i}. {med_name}: {med_pred.predicted_response_probability:.0%}"
                )
        
        # Recomendações por fatores de risco
        if any("trauma" in rf.lower() for rf in risk_factors):
            recommendations.append("\nPor história de trauma:")
            recommendations.append("  - Considerar TEC ou terapia específica para trauma")
            recommendations.append("  - Psicoterapia especializada em trauma (EMDR, CPT)")
        
        if any("inflamação" in rf.lower() for rf in risk_factors):
            recommendations.append("\nPor marcadores inflamatórios elevados:")
            recommendations.append("  - Avaliar dieta anti-inflamatória e ômega-3")
            recommendations.append("  - Considerar avaliação de infecções ocultas")
            recommendations.append("  - Ampliar investigação de comorbidades médicas")
        
        if any("isolamento" in rf.lower() for rf in risk_factors):
            recommendations.append("\nPor isolamento social:")
            recommendations.append("  - Priorizar terapia em grupo ou interpessoal (IPT)")
            recommendations.append("  - Ativação comportamental estruturada")
            recommendations.append("  - Envolvimento de apoio comunitário")
        
        # Recomendação geral de monitoramento
        recommendations.append("\nMonitoramento:")
        recommendations.append("  - Avaliação semanal nas primeiras 4 semanas")
        recommendations.append("  - Escalas estruturadas (PHQ-9, MADRS)")
        recommendations.append("  - Vigilância para ideação suicida")
        
        return recommendations
    
    def _generate_clinical_summary(
        self,
        therapeutic_response: PredictionResult,
        treatment_resistance: PredictionResult,
        risk_factors: List[str],
        protective_factors: List[str]
    ) -> str:
        """Gera sumário clínico narrativo"""
        
        summary = f"""
PREDIÇÃO CLÍNICA PERSONALIZADA
{'='*60}

RESPOSTA TERAPÊUTICA ESPERADA:
{therapeutic_response.predicted_response_category} 
Probabilidade: {therapeutic_response.predicted_response_probability:.0%}
Confiança: {therapeutic_response.confidence_score:.0%}

RISCO DE REFRATARIEDADE:
{treatment_resistance.predicted_response_category}
Probabilidade: {treatment_resistance.predicted_response_probability:.0%}
Confiança: {treatment_resistance.confidence_score:.0%}

FATORES DE RISCO IDENTIFICADOS:
{chr(10).join(f'• {rf}' for rf in risk_factors[:5])}

FATORES PROTETORES IDENTIFICADOS:
{chr(10).join(f'• pf' for pf in protective_factors[:5])}

RACIOCÍNIO DO MODELO:
{therapeutic_response.prediction_reasoning}
        """
        
        return summary.strip()
    
    def get_model_registry(self) -> Dict:
        """Retorna informações sobre modelos disponíveis"""
        
        return {
            "therapeutic_response": {
                "model": "Therapeutic Response Predictor",
                "type": "Ensemble (Genomic + Neurobiological + Psychosocial)",
                "status": "Production"
            },
            "treatment_resistance": {
                "model": "Treatment Resistance Classifier",
                "type": "Random Forest",
                "status": "Production"
            },
            "medication_response": {
                "model": "Medication Response Predictor",
                "type": "Logistic Regression",
                "medications": self.available_medications,
                "status": "Production"
            }
        }