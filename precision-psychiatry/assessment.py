"""
Módulo de Avaliação Clínica e Diagnóstico
Responsável pela avaliação inicial e contínua de pacientes com depressão
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class DepressionSeverity(Enum):
    """Classificação de severidade da depressão"""
    MINIMAL = "minimal"
    MILD = "mild"
    MODERATE = "moderate"
    MODERATELY_SEVERE = "moderately_severe"
    SEVERE = "severe"


class TreatmentResistance(Enum):
    """Classificação de resistência ao tratamento"""
    TREATMENT_NAIVE = "treatment_naive"
    PARTIAL_RESPONDER = "partial_responder"
    NON_RESPONDER = "non_responder"
    REFRACTORY = "refractory"


@dataclass
class ClinicalAssessment:
    """Dados de avaliação clínica do paciente"""
    patient_id: str
    assessment_date: datetime
    
    # Escalas clínicas
    phq9_score: int  # PHQ-9 (0-27)
    hamilton_score: int  # HAM-D (0-52)
    madrs_score: int  # MADRS (0-60)
    
    # Características clínicas
    episode_duration_months: int
    num_previous_episodes: int
    age_of_onset: int
    
    # Comorbididades
    anxiety_disorder: bool = False
    substance_abuse: bool = False
    personality_disorder: bool = False
    
    # Histórico de tratamento
    medication_trials: List[str] = field(default_factory=list)
    psychotherapy_history: bool = False
    hospitalization_history: bool = False
    
    # Informações adicionais
    suicidal_ideation: bool = False
    family_history_depression: bool = False
    
    notes: str = ""


@dataclass
class AssessmentResult:
    """Resultado da avaliação"""
    patient_id: str
    assessment_date: datetime
    severity: DepressionSeverity
    treatment_resistance: TreatmentResistance
    risk_score: float  # 0-100
    confidence_score: float  # 0-1
    recommendations: List[str]
    clinical_notes: str


class ClinicalAssessor:
    """Classe para realizar avaliações clínicas"""
    
    def __init__(self):
        self.logger = logger
    
    def calculate_severity(self, phq9: int, hamilton: int, madrs: int) -> DepressionSeverity:
        """
        Calcula severidade da depressão baseado em múltiplas escalas
        
        PHQ-9: 0-4 (min), 5-9 (mild), 10-14 (mod), 15-19 (mod-sev), 20+ (sev)
        HAM-D: 0-7 (normal), 8-10 (min), 11-16 (mild), 17-23 (mod), 24+ (sev)
        MADRS: 0-6 (normal), 7-19 (mild), 20-34 (mod), 35-60 (severe)
        """
        
        # Normalizar escores para 0-100
        phq9_norm = (phq9 / 27) * 100
        hamilton_norm = (hamilton / 52) * 100
        madrs_norm = (madrs / 60) * 100
        
        # Média ponderada
        avg_severity = (phq9_norm * 0.3 + hamilton_norm * 0.4 + madrs_norm * 0.3)
        
        if avg_severity < 20:
            return DepressionSeverity.MINIMAL
        elif avg_severity < 35:
            return DepressionSeverity.MILD
        elif avg_severity < 50:
            return DepressionSeverity.MODERATE
        elif avg_severity < 70:
            return DepressionSeverity.MODERATELY_SEVERE
        else:
            return DepressionSeverity.SEVERE
    
    def calculate_treatment_resistance(
        self,
        num_medication_trials: int,
        response_to_previous: Optional[bool],
        refractory_indicators: int
    ) -> TreatmentResistance:
        """
        Classifica resistência ao tratamento
        
        Baseado em:
        - Número de tentativas medicamentosas
        - Resposta a tratamentos anteriores
        - Indicadores de refratariedade (comorbidades, cronicidade, etc)
        """
        
        if num_medication_trials == 0:
            return TreatmentResistance.TREATMENT_NAIVE
        elif num_medication_trials == 1:
            if response_to_previous is True:
                return TreatmentResistance.PARTIAL_RESPONDER
            else:
                return TreatmentResistance.NON_RESPONDER
        elif num_medication_trials <= 2:
            if refractory_indicators >= 2:
                return TreatmentResistance.REFRACTORY
            else:
                return TreatmentResistance.NON_RESPONDER
        else:
            return TreatmentResistance.REFRACTORY
    
    def assess_suicide_risk(self, assessment: ClinicalAssessment) -> float:
        """
        Avalia risco de suicídio (0-100)
        Fatores: ideação suicida, severidade, hospitalizações prévias, histórico familiar
        """
        risk = 0.0
        
        if assessment.suicidal_ideation:
            risk += 40
        
        # Aumentar risco com severidade
        severity = self.calculate_severity(
            assessment.phq9_score,
            assessment.hamilton_score,
            assessment.madrs_score
        )
        severity_risk = {
            DepressionSeverity.MINIMAL: 5,
            DepressionSeverity.MILD: 10,
            DepressionSeverity.MODERATE: 20,
            DepressionSeverity.MODERATELY_SEVERE: 30,
            DepressionSeverity.SEVERE: 40
        }
        risk += severity_risk[severity]
        
        if assessment.hospitalization_history:
            risk += 15
        
        if assessment.family_history_depression:
            risk += 10
        
        if assessment.substance_abuse:
            risk += 10
        
        return min(risk, 100.0)
    
    def perform_assessment(self, assessment: ClinicalAssessment) -> AssessmentResult:
        """Realiza avaliação completa do paciente"""
        
        self.logger.info(f"Iniciando avaliação para paciente {assessment.patient_id}")
        
        # Calcular severidade
        severity = self.calculate_severity(
            assessment.phq9_score,
            assessment.hamilton_score,
            assessment.madrs_score
        )
        
        # Calcular resistência ao tratamento
        resistance = self.calculate_treatment_resistance(
            num_medication_trials=len(assessment.medication_trials),
            response_to_previous=None,  # Seria obtido do histórico
            refractory_indicators=sum([
                assessment.anxiety_disorder,
                assessment.personality_disorder,
                assessment.substance_abuse
            ])
        )
        
        # Calcular risco de suicídio
        risk_score = self.assess_suicide_risk(assessment)
        
        # Gerar recomendações
        recommendations = self._generate_recommendations(
            severity, resistance, assessment
        )
        
        result = AssessmentResult(
            patient_id=assessment.patient_id,
            assessment_date=assessment.assessment_date,
            severity=severity,
            treatment_resistance=resistance,
            risk_score=risk_score,
            confidence_score=0.85,
            recommendations=recommendations,
            clinical_notes=assessment.notes
        )
        
        self.logger.info(f"Avaliação concluída: Severidade={severity.value}, "
                        f"Resistência={resistance.value}, Risco={risk_score:.1f}")
        
        return result
    
    def _generate_recommendations(
        self,
        severity: DepressionSeverity,
        resistance: TreatmentResistance,
        assessment: ClinicalAssessment
    ) -> List[str]:
        
        recommendations = []
        
        if severity == DepressionSeverity.SEVERE:
            recommendations.append("Considerar hospitalização ou programa de cuidado intensivo")
            recommendations.append("Avaliação imediata de risco de suicídio")
        
        if resistance == TreatmentResistance.REFRACTORY:
            recommendations.append("Considerar TEC (Terapia Eletroconvulsiva)")
            recommendations.append("Consideração para estimulação magnética transcraniana (TMS)")
            recommendations.append("Referência para especialista em depressão refratária")
        
        if assessment.anxiety_disorder:
            recommendations.append("Tratamento integrado para ansiedade")
        
        if assessment.substance_abuse:
            recommendations.append("Avaliação e tratamento de transtorno por uso de substância")
        
        if severity in [DepressionSeverity.MODERATE, DepressionSeverity.MODERATELY_SEVERE]:
            recommendations.append("Psicoterapia (TCC ou IPT) recomendada")
        
        if not assessment.psychotherapy_history:
            recommendations.append("Iniciar psicoterapia se ainda não realizada")
        
        return recommendations or ["Continuar monitoramento clínico regular"]