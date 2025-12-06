"""
Módulo de Monitoramento de Desfechos (Outcomes)
Rastreia resposta ao tratamento e progresso clínico longitudinal
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import logging
from statistics import mean, stdev

logger = logging.getLogger(__name__)


class ResponseStatus(Enum):
    """Status de resposta ao tratamento"""
    RESPONDER = "responder"  # ≥50% redução de sintomas
    PARTIAL_RESPONDER = "partial_responder"  # 25-49% redução
    NON_RESPONDER = "non_responder"  # <25% redução
    WORSENING = "worsening"  # Piora dos sintomas


class OutcomeCategory(Enum):
    """Categorias de desfecho"""
    SYMPTOM_REDUCTION = "Redução de Sintomas"
    FUNCTIONAL_IMPROVEMENT = "Melhoria Funcional"
    QUALITY_OF_LIFE = "Qualidade de Vida"
    ADVERSE_EFFECTS = "Efeitos Adversos"
    MEDICATION_ADHERENCE = "Adesão ao Medicamento"


@dataclass
class SymptomMeasurement:
    """Medição de sintomas em um ponto no tempo"""
    assessment_date: datetime
    
    # Escalas clínicas
    phq9_score: int  # Depression (0-27)
    gad7_score: int  # Anxiety (0-21)
    madrs_score: int  # Depression rating (0-60)
    hamilton_score: int  # Depression (0-52)
    
    # Sintomas específicos
    sleep_quality: int  # 0-10
    energy_level: int  # 0-10
    appetite: int  # 0-10
    concentration: int  # 0-10
    motivation: int  # 0-10
    
    # Ideação suicida
    suicidal_ideation_severity: int  # 0-10
    
    # Notas clínicas
    clinician_notes: str = ""
    patient_reported_status: str = ""


@dataclass
class FunctionalOutcome:
    """Medida de funcionamento e capacidade"""
    assessment_date: datetime
    
    # Trabalho/Estudo
    work_days_missed: int = 0
    productivity_percentage: float = 100.0
    able_to_work: bool = True
    
    # Vida pessoal
    social_activities: int  # 0-10
    self_care_ability: int  # 0-10
    household_tasks_completion: int  # 0-10
    
    # Relacionamentos
    relationship_quality: int  # 0-10
    social_isolation_score: int  # 0-10
    
    # Atividades de lazer
    engagement_in_activities: int  # 0-10
    
    # Score funcional geral
    overall_functioning_score: float = 0.0  # 0-100


@dataclass
class AdverseEffectRecord:
    """Registro de efeito adverso"""
    reported_date: datetime
    effect_name: str
    severity: int  # 1-10
    related_to_medication: str  # Nome do medicamento
    action_taken: str  # "Dose reduction", "Stopped", "Continued", etc
    resolved: bool = False
    resolution_date: Optional[datetime] = None


@dataclass
class OutcomeAssessment:
    """Avaliação completa de desfecho em um ponto no tempo"""
    patient_id: str
    assessment_date: datetime
    assessment_week: int  # Semana desde início do tratamento
    
    # Sintomas
    symptom_measurement: SymptomMeasurement
    
    # Funcionamento
    functional_outcome: FunctionalOutcome
    
    # Efeitos adversos
    adverse_effects: List[AdverseEffectRecord] = field(default_factory=list)
    
    # Adesão ao medicamento
    medication_adherence_percentage: float = 100.0
    missed_doses: int = 0
    
    # Medicações atuais
    current_medications: List[str] = field(default_factory=list)
    
    # Avaliação clínica geral
    clinician_global_impression: int = 4  # 1=Muito melhor, 7=Muito pior
    
    # Notas e observações
    clinical_notes: str = ""


@dataclass
class LongitudinalOutcomeData:
    """Dados longitudinais de um paciente ao longo do tempo"""
    patient_id: str
    treatment_start_date: datetime
    
    assessments: List[OutcomeAssessment] = field(default_factory=list)
    
    # Agregação de dados
    created_date: datetime = field(default_factory=datetime.now)
    last_assessment_date: Optional[datetime] = None


class OutcomeAnalyzer:
    """Analisa desfechos de tratamento"""
    
    def __init__(self):
        self.logger = logger
    
    def calculate_response_status(
        self,
        baseline_phq9: int,
        current_phq9: int
    ) -> ResponseStatus:
        """Calcula status de resposta baseado em PHQ-9"""
        
        reduction_percentage = ((baseline_phq9 - current_phq9) / baseline_phq9 * 100) if baseline_phq9 > 0 else 0
        
        if reduction_percentage >= 50:
            return ResponseStatus.RESPONDER
        elif reduction_percentage >= 25:
            return ResponseStatus.PARTIAL_RESPONDER
        elif current_phq9 > baseline_phq9:
            return ResponseStatus.WORSENING
        else:
            return ResponseStatus.NON_RESPONDER
    
    def analyze_trajectory(
        self,
        longitudinal_data: LongitudinalOutcomeData
    ) -> Dict:
        """Analisa trajetória de resposta ao longo do tempo"""
        
        if len(longitudinal_data.assessments) < 2:
            return {"error": "Insufficient data for trajectory analysis"}
        
        assessments = sorted(
            longitudinal_data.assessments,
            key=lambda x: x.assessment_date
        )
        
        # Extrair PHQ-9 ao longo do tempo
        phq9_scores = [a.symptom_measurement.phq9_score for a in assessments]
        dates = [a.assessment_date for a in assessments]
        
        # Calcular mudança total
        total_change = phq9_scores[-1] - phq9_scores[0]
        percent_change = (total_change / phq9_scores[0] * 100) if phq9_scores[0] > 0 else 0
        
        # Calcular tendência (slope)
        if len(phq9_scores) > 1:
            days_elapsed = (dates[-1] - dates[0]).days
            if days_elapsed > 0:
                slope = total_change / days_elapsed  # Mudança por dia
            else:
                slope = 0.0
        else:
            slope = 0.0
        
        # Calcular variabilidade
        if len(phq9_scores) > 1:
            variance = stdev(phq9_scores)
        else:
            variance = 0.0
        
        # Determinar status final
        baseline = phq9_scores[0]
        final = phq9_scores[-1]
        response_status = self.calculate_response_status(baseline, final)
        
        # Calcular velocidade de resposta (dias para 25% melhora)
        days_to_25_percent = self._calculate_days_to_improvement(phq9_scores, dates, 25)
        
        analysis = {
            "baseline_phq9": baseline,
            "final_phq9": final,
            "total_change": total_change,
            "percent_change": percent_change,
            "response_status": response_status.value,
            "slope_per_day": slope,
            "variability_stdev": variance,
            "assessment_points": len(phq9_scores),
            "days_of_followup": (dates[-1] - dates[0]).days,
            "days_to_25_percent_improvement": days_to_25_percent
        }
        
        return analysis
    
    def analyze_functional_improvement(
        self,
        longitudinal_data: LongitudinalOutcomeData
    ) -> Dict:
        """Analisa melhoria funcional ao longo do tempo"""
        
        if not longitudinal_data.assessments:
            return {"error": "No assessments available"}
        
        first_assessment = longitudinal_data.assessments[0]
        last_assessment = longitudinal_data.assessments[-1]
        
        first_functional = first_assessment.functional_outcome
        last_functional = last_assessment.functional_outcome
        
        return {
            "work_days_missed_baseline": first_functional.work_days_missed,
            "work_days_missed_current": last_functional.work_days_missed,
            "work_days_saved": first_functional.work_days_missed - last_functional.work_days_missed,
            "productivity_improvement": last_functional.productivity_percentage - first_functional.productivity_percentage,
            "social_activity_change": last_functional.social_activities - first_functional.social_activities,
            "relationship_quality_change": last_functional.relationship_quality - first_functional.relationship_quality,
            "self_care_improvement": last_functional.self_care_ability - first_functional.self_care_ability,
            "overall_functioning_improvement": last_functional.overall_functioning_score - first_functional.overall_functioning_score
        }
    
    def assess_adverse_effects_profile(
        self,
        longitudinal_data: LongitudinalOutcomeData
    ) -> Dict:
        """Avalia perfil de efeitos adversos"""
        
        all_adverse_effects = []
        for assessment in longitudinal_data.assessments:
            all_adverse_effects.extend(assessment.adverse_effects)
        
        if not all_adverse_effects:
            return {
                "total_adverse_effects_reported": 0,
                "severe_adverse_effects": 0,
                "resolved_effects": 0,
                "ongoing_effects": 0
            }
        
        severe_effects = [ae for ae in all_adverse_effects if ae.severity >= 7]
        resolved_effects = [ae for ae in all_adverse_effects if ae.resolved]
        ongoing_effects = [ae for ae in all_adverse_effects if not ae.resolved]
        
        # Agrupar por tipo
        effect_summary = {}
        for ae in all_adverse_effects:
            if ae.effect_name not in effect_summary:
                effect_summary[ae.effect_name] = {
                    "count": 0,
                    "avg_severity": 0,
                    "resolved": 0
                }
            effect_summary[ae.effect_name]["count"] += 1
            effect_summary[ae.effect_name]["resolved"] += 1 if ae.resolved else 0
        
        # Calcular severidade média por tipo
        for effect_name in effect_summary:
            effects = [ae for ae in all_adverse_effects if ae.effect_name == effect_name]
            effect_summary[effect_name]["avg_severity"] = mean([e.severity for e in effects])
        
        return {
            "total_adverse_effects_reported": len(all_adverse_effects),
            "severe_adverse_effects": len(severe_effects),
            "resolved_effects": len(resolved_effects),
            "ongoing_effects": len(ongoing_effects),
            "effect_breakdown": effect_summary
        }
    
    def assess_adherence_trajectory(
        self,
        longitudinal_data: LongitudinalOutcomeData
    ) -> Dict:
        """Avalia trajetória de adesão ao medicamento"""
        
        if not longitudinal_data.assessments:
            return {"error": "No assessments available"}
        
        adherence_scores = [a.medication_adherence_percentage for a in longitudinal_data.assessments]
        missed_doses = [a.missed_doses for a in longitudinal_data.assessments]
        
        return {
            "baseline_adherence": adherence_scores[0],
            "current_adherence": adherence_scores[-1],
            "avg_adherence": mean(adherence_scores),
            "min_adherence": min(adherence_scores),
            "adherence_declining": adherence_scores[-1] < adherence_scores[0],
            "total_missed_doses": sum(missed_doses),
            "avg_missed_doses_per_week": mean(missed_doses) if missed_doses else 0
        }
    
    def generate_outcome_report(
        self,
        longitudinal_data: LongitudinalOutcomeData
    ) -> Dict:
        """Gera relatório completo de desfechos"""
        
        self.logger.info(f"Gerando relatório de desfechos para {longitudinal_data.patient_id}")
        
        report = {
            "patient_id": longitudinal_data.patient_id,
            "report_date": datetime.now().isoformat(),
            "treatment_duration_days": (
                (longitudinal_data.assessments[-1].assessment_date - longitudinal_data.treatment_start_date).days
                if longitudinal_data.assessments else 0
            ),
            "symptom_trajectory": self.analyze_trajectory(longitudinal_data),
            "functional_improvement": self.analyze_functional_improvement(longitudinal_data),
            "adverse_effects": self.assess_adverse_effects_profile(longitudinal_data),
            "adherence_profile": self.assess_adherence_trajectory(longitudinal_data)
        }
        
        # Síntese final
        if "response_status" in report["symptom_trajectory"]:
            response = report["symptom_trajectory"]["response_status"]
            report["overall_assessment"] = {
                "treatment_response": response,
                "recommendation": self._get_recommendation(response, report)
            }
        
        return report
    
    def _calculate_days_to_improvement(
        self,
        scores: List[int],
        dates: List[datetime],
        improvement_percent: int
    ) -> Optional[int]:
        """Calcula dias até melhoria de X%"""
        
        baseline = scores[0]
        target = baseline - (baseline * improvement_percent / 100)
        
        for i, score in enumerate(scores):
            if score <= target:
                return (dates[i] - dates[0]).days
        
        return None
    
    def _get_recommendation(self, response_status: str, report: Dict) -> str:
        """Gera recomendação baseada no status de resposta"""
        
        if response_status == ResponseStatus.RESPONDER.value:
            return "Continuar tratamento atual e considerar manutenção a longo prazo"
        elif response_status == ResponseStatus.PARTIAL_RESPONDER.value:
            return "Considerar aumento de dose ou adição de medicamento complementar"
        elif response_status == ResponseStatus.NON_RESPONDER.value:
            return "Considerar mudança de medicamento ou terapia adicional"
        elif response_status == ResponseStatus.WORSENING.value:
            return "Avaliação urgente - considerar hospitalização ou mudança imediata de tratamento"
        
        return "Continuar monitoramento"