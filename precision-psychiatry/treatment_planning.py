"""
Módulo de Planejamento de Tratamento Personalizado
Gera recomendações terapêuticas baseadas em dados genômicos, neurobiológicos e psicossociais
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class MedicationClass(Enum):
    """Classes de medicamentos antidepressivos"""
    SSRI = "SSRI"  # Selective Serotonin Reuptake Inhibitor
    SNRI = "SNRI"  # Serotonin-Noradrenaline Reuptake Inhibitor
    TCA = "TCA"    # Tricyclic Antidepressant
    MAOI = "MAOI"  # Monoamine Oxidase Inhibitor
    ATYPICAL = "ATYPICAL"
    AUGMENTATION = "AUGMENTATION"


class TherapyType(Enum):
    """Tipos de terapia psicológica"""
    CBT = "Cognitive Behavioral Therapy"
    IPT = "Interpersonal Therapy"
    PSYCHODYNAMIC = "Psychodynamic Therapy"
    MINDFULNESS = "Mindfulness-Based Therapy"
    FAMILY = "Family Therapy"
    GROUP = "Group Therapy"


@dataclass
class Medication:
    """Informações sobre um medicamento"""
    name: str
    drug_class: MedicationClass
    typical_dose_mg: Tuple[int, int]
    efficacy_score: float  # 0-1
    side_effect_risk: float  # 0-1
    contraindications: List[str] = field(default_factory=list)
    genetic_interactions: Dict[str, float] = field(default_factory=dict)
    notes: str = ""


@dataclass
class TherapyRecommendation:
    """Recomendação de terapia psicológica"""
    therapy_type: TherapyType
    frequency_per_week: int
    duration_weeks: int
    priority: int  # 1 (alta) a 3 (baixa)
    rationale: str


@dataclass
class MedicationRecommendation:
    """Recomendação de medicamento"""
    medication: Medication
    recommended_dose_mg: int
    priority: int  # 1 (primeira linha) a 3 (terceira linha)
    expected_response_time_days: int
    confidence_score: float  # 0-1
    risk_mitigation: List[str]
    rationale: str


@dataclass
class TreatmentPlan:
    """Plano de tratamento personalizado"""
    patient_id: str
    created_date: datetime
    start_date: datetime
    
    medications: List[MedicationRecommendation]
    therapies: List[TherapyRecommendation]
    
    lifestyle_interventions: List[str] = field(default_factory=list)
    monitoring_plan: List[str] = field(default_factory=list)
    follow_up_schedule: Dict[str, datetime] = field(default_factory=dict)
    
    estimated_response_probability: float = 0.0
    overall_confidence: float = 0.0
    
    notes: str = ""


class TreatmentPlanner:
    """Classe para gerar planos de tratamento personalizados"""
    
    # Banco de dados de medicamentos
    MEDICATION_DATABASE = {
        "sertraline": Medication(
            name="Sertraline",
            drug_class=MedicationClass.SSRI,
            typical_dose_mg=(50, 200),
            efficacy_score=0.72,
            side_effect_risk=0.35,
            genetic_interactions={"CYP2D6": 0.4, "CYP3A4": 0.3},
            notes="Primeira linha, bem tolerado"
        ),
        "escitalopram": Medication(
            name="Escitalopram",
            drug_class=MedicationClass.SSRI,
            typical_dose_mg=(10, 20),
            efficacy_score=0.75,
            side_effect_risk=0.32,
            genetic_interactions={"CYP2D6": 0.5, "CYP3A4": 0.2},
            notes="Isômero ativo, alta eficácia"
        ),
        "venlafaxine": Medication(
            name="Venlafaxina",
            drug_class=MedicationClass.SNRI,
            typical_dose_mg=(75, 375),
            efficacy_score=0.78,
            side_effect_risk=0.42,
            genetic_interactions={"CYP2D6": 0.8, "CYP3A4": 0.4},
            notes="Bom para depressão moderada a grave"
        ),
        "bupropion": Medication(
            name="Bupropion",
            drug_class=MedicationClass.ATYPICAL,
            typical_dose_mg=(150, 450),
            efficacy_score=0.70,
            side_effect_risk=0.30,
            genetic_interactions={"CYP2D6": 0.9, "CYP3A4": 0.5},
            notes="Bom para apatia e fadiga"
        ),
        "aripiprazole": Medication(
            name="Aripiprazol",
            drug_class=MedicationClass.AUGMENTATION,
            typical_dose_mg=(2, 10),
            efficacy_score=0.65,
            side_effect_risk=0.40,
            genetic_interactions={"CYP3A4": 0.6, "CYP2D6": 0.5},
            notes="Aumento potencial para SSRIs"
        )
    }
    
    def __init__(self):
        self.logger = logger
    
    def plan_treatment(
        self,
        patient_id: str,
        genetic_profile: Dict[str, any],
        clinical_assessment: Dict[str, any],
        neurobiological_markers: Dict[str, float],
        psychosocial_factors: Dict[str, any],
        previous_medications: List[str] = None
    ) -> TreatmentPlan:
        """
        Gera plano de tratamento personalizado baseado em múltiplos fatores
        """
        
        self.logger.info(f"Gerando plano de tratamento para paciente {patient_id}")
        
        # Selecionar medicamentos
        medication_recommendations = self._select_medications(
            genetic_profile=genetic_profile,
            clinical_assessment=clinical_assessment,
            neurobiological_markers=neurobiological_markers,
            previous_medications=previous_medications or []
        )
        
        # Selecionar terapias psicológicas
        therapy_recommendations = self._select_therapies(
            clinical_assessment=clinical_assessment,
            psychosocial_factors=psychosocial_factors
        )
        
        # Gerar intervenções de lifestyle
        lifestyle = self._generate_lifestyle_interventions(
            clinical_assessment=clinical_assessment,
            psychosocial_factors=psychosocial_factors
        )
        
        # Planejar monitoramento
        monitoring = self._plan_monitoring(
            medication_recommendations=medication_recommendations,
            therapy_recommendations=therapy_recommendations
        )
        
        # Agendar follow-ups
        follow_up_schedule = self._schedule_follow_ups()
        
        # Calcular probabilidade de resposta
        response_probability = self._estimate_response_probability(
            medication_recommendations=medication_recommendations,
            therapy_recommendations=therapy_recommendations,
            clinical_assessment=clinical_assessment
        )
        
        plan = TreatmentPlan(
            patient_id=patient_id,
            created_date=datetime.now(),
            start_date=datetime.now() + timedelta(days=1),
            medications=medication_recommendations,
            therapies=therapy_recommendations,
            lifestyle_interventions=lifestyle,
            monitoring_plan=monitoring,
            follow_up_schedule=follow_up_schedule,
            estimated_response_probability=response_probability,
            overall_confidence=0.82
        )
        
        self.logger.info(f"Plano criado: {len(medication_recommendations)} medicamentos, "
                        f"{len(therapy_recommendations)} terapias, "
                        f"Prob. resposta: {response_probability:.1%}")
        
        return plan
    
    def _select_medications(
        self,
        genetic_profile: Dict,
        clinical_assessment: Dict,
        neurobiological_markers: Dict,
        previous_medications: List[str]
    ) -> List[MedicationRecommendation]:
        """Seleciona medicamentos baseado em perfil genético e clínico"""
        
        recommendations = []
        severity = clinical_assessment.get("severity", "moderate")
        
        # Prioridade 1: SSRIs para primeira linha (se tolerados antes)
        if "sertraline" not in previous_medications:
            med = self.MEDICATION_DATABASE["sertraline"]
            dose = self._calculate_optimal_dose(med, genetic_profile, severity)
            recommendations.append(MedicationRecommendation(
                medication=med,
                recommended_dose_mg=dose,
                priority=1,
                expected_response_time_days=14,
                confidence_score=0.73,
                risk_mitigation=["Monitorar efeitos adversos iniciais"],
                rationale="SSRI de primeira linha, bem tolerado, perfil genético favorável"
            ))
        
        # Prioridade 2: Alternativa SSRI ou SNRI
        if severity in ["moderate", "moderately_severe"]:
            med = self.MEDICATION_DATABASE["venlafaxine"]
            dose = self._calculate_optimal_dose(med, genetic_profile, severity)
            recommendations.append(MedicationRecommendation(
                medication=med,
                recommended_dose_mg=dose,
                priority=2,
                expected_response_time_days=14,
                confidence_score=0.71,
                risk_mitigation=["Monitorar pressão arterial", "Aumentar dose gradualmente"],
                rationale="SNRI com eficácia moderadamente superior para depressão mais grave"
            ))
        
        # Prioridade 3: Augmentação se refratário
        if len(previous_medications) >= 2:
            med = self.MEDICATION_DATABASE["aripiprazole"]
            recommendations.append(MedicationRecommendation(
                medication=med,
                recommended_dose_mg=5,
                priority=3,
                expected_response_time_days=21,
                confidence_score=0.65,
                risk_mitigation=["Monitorar ganho de peso", "Monitorar metabolismo glicêmico"],
                rationale="Potencial aumento para pacientes refratários"
            ))
        
        return recommendations
    
    def _select_therapies(
        self,
        clinical_assessment: Dict,
        psychosocial_factors: Dict
    ) -> List[TherapyRecommendation]:
        """Seleciona terapias psicológicas"""
        
        recommendations = []
        severity = clinical_assessment.get("severity", "moderate")
        
        # TCC é primeira linha para maioria dos casos
        recommendations.append(TherapyRecommendation(
            therapy_type=TherapyType.CBT,
            frequency_per_week=1,
            duration_weeks=12,
            priority=1,
            rationale="Primeira linha, eficácia comprovada para depressão"
        ))
        
        # IPT se problemas interpessoais significativos
        if psychosocial_factors.get("relationship_problems", False):
            recommendations.append(TherapyRecommendation(
                therapy_type=TherapyType.IPT,
                frequency_per_week=1,
                duration_weeks=16,
                priority=2,
                rationale="Problemas interpessoais detectados"
            ))
        
        # Mindfulness para depressão recorrente
        if clinical_assessment.get("num_episodes", 0) >= 2:
            recommendations.append(TherapyRecommendation(
                therapy_type=TherapyType.MINDFULNESS,
                frequency_per_week=1,
                duration_weeks=8,
                priority=2,
                rationale="Prevenção de recaída em depressão recorrente"
            ))
        
        return recommendations
    
    def _generate_lifestyle_interventions(
        self,
        clinical_assessment: Dict,
        psychosocial_factors: Dict
    ) -> List[str]:
        """Gera recomendações de estilo de vida"""
        
        interventions = [
            "Atividade física: 150 min/semana de exercício moderado",
            "Higiene do sono: 7-9 horas/noite, horário regular",
            "Nutrição: Dieta balanceada, evitar álcool em excesso"
        ]
        
        if psychosocial_factors.get("social_isolation", False):
            interventions.append("Aumento de atividades sociais e apoio comunitário")
        
        if clinical_assessment.get("anxiety_disorder", False):
            interventions.append("Técnicas de respiração e relaxamento")
        
        return interventions
    
    def _plan_monitoring(
        self,
        medication_recommendations: List[MedicationRecommendation],
        therapy_recommendations: List[TherapyRecommendation]
    ) -> List[str]:
        """Planeja estratégia de monitoramento"""
        
        monitoring = [
            "Escala PHQ-9 semanal nas primeiras 4 semanas",
            "Avaliação de efeitos adversos em 1-2 semanas",
            "Hemograma e função renal antes do tratamento",
        ]
        
        # Monitoramento específico por medicamento
        if any(m.medication.name == "Venlafaxina" for m in medication_recommendations):
            monitoring.append("Monitorar pressão arterial regularmente")
        
        if any(m.medication.drug_class == MedicationClass.AUGMENTATION 
               for m in medication_recommendations):
            monitoring.append("Monitorar glicemia e peso mensalmente")
        
        return monitoring
    
    def _schedule_follow_ups(self) -> Dict[str, datetime]:
        """Agenda consultas de acompanhamento"""
        
        now = datetime.now()
        return {
            "Initial Assessment": now,
            "Week 1-2 Medication": now + timedelta(days=10),
            "Week 4 Response": now + timedelta(days=28),
            "Week 8 Efficacy": now + timedelta(days=56),
            "Week 12 Outcome": now + timedelta(days=84)
        }
    
    def _estimate_response_probability(
        self,
        medication_recommendations: List[MedicationRecommendation],
        therapy_recommendations: List[TherapyRecommendation],
        clinical_assessment: Dict
    ) -> float:
        """Estima probabilidade de resposta ao tratamento"""
        
        # Score base
        score = 0.50
        
        # Adicionar confiança de medicamento (média ponderada)
        if medication_recommendations:
            med_confidence = sum(m.confidence_score for m in medication_recommendations) / len(medication_recommendations)
            score += med_confidence * 0.20
        
        # Adicionar benefício de terapia
        score += len(therapy_recommendations) * 0.08
        
        # Fatores de redução
        if clinical_assessment.get("treatment_resistant", False):
            score *= 0.75
        
        if clinical_assessment.get("comorbidities_count", 0) > 2:
            score *= 0.85
        
        return min(max(score, 0.0), 1.0)
    
    def _calculate_optimal_dose(
        self,
        medication: Medication,
        genetic_profile: Dict,
        severity: str
    ) -> int:
        """Calcula dose ótima baseada em genética e severidade"""
        
        min_dose, max_dose = medication.typical_dose_mg
        
        # Começar com dose conservadora
        base_dose = min_dose
        
        # Aumentar com severidade
        severity_multiplier = {
            "mild": 1.0,
            "moderate": 1.2,
            "moderately_severe": 1.4,
            "severe": 1.6
        }
        
        dose = base_dose * severity_multiplier.get(severity, 1.0)
        
        # Ajustar por metabolizador CYP2D6
        cyp2d6_status = genetic_profile.get("CYP2D6_phenotype", "normal")
        if cyp2d6_status == "poor":
            dose *= 0.75
        elif cyp2d6_status == "ultra-rapid":
            dose *= 1.25
        
        return int(min(dose, max_dose))