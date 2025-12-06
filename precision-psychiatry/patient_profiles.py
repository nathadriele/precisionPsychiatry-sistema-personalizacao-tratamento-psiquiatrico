"""
Módulo de Gestão de Perfis de Pacientes
Armazena e gerencia dados abrangentes de pacientes para análise
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Set
from datetime import datetime
import logging
from uuid import uuid4

logger = logging.getLogger(__name__)


class BiologicalSex(Enum):
    """Sexo biológico"""
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"


class EducationLevel(Enum):
    """Nível de educação"""
    PRIMARY = "primary"
    SECONDARY = "secondary"
    TERTIARY = "tertiary"
    POSTGRADUATE = "postgraduate"


class EmploymentStatus(Enum):
    """Status de emprego"""
    EMPLOYED = "employed"
    UNEMPLOYED = "unemployed"
    STUDENT = "student"
    RETIRED = "retired"
    DISABLED = "disabled"


@dataclass
class GeneticProfile:
    """Perfil genético do paciente"""
    patient_id: str
    
    # Polimorfismos relevantes para psiquiatria
    serotonin_transporter_genotype: str  # 5-HTTLPR (L/L, L/S, S/S)
    bdnf_genotype: str  # Val66Met
    comt_genotype: str  # Val158Met
    cyp2d6_phenotype: str  # Ultra-rapid, Rapid, Normal, Intermediate, Poor
    cyp3a4_activity: str  # Low, Normal, High
    mthfr_genotype: str  # C677T
    fkbp5_genotype: str  # F/F, F/T, T/T (Gene x Environment)
    
    # Dados gerais
    sequencing_date: datetime = field(default_factory=datetime.now)
    laboratory: str = ""
    notes: str = ""


@dataclass
class NeurobiologicalMarkers:
    """Marcadores neurobiológicos"""
    patient_id: str
    assessment_date: datetime = field(default_factory=datetime.now)
    
    # Biomarcadores sanguíneos
    cortisol_morning_mcg_dl: Optional[float] = None
    cortisol_evening_mcg_dl: Optional[float] = None
    inflammatory_markers: Dict[str, float] = field(default_factory=dict)  # IL-6, TNF-a, etc
    
    # EEG
    eeg_theta_power: Optional[float] = None  # Frontal asymmetry
    eeg_beta_power: Optional[float] = None
    eeg_alpha_power: Optional[float] = None
    
    # Neuroimagem
    brain_volume_mm3: Optional[float] = None
    hippocampal_volume_mm3: Optional[float] = None
    prefrontal_cortex_activation: Optional[float] = None
    amygdala_reactivity: Optional[float] = None
    
    # Outros marcadores
    sleep_quality_score: Optional[int] = None  # 0-100
    cognitive_performance_score: Optional[float] = None
    
    notes: str = ""


@dataclass
class PsychosocialFactors:
    """Fatores psicossociais e ambientais"""
    patient_id: str
    assessment_date: datetime = field(default_factory=datetime.now)
    
    # Vida pessoal
    marital_status: str  # Single, Married, Divorced, Widowed
    relationship_quality_score: Optional[int] = None  # 0-10
    children: int = 0
    living_situation: str  # Alone, With family, Institutional
    
    # Suporte social
    close_relationships: int = 0  # Número de relacionamentos próximos
    social_network_quality: int = 0  # 0-10
    perceived_support_score: int = 0  # 0-10
    social_isolation: bool = False
    
    # Estressores recentes
    major_life_stressors: List[str] = field(default_factory=list)
    recent_trauma: bool = False
    childhood_trauma: bool = False
    ace_score: int = 0  # Adverse Childhood Experiences (0-10)
    
    # Ocupação
    employment_status: EmploymentStatus = EmploymentStatus.EMPLOYED
    job_satisfaction_score: Optional[int] = None  # 0-10
    work_stress_level: str = "moderate"  # Low, Moderate, High
    financial_stress: bool = False
    
    # Qualidade de vida
    quality_of_life_score: Optional[float] = None  # 0-100
    activities_engagement: int = 0  # 0-10
    purpose_meaning_score: Optional[int] = None  # 0-10
    
    notes: str = ""


@dataclass
class TreatmentHistory:
    """Histórico de tratamento do paciente"""
    patient_id: str
    
    # Medicações
    current_medications: List[str] = field(default_factory=list)
    medication_history: Dict[str, Dict] = field(default_factory=dict)  # med_name -> {dose, dates, response}
    
    # Terapias
    psychotherapy_history: List[Dict] = field(default_factory=list)
    hospitalization_episodes: int = 0
    ect_treatments: int = 0
    
    # Respostas e efeitos adversos
    medication_allergies: List[str] = field(default_factory=list)
    adverse_reactions: List[str] = field(default_factory=list)
    
    # Adesão
    adherence_score: float = 1.0  # 0-1
    adherence_notes: str = ""


@dataclass
class DemographicInfo:
    """Informações demográficas"""
    patient_id: str
    first_name: str
    last_name: str
    date_of_birth: datetime
    biological_sex: BiologicalSex
    
    ethnicity: str = ""
    education_level: EducationLevel = EducationLevel.SECONDARY
    primary_language: str = "Portuguese"
    
    contact_email: str = ""
    contact_phone: str = ""
    
    # Informações clínicas básicas
    enrollment_date: datetime = field(default_factory=datetime.now)
    primary_diagnosis: str = "Major Depressive Disorder"


@dataclass
class PatientProfile:
    """Perfil completo do paciente - agregação de todos os dados"""
    
    patient_id: str
    
    # Dados demográficos
    demographics: DemographicInfo
    
    # Dados genéticos
    genetic_profile: Optional[GeneticProfile] = None
    
    # Marcadores neurobiológicos
    neurobiological_markers: Optional[NeurobiologicalMarkers] = None
    
    # Fatores psicossociais
    psychosocial_factors: Optional[PsychosocialFactors] = None
    
    # Histórico de tratamento
    treatment_history: TreatmentHistory = field(default_factory=TreatmentHistory)
    
    # Metadata
    created_date: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    last_assessment_date: Optional[datetime] = None
    
    # Status e notas
    is_active: bool = True
    clinical_notes: str = ""
    research_eligible: bool = False
    
    def get_age(self) -> int:
        """Calcula idade do paciente"""
        today = datetime.now()
        return today.year - self.demographics.date_of_birth.year - (
            (today.month, today.day) < (
                self.demographics.date_of_birth.month,
                self.demographics.date_of_birth.day
            )
        )
    
    def has_complete_data(self) -> bool:
        """Verifica se perfil tem dados suficientes para análise"""
        return (
            self.genetic_profile is not None and
            self.neurobiological_markers is not None and
            self.psychosocial_factors is not None
        )
    
    def get_data_completeness_percentage(self) -> float:
        """Retorna percentual de completude dos dados"""
        fields = 0
        filled = 0
        
        # Verificar cada seção
        if self.genetic_profile:
            fields += 8
            filled += sum(1 for v in [
                self.genetic_profile.serotonin_transporter_genotype,
                self.genetic_profile.bdnf_genotype,
                self.genetic_profile.comt_genotype,
                self.genetic_profile.cyp2d6_phenotype,
                self.genetic_profile.cyp3a4_activity,
                self.genetic_profile.mthfr_genotype,
                self.genetic_profile.fkbp5_genotype
            ] if v)
        
        if self.neurobiological_markers:
            fields += 8
            filled += sum(1 for v in [
                self.neurobiological_markers.cortisol_morning_mcg_dl,
                self.neurobiological_markers.eeg_theta_power,
                self.neurobiological_markers.brain_volume_mm3,
                self.neurobiological_markers.hippocampal_volume_mm3,
                self.neurobiological_markers.prefrontal_cortex_activation,
                self.neurobiological_markers.amygdala_reactivity,
                self.neurobiological_markers.sleep_quality_score,
                self.neurobiological_markers.cognitive_performance_score
            ] if v is not None)
        
        if self.psychosocial_factors:
            fields += 6
            filled += sum(1 for v in [
                self.psychosocial_factors.relationship_quality_score,
                self.psychosocial_factors.social_network_quality,
                self.psychosocial_factors.perceived_support_score,
                self.psychosocial_factors.job_satisfaction_score,
                self.psychosocial_factors.quality_of_life_score,
                self.psychosocial_factors.purpose_meaning_score
            ] if v is not None)
        
        if fields == 0:
            return 0.0
        
        return (filled / fields) * 100


class PatientProfileRepository:
    """Gerencia perfis de pacientes"""
    
    def __init__(self):
        self.logger = logger
        self.profiles: Dict[str, PatientProfile] = {}
    
    def create_patient(
        self,
        first_name: str,
        last_name: str,
        date_of_birth: datetime,
        biological_sex: BiologicalSex
    ) -> PatientProfile:
        """Cria novo perfil de paciente"""
        
        patient_id = str(uuid4())
        
        demographics = DemographicInfo(
            patient_id=patient_id,
            first_name=first_name,
            last_name=last_name,
            date_of_birth=date_of_birth,
            biological_sex=biological_sex
        )
        
        profile = PatientProfile(
            patient_id=patient_id,
            demographics=demographics,
            treatment_history=TreatmentHistory(patient_id=patient_id)
        )
        
        self.profiles[patient_id] = profile
        self.logger.info(f"Paciente criado: {patient_id}")
        
        return profile
    
    def get_patient(self, patient_id: str) -> Optional[PatientProfile]:
        """Recupera perfil de paciente"""
        return self.profiles.get(patient_id)
    
    def update_patient(self, profile: PatientProfile) -> None:
        """Atualiza perfil de paciente"""
        profile.last_updated = datetime.now()
        self.profiles[profile.patient_id] = profile
        self.logger.info(f"Paciente atualizado: {profile.patient_id}")
    
    def get_patients_by_criteria(
        self,
        min_age: Optional[int] = None,
        max_age: Optional[int] = None,
        gender: Optional[BiologicalSex] = None,
        has_genetic_data: bool = False
    ) -> List[PatientProfile]:
        """Recupera pacientes por critérios"""
        
        results = []
        for profile in self.profiles.values():
            age = profile.get_age()
            
            if min_age and age < min_age:
                continue
            if max_age and age > max_age:
                continue
            if gender and profile.demographics.biological_sex != gender:
                continue
            if has_genetic_data and profile.genetic_profile is None:
                continue
            
            results.append(profile)
        
        return results
    
    def get_statistics(self) -> Dict:
        
        total = len(self.profiles)
        with_genetic = sum(1 for p in self.profiles.values() if p.genetic_profile)
        with_neurobiological = sum(1 for p in self.profiles.values() if p.neurobiological_markers)
        with_psychosocial = sum(1 for p in self.profiles.values() if p.psychosocial_factors)
        complete = sum(1 for p in self.profiles.values() if p.has_complete_data())
        
        return {
            "total_patients": total,
            "with_genetic_data": with_genetic,
            "with_neurobiological_data": with_neurobiological,
            "with_psychosocial_data": with_psychosocial,
            "complete_profiles": complete,
            "avg_completeness_percent": (
                sum(p.get_data_completeness_percentage() for p in self.profiles.values()) / total
                if total > 0 else 0.0
            )
        }