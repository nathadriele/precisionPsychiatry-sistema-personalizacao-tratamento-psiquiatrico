"""
Modelos SQLAlchemy ORM para Precision Psychiatry
Define schema do banco de dados
"""

from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, Text, JSON, ForeignKey, Enum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
from uuid import uuid4
import enum

Base = declarative_base()


class PatientModel(Base):
    """Modelo de Paciente"""
    __tablename__ = "patients"
    
    patient_id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    date_of_birth = Column(DateTime, nullable=False)
    biological_sex = Column(String(20), nullable=False)  # male, female, other
    
    # Contato
    email = Column(String(100), unique=True, nullable=True)
    phone = Column(String(20), nullable=True)
    
    # Demographics
    ethnicity = Column(String(100), nullable=True)
    education_level = Column(String(50), nullable=True)
    primary_language = Column(String(50), default="Portuguese")
    
    # Status
    is_active = Column(Boolean, default=True)
    enrollment_date = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relacionamentos
    genetic_profiles = relationship("GeneticProfileModel", back_populates="patient")
    neurobiological_markers = relationship("NeurobiologicalMarkerModel", back_populates="patient")
    psychosocial_factors = relationship("PsychosocialFactorModel", back_populates="patient")
    clinical_assessments = relationship("ClinicalAssessmentModel", back_populates="patient")
    treatment_plans = relationship("TreatmentPlanModel", back_populates="patient")
    outcome_assessments = relationship("OutcomeAssessmentModel", back_populates="patient")
    predictions = relationship("PredictionModel", back_populates="patient")
    
    def __repr__(self):
        return f"<Patient {self.patient_id}: {self.first_name} {self.last_name}>"


class GeneticProfileModel(Base):
    """Modelo de Perfil Genético"""
    __tablename__ = "genetic_profiles"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    patient_id = Column(String, ForeignKey("patients.patient_id"), nullable=False)
    
    # Polimorfismos
    serotonin_transporter_genotype = Column(String(50), nullable=True)  # 5-HTTLPR
    bdnf_genotype = Column(String(50), nullable=True)  # Val66Met
    comt_genotype = Column(String(50), nullable=True)  # Val158Met
    cyp2d6_phenotype = Column(String(50), nullable=True)  # Metabolizer status
    cyp3a4_activity = Column(String(50), nullable=True)
    mthfr_genotype = Column(String(50), nullable=True)  # C677T
    fkbp5_genotype = Column(String(50), nullable=True)  # Gene x Environment
    
    # Metadados
    sequencing_date = Column(DateTime, default=datetime.utcnow)
    laboratory = Column(String(200), nullable=True)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relacionamento
    patient = relationship("PatientModel", back_populates="genetic_profiles")
    
    def __repr__(self):
        return f"<GeneticProfile {self.id}>"


class NeurobiologicalMarkerModel(Base):
    """Modelo de Marcadores Neurobiológicos"""
    __tablename__ = "neurobiological_markers"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    patient_id = Column(String, ForeignKey("patients.patient_id"), nullable=False)
    
    assessment_date = Column(DateTime, default=datetime.utcnow)
    
    # Biomarcadores
    cortisol_morning_mcg_dl = Column(Float, nullable=True)
    cortisol_evening_mcg_dl = Column(Float, nullable=True)
    inflammatory_markers = Column(JSON, nullable=True)  # IL-6, TNF-a, CRP
    
    # EEG
    eeg_theta_power = Column(Float, nullable=True)
    eeg_beta_power = Column(Float, nullable=True)
    eeg_alpha_power = Column(Float, nullable=True)
    frontal_asymmetry_index = Column(Float, nullable=True)
    
    # Neuroimagem
    brain_volume_mm3 = Column(Float, nullable=True)
    hippocampal_volume_mm3 = Column(Float, nullable=True)
    prefrontal_cortex_activation = Column(Float, nullable=True)
    amygdala_reactivity = Column(Float, nullable=True)
    
    # Outros marcadores
    sleep_quality_score = Column(Integer, nullable=True)  # 0-100
    cognitive_performance_score = Column(Float, nullable=True)
    
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relacionamento
    patient = relationship("PatientModel", back_populates="neurobiological_markers")
    
    def __repr__(self):
        return f"<NeurobiologicalMarker {self.id}>"


class PsychosocialFactorModel(Base):
    """Modelo de Fatores Psicossociais"""
    __tablename__ = "psychosocial_factors"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    patient_id = Column(String, ForeignKey("patients.patient_id"), nullable=False)
    
    assessment_date = Column(DateTime, default=datetime.utcnow)
    
    # Vida pessoal
    marital_status = Column(String(50), nullable=True)
    relationship_quality_score = Column(Integer, nullable=True)  # 0-10
    children = Column(Integer, default=0)
    living_situation = Column(String(100), nullable=True)
    
    # Suporte social
    close_relationships = Column(Integer, default=0)
    social_network_quality = Column(Integer, nullable=True)  # 0-10
    perceived_support_score = Column(Integer, nullable=True)  # 0-10
    social_isolation = Column(Boolean, default=False)
    
    # Estressores
    major_life_stressors = Column(JSON, nullable=True)  # Lista de estressores
    recent_trauma = Column(Boolean, default=False)
    childhood_trauma = Column(Boolean, default=False)
    ace_score = Column(Integer, nullable=True)  # Adverse Childhood Experiences
    
    # Ocupação
    employment_status = Column(String(50), nullable=True)
    job_satisfaction_score = Column(Integer, nullable=True)  # 0-10
    work_stress_level = Column(String(50), default="moderate")
    financial_stress = Column(Boolean, default=False)
    
    # Qualidade de vida
    quality_of_life_score = Column(Float, nullable=True)  # 0-100
    activities_engagement = Column(Integer, nullable=True)  # 0-10
    purpose_meaning_score = Column(Integer, nullable=True)  # 0-10
    
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relacionamento
    patient = relationship("PatientModel", back_populates="psychosocial_factors")
    
    def __repr__(self):
        return f"<PsychosocialFactor {self.id}>"


class ClinicalAssessmentModel(Base):
    """Modelo de Avaliação Clínica"""
    __tablename__ = "clinical_assessments"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    patient_id = Column(String, ForeignKey("patients.patient_id"), nullable=False)
    
    assessment_date = Column(DateTime, default=datetime.utcnow)
    
    # Escalas clínicas
    phq9_score = Column(Integer, nullable=False)  # 0-27
    hamilton_score = Column(Integer, nullable=False)  # 0-52
    madrs_score = Column(Integer, nullable=False)  # 0-60
    
    # Características clínicas
    episode_duration_months = Column(Integer, nullable=True)
    num_previous_episodes = Column(Integer, default=0)
    age_of_onset = Column(Integer, nullable=True)
    
    # Comorbidades
    anxiety_disorder = Column(Boolean, default=False)
    substance_abuse = Column(Boolean, default=False)
    personality_disorder = Column(Boolean, default=False)
    
    # Histórico
    medication_trials = Column(JSON, default=[])  # Lista de medicações
    psychotherapy_history = Column(Boolean, default=False)
    hospitalization_history = Column(Boolean, default=False)
    ect_treatments = Column(Integer, default=0)
    
    # Ideação suicida
    suicidal_ideation = Column(Boolean, default=False)
    suicidal_ideation_severity = Column(Integer, nullable=True)  # 0-10
    
    # Histórico familiar
    family_history_depression = Column(Boolean, default=False)
    
    # Avaliação
    severity_classification = Column(String(50), nullable=True)
    treatment_resistance_classification = Column(String(50), nullable=True)
    
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relacionamento
    patient = relationship("PatientModel", back_populates="clinical_assessments")
    
    def __repr__(self):
        return f"<ClinicalAssessment {self.id}>"


class TreatmentPlanModel(Base):
    """Modelo de Plano de Tratamento"""
    __tablename__ = "treatment_plans"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    patient_id = Column(String, ForeignKey("patients.patient_id"), nullable=False)
    
    created_date = Column(DateTime, default=datetime.utcnow)
    start_date = Column(DateTime, nullable=False)
    
    # Recomendações
    medications = Column(JSON, nullable=True)  # Lista de medicações recomendadas
    therapies = Column(JSON, nullable=True)  # Lista de terapias recomendadas
    lifestyle_interventions = Column(JSON, nullable=True)
    monitoring_plan = Column(JSON, nullable=True)
    
    # Estimativas
    estimated_response_probability = Column(Float, nullable=True)
    overall_confidence = Column(Float, nullable=True)
    
    # Status
    is_active = Column(Boolean, default=True)
    
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relacionamento
    patient = relationship("PatientModel", back_populates="treatment_plans")
    
    def __repr__(self):
        return f"<TreatmentPlan {self.id}>"


class OutcomeAssessmentModel(Base):
    """Modelo de Avaliação de Desfechos"""
    __tablename__ = "outcome_assessments"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    patient_id = Column(String, ForeignKey("patients.patient_id"), nullable=False)
    
    assessment_date = Column(DateTime, default=datetime.utcnow)
    assessment_week = Column(Integer, nullable=True)  # Semana desde início tratamento
    
    # Sintomas
    phq9_score = Column(Integer, nullable=True)
    gad7_score = Column(Integer, nullable=True)
    madrs_score = Column(Integer, nullable=True)
    hamilton_score = Column(Integer, nullable=True)
    
    # Sintomas específicos
    sleep_quality = Column(Integer, nullable=True)  # 0-10
    energy_level = Column(Integer, nullable=True)  # 0-10
    appetite = Column(Integer, nullable=True)  # 0-10
    concentration = Column(Integer, nullable=True)  # 0-10
    motivation = Column(Integer, nullable=True)  # 0-10
    
    # Funcionamento
    work_days_missed = Column(Integer, default=0)
    productivity_percentage = Column(Float, default=100.0)
    social_activities = Column(Integer, nullable=True)  # 0-10
    
    # Adesão
    medication_adherence_percentage = Column(Float, default=100.0)
    missed_doses = Column(Integer, default=0)
    
    # Efeitos adversos
    adverse_effects = Column(JSON, nullable=True)  # Lista de efeitos adversos
    
    # Medicações atuais
    current_medications = Column(JSON, nullable=True)
    
    # Avaliação geral
    clinician_global_impression = Column(Integer, nullable=True)  # 1-7
    
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relacionamento
    patient = relationship("PatientModel", back_populates="outcome_assessments")
    
    def __repr__(self):
        return f"<OutcomeAssessment {self.id}>"


class PredictionModel(Base):
    """Modelo de Predições ML"""
    __tablename__ = "predictions"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    patient_id = Column(String, ForeignKey("patients.patient_id"), nullable=False)
    
    prediction_date = Column(DateTime, default=datetime.utcnow)
    
    # Predição de resposta terapêutica
    therapeutic_response_probability = Column(Float, nullable=True)
    therapeutic_response_category = Column(String(50), nullable=True)  # Responder, Partial, Non-responder
    therapeutic_response_confidence = Column(Float, nullable=True)
    
    # Predição de resistência
    treatment_resistance_probability = Column(Float, nullable=True)
    treatment_resistance_category = Column(String(50), nullable=True)
    treatment_resistance_confidence = Column(Float, nullable=True)
    
    # Predições de medicamentos
    medication_predictions = Column(JSON, nullable=True)  # Dict de medicações e probabilidades
    
    # Features utilizadas
    features_used = Column(JSON, nullable=True)
    features_count = Column(Integer, nullable=True)
    
    # Interpretabilidade
    top_features = Column(JSON, nullable=True)  # Features mais importantes
    
    # Metadados
    model_name = Column(String(100), nullable=True)
    model_version = Column(String(20), nullable=True)
    response_time_ms = Column(Float, nullable=True)
    
    # Status
    is_current = Column(Boolean, default=True)
    
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relacionamento
    patient = relationship("PatientModel", back_populates="predictions")
    
    def __repr__(self):
        return f"<Prediction {self.id}>"


class AuditLogModel(Base):
    """Modelo de Log de Auditoria (GDPR/HIPAA)"""
    __tablename__ = "audit_logs"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    
    timestamp = Column(DateTime, default=datetime.utcnow)
    action = Column(String(100), nullable=False)  # create, read, update, delete
    actor = Column(String(255), nullable=True)  # User who performed action (hashed for PII)
    resource = Column(String(100), nullable=False)  # patient, assessment, prediction
    resource_id = Column(String(255), nullable=False)  # ID of resource
    
    # Detalhes
    change_summary = Column(Text, nullable=True)
    previous_value = Column(JSON, nullable=True)
    new_value = Column(JSON, nullable=True)
    
    # Contexto
    ip_address = Column(String(50), nullable=True)
    user_agent = Column(String(255), nullable=True)
    
    # Compliance
    gdpr_relevant = Column(Boolean, default=False)
    hipaa_relevant = Column(Boolean, default=False)
    
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    def __repr__(self):
        return f"<AuditLog {self.id}>"


class ModelPerformanceModel(Base):
    """Modelo para Rastrear Performance de Modelos"""
    __tablename__ = "model_performance"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    
    model_name = Column(String(100), nullable=False)
    model_version = Column(String(20), nullable=False)
    
    evaluation_date = Column(DateTime, default=datetime.utcnow)
    cohort_name = Column(String(100), nullable=True)  # validation, external, production
    
    # Métricas
    auc_roc = Column(Float, nullable=True)
    sensitivity = Column(Float, nullable=True)
    specificity = Column(Float, nullable=True)
    precision = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)
    accuracy = Column(Float, nullable=True)
    
    # Calibração
    expected_calibration_error = Column(Float, nullable=True)
    brier_score = Column(Float, nullable=True)
    
    # Fairness
    bias_detected = Column(Boolean, default=False)
    max_bias_value = Column(Float, nullable=True)
    
    # Contexto
    sample_size = Column(Integer, nullable=True)
    notes = Column(Text, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<ModelPerformance {self.model_name} v{self.model_version}>"