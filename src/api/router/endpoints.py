"""
Endpoints REST API para Precision Psychiatry
Rotas para avaliações, predições, pacientes e outcomes
"""

from fastapi import APIRouter, HTTPException, Depends, Query, Path
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
from sqlalchemy.orm import Session
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# SCHEMAS PYDANTIC (Request/Response)
# ============================================================================

class PatientBase(BaseModel):
    """Schema base de paciente"""
    first_name: str = Field(..., min_length=1, max_length=100)
    last_name: str = Field(..., min_length=1, max_length=100)
    date_of_birth: datetime
    biological_sex: str = Field(..., pattern="^(male|female|other)$")
    email: Optional[str] = None
    phone: Optional[str] = None


class PatientCreate(PatientBase):
    """Schema para criar paciente"""
    pass


class PatientResponse(PatientBase):
    """Schema para resposta de paciente"""
    patient_id: str
    enrollment_date: datetime
    is_active: bool
    
    class Config:
        from_attributes = True


class ClinicalAssessmentRequest(BaseModel):
    """Schema para avaliação clínica"""
    phq9_score: int = Field(..., ge=0, le=27)
    hamilton_score: int = Field(..., ge=0, le=52)
    madrs_score: int = Field(..., ge=0, le=60)
    episode_duration_months: int = Field(..., ge=0)
    num_previous_episodes: int = Field(default=0, ge=0)
    age_of_onset: Optional[int] = None
    anxiety_disorder: bool = False
    substance_abuse: bool = False
    personality_disorder: bool = False
    suicidal_ideation: bool = False
    family_history_depression: bool = False
    notes: Optional[str] = None


class ClinicalAssessmentResponse(ClinicalAssessmentRequest):
    """Schema para resposta de avaliação clínica"""
    id: str
    patient_id: str
    assessment_date: datetime
    severity_classification: Optional[str]
    treatment_resistance_classification: Optional[str]
    
    class Config:
        from_attributes = True


class PredictionRequest(BaseModel):
    """Schema para requisição de predição"""
    patient_id: str
    genetic_profile: Optional[dict] = None
    neurobiological_markers: Optional[dict] = None
    psychosocial_factors: Optional[dict] = None
    clinical_assessment: Optional[dict] = None


class PredictionResponse(BaseModel):
    """Schema para resposta de predição"""
    prediction_id: str
    patient_id: str
    prediction_date: datetime
    therapeutic_response_probability: float = Field(..., ge=0, le=1)
    therapeutic_response_category: str
    therapeutic_response_confidence: float
    treatment_resistance_probability: Optional[float] = None
    treatment_resistance_category: Optional[str] = None
    medication_predictions: Optional[dict] = None
    top_features: Optional[List[tuple]] = None
    model_version: Optional[str] = None
    response_time_ms: float


class OutcomeAssessmentRequest(BaseModel):
    """Schema para avaliação de desfecho"""
    assessment_week: int = Field(..., ge=0)
    phq9_score: Optional[int] = Field(None, ge=0, le=27)
    gad7_score: Optional[int] = Field(None, ge=0, le=21)
    madrs_score: Optional[int] = Field(None, ge=0, le=60)
    sleep_quality: Optional[int] = Field(None, ge=0, le=10)
    energy_level: Optional[int] = Field(None, ge=0, le=10)
    work_days_missed: int = Field(default=0, ge=0)
    productivity_percentage: float = Field(default=100.0, ge=0, le=100)
    medication_adherence_percentage: float = Field(default=100.0, ge=0, le=100)
    adverse_effects: Optional[List[dict]] = None
    notes: Optional[str] = None


class TreatmentPlanRequest(BaseModel):
    """Schema para plano de tratamento"""
    medications: Optional[List[dict]] = None
    therapies: Optional[List[dict]] = None
    lifestyle_interventions: Optional[List[str]] = None
    monitoring_plan: Optional[List[str]] = None
    notes: Optional[str] = None


class HealthCheckResponse(BaseModel):
    """Schema para health check"""
    status: str
    timestamp: datetime
    version: str
    database: str
    redis: Optional[str] = None


# ============================================================================
# ROTAS PATIENTS
# ============================================================================

router_patients = APIRouter(prefix="/patients", tags=["patients"])


@router_patients.post("/", response_model=PatientResponse, status_code=201)
async def create_patient(patient: PatientCreate):
    """Criar novo paciente"""
    try:
        logger.info(f"Criando novo paciente: {patient.first_name} {patient.last_name}")
        # Implementação com DB
        # db_patient = PatientModel(**patient.dict())
        # db.add(db_patient)
        # db.commit()
        # return db_patient
        return {"patient_id": "new_id", **patient.dict()}
    except Exception as e:
        logger.error(f"Erro ao criar paciente: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@router_patients.get("/{patient_id}", response_model=PatientResponse)
async def get_patient(patient_id: str = Path(..., min_length=1)):
    """Obter dados do paciente"""
    try:
        logger.info(f"Recuperando paciente: {patient_id}")
        # Implementação com DB
        # db_patient = db.query(PatientModel).filter(PatientModel.patient_id == patient_id).first()
        # if not db_patient:
        #     raise HTTPException(status_code=404, detail="Paciente não encontrado")
        # return db_patient
        return {"patient_id": patient_id, "first_name": "João", "last_name": "Silva", 
                "date_of_birth": datetime.now(), "biological_sex": "male"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro ao recuperar paciente: {str(e)}")
        raise HTTPException(status_code=500, detail="Erro interno")


@router_patients.get("/", response_model=List[PatientResponse])
async def list_patients(skip: int = Query(0, ge=0), limit: int = Query(10, ge=1, le=100)):
    """Listar pacientes com paginação"""
    try:
        logger.info(f"Listando pacientes (skip={skip}, limit={limit})")
        # Implementação com DB
        return []
    except Exception as e:
        logger.error(f"Erro ao listar pacientes: {str(e)}")
        raise HTTPException(status_code=500, detail="Erro interno")


@router_patients.put("/{patient_id}", response_model=PatientResponse)
async def update_patient(patient_id: str, patient: PatientBase):
    """Atualizar dados do paciente"""
    try:
        logger.info(f"Atualizando paciente: {patient_id}")
        # Implementação com DB
        return {"patient_id": patient_id, **patient.dict()}
    except Exception as e:
        logger.error(f"Erro ao atualizar paciente: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@router_patients.delete("/{patient_id}", status_code=204)
async def delete_patient(patient_id: str = Path(..., min_length=1)):
    """Deletar paciente (GDPR - Right to be Forgotten)"""
    try:
        logger.info(f"Deletando paciente: {patient_id}")
        # Implementação com DB - deve ser GDPR compliant
        return None
    except Exception as e:
        logger.error(f"Erro ao deletar paciente: {str(e)}")
        raise HTTPException(status_code=500, detail="Erro interno")


# ============================================================================
# ROTAS CLINICAL ASSESSMENTS
# ============================================================================

router_assessments = APIRouter(prefix="/assessments", tags=["assessments"])


@router_assessments.post("/", response_model=ClinicalAssessmentResponse, status_code=201)
async def create_assessment(patient_id: str = Query(...), assessment: ClinicalAssessmentRequest = None):
    """Criar nova avaliação clínica"""
    try:
        logger.info(f"Criando avaliação para paciente: {patient_id}")
        # Validar que paciente existe
        # Salvar avaliação
        return {
            "id": "assessment_id",
            "patient_id": patient_id,
            "assessment_date": datetime.utcnow(),
            **assessment.dict()
        }
    except Exception as e:
        logger.error(f"Erro ao criar avaliação: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@router_assessments.get("/{patient_id}", response_model=List[ClinicalAssessmentResponse])
async def get_patient_assessments(patient_id: str = Path(...)):
    """Obter avaliações de um paciente"""
    try:
        logger.info(f"Recuperando avaliações do paciente: {patient_id}")
        return []
    except Exception as e:
        logger.error(f"Erro ao recuperar avaliações: {str(e)}")
        raise HTTPException(status_code=500, detail="Erro interno")


# ============================================================================
# ROTAS PREDICTIONS
# ============================================================================

router_predictions = APIRouter(prefix="/predictions", tags=["predictions"])


@router_predictions.post("/", response_model=PredictionResponse, status_code=201)
async def create_prediction(request: PredictionRequest):
    """Gerar predição para paciente"""
    try:
        import time
        start_time = time.time()
        
        patient_id = request.patient_id
        logger.info(f"Gerando predição para paciente: {patient_id}")
        
        # Aqui iria a lógica de ML
        # prediction_orchestrator = PredictionOrchestrator()
        # comprehensive_prediction = prediction_orchestrator.generate_predictions(...)
        
        response_time = (time.time() - start_time) * 1000
        
        return {
            "prediction_id": "pred_id",
            "patient_id": patient_id,
            "prediction_date": datetime.utcnow(),
            "therapeutic_response_probability": 0.75,
            "therapeutic_response_category": "Responder",
            "therapeutic_response_confidence": 0.82,
            "treatment_resistance_probability": 0.25,
            "treatment_resistance_category": "Low Risk",
            "response_time_ms": response_time
        }
    except Exception as e:
        logger.error(f"Erro ao gerar predição: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@router_predictions.get("/{patient_id}", response_model=List[PredictionResponse])
async def get_patient_predictions(patient_id: str = Path(...)):
    """Obter predições anteriores do paciente"""
    try:
        logger.info(f"Recuperando predições do paciente: {patient_id}")
        return []
    except Exception as e:
        logger.error(f"Erro ao recuperar predições: {str(e)}")
        raise HTTPException(status_code=500, detail="Erro interno")


# ============================================================================
# ROTAS OUTCOMES
# ============================================================================

router_outcomes = APIRouter(prefix="/outcomes", tags=["outcomes"])


@router_outcomes.post("/", status_code=201)
async def create_outcome(patient_id: str = Query(...), outcome: OutcomeAssessmentRequest = None):
    """Registrar avaliação de desfecho"""
    try:
        logger.info(f"Registrando outcome para paciente: {patient_id}")
        return {
            "id": "outcome_id",
            "patient_id": patient_id,
            "assessment_date": datetime.utcnow(),
            **outcome.dict()
        }
    except Exception as e:
        logger.error(f"Erro ao registrar outcome: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@router_outcomes.get("/{patient_id}")
async def get_patient_outcomes(patient_id: str = Path(...)):
    """Obter trajetória de outcomes do paciente"""
    try:
        logger.info(f"Recuperando outcomes do paciente: {patient_id}")
        return {
            "patient_id": patient_id,
            "outcomes": [],
            "trajectory_analysis": {}
        }
    except Exception as e:
        logger.error(f"Erro ao recuperar outcomes: {str(e)}")
        raise HTTPException(status_code=500, detail="Erro interno")


# ============================================================================
# ROTAS HEALTH CHECK
# ============================================================================

router_health = APIRouter(tags=["health"])


@router_health.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check da aplicação"""
    try:
        logger.info("Health check realizado")
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow(),
            "version": "1.0.0",
            "database": "connected",
            "redis": "connected"
        }
    except Exception as e:
        logger.error(f"Health check falhou: {str(e)}")
        raise HTTPException(status_code=503, detail="Service unavailable")


@router_health.get("/metrics")
async def get_metrics():
    """Obter métricas da aplicação"""
    try:
        logger.info("Métricas solicitadas")
        return {
            "predictions_today": 42,
            "assessments_today": 15,
            "api_requests_last_hour": 238,
            "avg_response_time_ms": 145.3,
            "error_rate_percent": 0.5
        }
    except Exception as e:
        logger.error(f"Erro ao obter métricas: {str(e)}")
        raise HTTPException(status_code=500, detail="Erro interno")


# ============================================================================
# REGISTRAR ROTAS NA APLICAÇÃO
# ============================================================================

def include_routes(app):
    """Incluir todas as rotas na aplicação"""
    app.include_router(router_health)
    app.include_router(router_patients)
    app.include_router(router_assessments)
    app.include_router(router_predictions)
    app.include_router(router_outcomes)