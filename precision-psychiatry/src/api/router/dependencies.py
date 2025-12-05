"""
src/api/dependencies.py - Injeção de dependências FastAPI
"""

from typing import Optional
from functools import lru_cache
import logging
from sqlalchemy.orm import Session

from src.config import get_config
from src.ml_pipeline.pipeline import PrecisionPsychiatryPipeline
from src.database.db import DatabaseManager
from src.database.repositories import PatientRepository, PredictionRepository

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_config_instance():
    """Obter configuração (cached)."""
    return get_config()


@lru_cache(maxsize=1)
def get_model() -> PrecisionPsychiatryPipeline:
    """
    Obter modelo treinado (cached).
    """
    config = get_config_instance()
    try:
        pipeline = PrecisionPsychiatryPipeline(model_type="xgboost")
        model_path = config.models_dir / "v1"
        if model_path.exists():
            pipeline.load(model_path)
            logger.info("✓ Modelo carregado com sucesso")
        else:
            logger.warning("Modelo não encontrado. Usando modelo sem peso.")
        return pipeline
    except Exception as e:
        logger.error(f"Erro ao carregar modelo: {e}")
        raise


@lru_cache(maxsize=1)
def get_db_manager() -> DatabaseManager:
    """Obter gerenciador de BD (cached)."""
    config = get_config_instance()
    return DatabaseManager(config.database.url)


def get_db_session() -> Session:
    """Obter sessão de BD."""
    db_manager = get_db_manager()
    with db_manager.get_session() as session:
        yield session


def get_patient_repository(session: Session = None) -> PatientRepository:
    """Obter repository de pacientes."""
    if session is None:
        db_manager = get_db_manager()
        session = db_manager.get_session_sync()
    return PatientRepository(session)


def get_prediction_repository(session: Session = None) -> PredictionRepository:
    """Obter repository de predições."""
    if session is None:
        db_manager = get_db_manager()
        session = db_manager.get_session_sync()
    return PredictionRepository(session)


---

"""
src/api/routes/predictions.py - Endpoints de predições
"""

from fastapi import APIRouter, Depends, HTTPException, Header
from typing import List, Dict, Any
from datetime import datetime

from src.api.schemas import PatientProfile, PredictionResponse
from src.api.dependencies import get_model, get_prediction_repository

router = APIRouter(prefix="/api/v1", tags=["predictions"])


@router.post("/predict", response_model=PredictionResponse)
async def predict_treatment_response(
    patient: PatientProfile,
    x_api_key: str = Header(...),
    model = Depends(get_model),
    pred_repo = Depends(get_prediction_repository)
) -> PredictionResponse:
    """
    Predizer resposta ao tratamento.
    
    Args:
        patient: Perfil do paciente
        x_api_key: API Key
        model: Modelo ML (dependência)
        pred_repo: Repository de predições
    
    Returns:
        PredictionResponse
    """
    # Validar API key
    from src.config import get_config
    config = get_config()
    if x_api_key != config.api_key:
        raise HTTPException(status_code=403, detail="API key inválida")
    
    # Fazer predição (implementação similar a scripts/inference.py)
    try:
        # ... lógica de predição ...
        prediction_response = PredictionResponse(
            patient_id=patient.patient_id,
            prediction=0.75,
            risk_category="moderate",
            confidence=0.75,
            recommended_medications=[],
            contraindications=[],
            explanation={},
            generated_at=datetime.now()
        )
        
        # Salvar no BD
        pred_repo.save({
            "patient_id": patient.patient_id,
            "prediction_probability": prediction_response.prediction,
            "risk_category": prediction_response.risk_category
        })
        
        return prediction_response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/patients/{patient_id}/predictions")
async def get_patient_predictions(
    patient_id: str,
    x_api_key: str = Header(...),
    pred_repo = Depends(get_prediction_repository)
) -> List[Dict[str, Any]]:
    """Obter predições de um paciente."""
    from src.config import get_config
    config = get_config()
    if x_api_key != config.api_key:
        raise HTTPException(status_code=403, detail="API key inválida")
    
    predictions = pred_repo.get_by_version("latest")
    return [{"id": p.id, "probability": p.prediction_probability} for p in predictions]


---

"""
src/api/routes/patients.py - Endpoints de pacientes
"""

from fastapi import APIRouter, Depends, HTTPException, Header
from typing import List

from src.api.dependencies import get_patient_repository

router = APIRouter(prefix="/api/v1", tags=["patients"])


@router.get("/patients/{patient_id}")
async def get_patient(
    patient_id: str,
    x_api_key: str = Header(...),
    repo = Depends(get_patient_repository)
):
    """Obter dados de paciente."""
    from src.config import get_config
    config = get_config()
    if x_api_key != config.api_key:
        raise HTTPException(status_code=403, detail="API key inválida")
    
    patient = repo.get_by_id(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Paciente não encontrado")
    
    return {
        "patient_id": patient.patient_id,
        "age": patient.age,
        "gender": patient.gender,
        "created_at": patient.created_at
    }


@router.get("/patients")
async def list_patients(
    limit: int = 100,
    x_api_key: str = Header(...),
    repo = Depends(get_patient_repository)
) -> List[dict]:
    """Listar pacientes."""
    from src.config import get_config
    config = get_config()
    if x_api_key != config.api_key:
        raise HTTPException(status_code=403, detail="API key inválida")
    
    patients = repo.get_all(limit)
    return [{"patient_id": p.patient_id, "age": p.age} for p in patients]


---

"""
src/api/routes/__init__.py
"""

from src.api.routes.predictions import router as predictions_router
from src.api.routes.patients import router as patients_router

__all__ = ['predictions_router', 'patients_router']