from celery import shared_task
from src.tasks.celery_app import app
from src.ml_pipeline.pipeline import run_full_pipeline
from src.data.generators import generate_and_save_dataset
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@shared_task(bind=True, max_retries=3)
def train_model_task(self, n_samples: int = 500, model_type: str = "xgboost"):
    """
    Task de treinamento de modelo.
    
    Args:
        n_samples: Número de amostras
        model_type: Tipo de modelo
    """
    try:
        logger.info(f"Iniciando treinamento com {n_samples} amostras")
        
        # Gerar dados
        dataset, _ = generate_and_save_dataset(n_patients=n_samples)
        
        # Preparar
        y = dataset["responder"].astype(int)
        drop_cols = ["responder", "response_status", "refractory",
                    "response_probability", "patient_id", "created_at",
                    "cohort", "phq9_baseline", "phq9_week12"]
        X = dataset.drop(columns=[col for col in drop_cols if col in dataset.columns])
        
        # Treinar
        pipeline, metrics = run_full_pipeline(
            X, y,
            test_size=0.2,
            model_type=model_type
        )
        
        logger.info(f"✓ Treinamento concluído: AUC={metrics['auc_roc']:.4f}")
        
        return {
            "status": "success",
            "auc": metrics["auc_roc"],
            "n_samples": n_samples
        }
    
    except Exception as exc:
        logger.error(f"Erro no treinamento: {exc}")
        self.retry(exc=exc, countdown=60)


@shared_task
def cleanup_old_predictions(days: int = 30):
    """Limpar predições antigas."""
    logger.info(f"Limpando predições com mais de {days} dias")
    # Implementar lógica de limpeza
    return {"status": "cleaned"}


@shared_task
def generate_daily_report():
    """Gerar relatório diário."""
    logger.info("Gerando relatório diário")
    # Implementar lógica de relatório
    return {"status": "report_generated"}