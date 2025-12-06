"""
Sistema de Logging Estruturado para Precision Psychiatry
Registra eventos clínicos, ML e operacionais com rastreabilidade completa
"""

import logging
import logging.handlers
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
from enum import Enum
import structlog
from pythonjsonlogger import jsonlogger
import traceback
import os


class LogLevel(Enum):
    """Níveis de logging customizados"""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL
    AUDIT = 25  # Nível customizado para auditoria


class LogCategory(Enum):
    """Categorias de log para o projeto"""
    CLINICAL = "CLINICAL"  # Eventos clínicos
    ML_MODEL = "ML_MODEL"  # Eventos de ML
    DATA = "DATA"  # Eventos de processamento de dados
    API = "API"  # Requisições API
    SECURITY = "SECURITY"  # Eventos de segurança
    PERFORMANCE = "PERFORMANCE"  # Métricas de performance
    AUDIT = "AUDIT"  # Rastreabilidade e compliance
    VALIDATION = "VALIDATION"  # Validação de dados
    ERROR = "ERROR"  # Erros e exceções


class PrecisionPsychiatryLogger:
    """Logger estruturado para o projeto"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        self.logger = self._setup_logging()
    
    def _setup_logging(self) -> structlog.BoundLogger:
        """Configura logging estruturado com structlog"""
        
        # Criar diretório de logs se não existir
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        
        # Configurar structlog
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )
        
        # Configurar logging stdlib
        logging.basicConfig(
            format="%(message)s",
            stream=sys.stdout,
            level=logging.INFO,
        )
        
        # Handler para arquivo JSON (estruturado)
        json_handler = logging.handlers.RotatingFileHandler(
            filename=logs_dir / "precision_psychiatry.json",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=10,
            encoding='utf-8'
        )
        json_handler.setFormatter(jsonlogger.JsonFormatter())
        logging.getLogger().addHandler(json_handler)
        
        # Handler para arquivo de texto legível
        text_handler = logging.handlers.RotatingFileHandler(
            filename=logs_dir / "precision_psychiatry.log",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=10,
            encoding='utf-8'
        )
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        text_handler.setFormatter(formatter)
        logging.getLogger().addHandler(text_handler)
        
        # Handler para erros críticos (arquivo separado)
        error_handler = logging.handlers.RotatingFileHandler(
            filename=logs_dir / "errors.log",
            maxBytes=5 * 1024 * 1024,  # 5MB
            backupCount=5,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        logging.getLogger().addHandler(error_handler)
        
        # Handler para auditoria (compliance)
        audit_handler = logging.handlers.RotatingFileHandler(
            filename=logs_dir / "audit.log",
            maxBytes=20 * 1024 * 1024,  # 20MB
            backupCount=20,
            encoding='utf-8'
        )
        audit_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        audit_handler.setFormatter(audit_formatter)
        logging.getLogger().addHandler(audit_handler)
        
        return structlog.get_logger()
    
    def log_clinical_event(
        self,
        event: str,
        patient_id: str,
        event_type: str,
        details: Dict[str, Any],
        severity: str = "INFO"
    ):
        """Log de eventos clínicos (HIPAA-compliant)"""
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "category": LogCategory.CLINICAL.value,
            "event": event,
            "patient_id": self._hash_pii(patient_id),  # Hash para privacidade
            "event_type": event_type,
            "details": details,
            "severity": severity
        }
        
        self.logger.log(
            severity,
            "clinical_event",
            **log_entry
        )
    
    def log_prediction(
        self,
        patient_id: str,
        model_name: str,
        prediction: float,
        confidence: float,
        features_used: int,
        response_time_ms: float
    ):
        """Log de predições ML"""
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "category": LogCategory.ML_MODEL.value,
            "event": "prediction_generated",
            "patient_id": self._hash_pii(patient_id),
            "model": model_name,
            "prediction_value": round(prediction, 4),
            "confidence_score": round(confidence, 4),
            "features_count": features_used,
            "response_time_ms": round(response_time_ms, 2)
        }
        
        self.logger.info("ml_prediction", **log_entry)
    
    def log_data_processing(
        self,
        step: str,
        input_count: int,
        output_count: int,
        success: bool,
        processing_time_ms: float,
        error_message: Optional[str] = None
    ):
        """Log de processamento de dados"""
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "category": LogCategory.DATA.value,
            "step": step,
            "input_records": input_count,
            "output_records": output_count,
            "success": success,
            "processing_time_ms": round(processing_time_ms, 2),
            "error": error_message
        }
        
        level = "info" if success else "warning"
        self.logger.log(level, "data_processing", **log_entry)
    
    def log_api_request(
        self,
        endpoint: str,
        method: str,
        status_code: int,
        response_time_ms: float,
        user_id: Optional[str] = None,
        error: Optional[str] = None
    ):
        """Log de requisições API"""
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "category": LogCategory.API.value,
            "endpoint": endpoint,
            "method": method,
            "status_code": status_code,
            "response_time_ms": round(response_time_ms, 2),
            "user_id": self._hash_pii(user_id) if user_id else "anonymous",
            "error": error
        }
        
        level = "info" if status_code < 400 else "warning"
        self.logger.log(level, "api_request", **log_entry)
    
    def log_security_event(
        self,
        event_type: str,
        severity: str,
        user_id: Optional[str],
        resource: str,
        action: str,
        result: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """Log de eventos de segurança (COMPLIANCE)"""
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "category": LogCategory.SECURITY.value,
            "event_type": event_type,
            "severity": severity,
            "user_id": self._hash_pii(user_id) if user_id else "system",
            "resource": resource,
            "action": action,
            "result": result,
            "details": details,
            "ip_address": os.environ.get("CLIENT_IP", "unknown")
        }
        
        self.logger.log(severity, "security_event", **log_entry)
    
    def log_model_performance(
        self,
        model_name: str,
        metric_name: str,
        metric_value: float,
        threshold: float,
        passed: bool,
        cohort: str = "validation"
    ):
        """Log de métricas de performance do modelo"""
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "category": LogCategory.PERFORMANCE.value,
            "model": model_name,
            "metric": metric_name,
            "value": round(metric_value, 4),
            "threshold": round(threshold, 4),
            "passed": passed,
            "cohort": cohort
        }
        
        level = "info" if passed else "warning"
        self.logger.log(level, "model_performance", **log_entry)
    
    def log_validation_check(
        self,
        check_name: str,
        status: str,
        details: Dict[str, Any],
        data_quality_score: Optional[float] = None
    ):
        """Log de validação de dados"""
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "category": LogCategory.VALIDATION.value,
            "check": check_name,
            "status": status,
            "details": details,
            "data_quality_score": data_quality_score
        }
        
        level = "info" if status == "PASSED" else "warning"
        self.logger.log(level, "validation_check", **log_entry)
    
    def log_audit_trail(
        self,
        action: str,
        actor: str,
        resource: str,
        resource_id: str,
        change_summary: str,
        timestamp: Optional[datetime] = None
    ):
        """Log de auditoria para compliance GDPR/HIPAA"""
        
        audit_entry = {
            "timestamp": (timestamp or datetime.now()).isoformat(),
            "category": LogCategory.AUDIT.value,
            "action": action,
            "actor": self._hash_pii(actor),
            "resource": resource,
            "resource_id": self._hash_pii(resource_id),
            "change_summary": change_summary
        }
        
        # Logging para arquivo de auditoria (não estruturado para simplicidade)
        audit_msg = (
            f"{audit_entry['timestamp']} | {action} | {resource} | "
            f"{change_summary}"
        )
        self.logger.info("audit_trail", **audit_entry)
    
    def log_error(
        self,
        error_type: str,
        error_message: str,
        patient_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        traceback_str: Optional[str] = None
    ):
        """Log de erros com contexto completo"""
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "category": LogCategory.ERROR.value,
            "error_type": error_type,
            "error_message": error_message,
            "patient_id": self._hash_pii(patient_id) if patient_id else None,
            "context": context,
            "traceback": traceback_str
        }
        
        self.logger.error("error_occurred", **log_entry)
    
    def log_fairness_check(
        self,
        demographic_group: str,
        metric_name: str,
        value: float,
        expected_range: tuple,
        bias_detected: bool
    ):
        """Log de verificações de fairness/equidade"""
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "category": LogCategory.VALIDATION.value,
            "check_type": "fairness",
            "demographic_group": demographic_group,
            "metric": metric_name,
            "value": round(value, 4),
            "expected_range": expected_range,
            "bias_detected": bias_detected
        }
        
        level = "info" if not bias_detected else "warning"
        self.logger.log(level, "fairness_check", **log_entry)
    
    @staticmethod
    def _hash_pii(data: Optional[str]) -> Optional[str]:
        """Hashear dados pessoais identificáveis (PII) para privacidade"""
        
        if not data:
            return None
        
        import hashlib
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def get_logger(self) -> structlog.BoundLogger:
        """Retorna o logger para uso direto"""
        return self.logger


# Singleton global
logger_instance = PrecisionPsychiatryLogger()


def get_logger() -> structlog.BoundLogger:
    """Função de conveniência para obter o logger"""
    return logger_instance.get_logger()


# Context managers para rastreamento de eventos

class LoggingContext:
    """Context manager para log de operações com tempo de execução"""
    
    def __init__(self, operation_name: str, **context):
        self.operation_name = operation_name
        self.context = context
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        logger_instance.logger.info(
            f"operation_start",
            operation=self.operation_name,
            **self.context
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_ms = (datetime.now() - self.start_time).total_seconds() * 1000
        
        if exc_type:
            logger_instance.log_error(
                error_type=exc_type.__name__,
                error_message=str(exc_val),
                context={
                    "operation": self.operation_name,
                    "elapsed_ms": round(elapsed_ms, 2),
                    **self.context
                },
                traceback_str=traceback.format_exc()
            )
        else:
            logger_instance.logger.info(
                "operation_complete",
                operation=self.operation_name,
                elapsed_ms=round(elapsed_ms, 2),
                **self.context
            )


# Exemplo de uso
if __name__ == "__main__":
    logger = get_logger()
    
    # Log clínico
    logger_instance.log_clinical_event(
        event="Patient Assessment",
        patient_id="PAT_12345",
        event_type="initial_evaluation",
        details={"phq9_score": 22, "severity": "moderate"},
        severity="INFO"
    )
    
    # Log de predição
    logger_instance.log_prediction(
        patient_id="PAT_12345",
        model_name="therapeutic_response_v1",
        prediction=0.75,
        confidence=0.82,
        features_used=35,
        response_time_ms=145.3
    )
    
    # Log com contexto
    with LoggingContext("data_import", source="genomic_lab", batch_id="BATCH_001"):
        logger.info("Processing genomic data", records=500)