"""
Configuração centralizada do projeto de Medicina de Precisão em Psiquiatria.
Segue padrão de 12-factor app com suporte a múltiplos ambientes.
"""

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class Environment(str, Enum):
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class ModelType(str, Enum):
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    SVM = "svm"
    NEURAL_NETWORK = "neural_network"
    ENSEMBLE = "ensemble"


@dataclass
class DatabaseConfig:
    """Configuração de banco de dados."""
    host: str = os.getenv("DB_HOST", "localhost")
    port: int = int(os.getenv("DB_PORT", "5432"))
    user: str = os.getenv("DB_USER", "postgres")
    password: str = os.getenv("DB_PASSWORD", "postgres")
    database: str = os.getenv("DB_NAME", "precision_psychiatry")
    echo: bool = False
    pool_size: int = 20
    max_overflow: int = 40
    
    @property
    def url(self) -> str:
        """Construir URL de conexão."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


@dataclass
class Neo4jConfig:
    """Configuração de Neo4j para grafo genômico."""
    host: str = os.getenv("NEO4J_HOST", "localhost")
    port: int = int(os.getenv("NEO4J_PORT", "7687"))
    user: str = os.getenv("NEO4J_USER", "neo4j")
    password: str = os.getenv("NEO4J_PASSWORD", "password")
    database: str = os.getenv("NEO4J_DB", "neo4j")
    
    @property
    def uri(self) -> str:
        """Construir URI de conexão."""
        return f"bolt://{self.host}:{self.port}"


@dataclass
class RedisConfig:
    """Configuração de Redis para cache."""
    host: str = os.getenv("REDIS_HOST", "localhost")
    port: int = int(os.getenv("REDIS_PORT", "6379"))
    db: int = int(os.getenv("REDIS_DB", "0"))
    password: Optional[str] = os.getenv("REDIS_PASSWORD")
    ssl: bool = os.getenv("REDIS_SSL", "false").lower() == "true"


@dataclass
class APIConfig:
    """Configuração da API REST."""
    host: str = os.getenv("API_HOST", "0.0.0.0")
    port: int = int(os.getenv("API_PORT", "8000"))
    debug: bool = False
    workers: int = int(os.getenv("API_WORKERS", "4"))
    cors_origins: List[str] = field(default_factory=lambda: [
        os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
    ])


@dataclass
class MLConfig:
    """Configuração de modelos de ML."""
    default_model: ModelType = ModelType.XGBOOST
    random_state: int = 42
    test_size: float = 0.2
    validation_size: float = 0.1
    cv_folds: int = 5
    
    # Hiperparâmetros padrão
    xgboost_params: Dict = field(default_factory=lambda: {
        "max_depth": 7,
        "learning_rate": 0.05,
        "n_estimators": 200,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42
    })
    
    rf_params: Dict = field(default_factory=lambda: {
        "n_estimators": 100,
        "max_depth": 15,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "random_state": 42
    })
    
    ensemble_weights: Dict[ModelType, float] = field(default_factory=lambda: {
        ModelType.XGBOOST: 0.4,
        ModelType.RANDOM_FOREST: 0.3,
        ModelType.SVM: 0.3
    })


@dataclass
class FeatureConfig:
    """Configuração de features e limites de dados."""
    # Genes relevantes
    relevant_genes: List[str] = field(default_factory=lambda: [
        "CYP2D6", "CYP3A4", "CYP1A2", "CYP2C19", "CYP2B6",
        "MTHFR", "COMT", "BDNF", "5HTR1A", "TPH1", "SLC6A4"
    ])
    
    # Biomarcadores neurobiológicos
    neurobiological_markers: List[str] = field(default_factory=lambda: [
        "cortisol", "acth", "il6", "tnf_alpha", "crp",
        "serotonin", "dopamine", "noradrenaline",
        "tryptophan", "kynurenine", "bdnf"
    ])
    
    # Escalas psicométricas
    psychometric_scales: List[str] = field(default_factory=lambda: [
        "phq9", "gad7", "panss_positive", "panss_negative",
        "ctq_total", "ace_total", "psqi"
    ])
    
    # Variáveis demográficas
    demographic_vars: List[str] = field(default_factory=lambda: [
        "age", "gender", "bmi", "education_years"
    ])
    
    # Estratificação de risco
    risk_stratification: Dict[str, tuple] = field(default_factory=lambda: {
        "low": (0.0, 0.33),
        "moderate": (0.33, 0.66),
        "high": (0.66, 1.0)
    })


@dataclass
class ValidationConfig:
    """Configuração de validação clínica."""
    min_auc: float = 0.70
    min_sensitivity: float = 0.60
    min_specificity: float = 0.60
    max_demographic_bias: float = 0.05  # 5% máximo
    calibration_method: str = "isotonic"  # ou "sigmoid"


@dataclass
class LoggingConfig:
    """Configuração de logging."""
    level: str = os.getenv("LOG_LEVEL", "INFO")
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: Optional[str] = os.getenv("LOG_FILE")
    structured_logging: bool = True


@dataclass
class Config:
    """Configuração principal consolidada."""
    environment: Environment = Environment(
        os.getenv("ENVIRONMENT", "development")
    )
    debug: bool = environment == Environment.DEVELOPMENT
    
    # Configurações de subsistemas
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    neo4j: Neo4jConfig = field(default_factory=Neo4jConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    api: APIConfig = field(default_factory=APIConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # Caminhos
    project_root: Path = Path(__file__).parent.parent.parent
    data_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent / "data")
    models_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent / "models")
    logs_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent / "logs")
    
    # Segurança
    api_key: str = os.getenv("API_KEY", "dev-key-change-in-production")
    jwt_secret: str = os.getenv("JWT_SECRET", "dev-secret-change-in-production")
    
    def __post_init__(self):
        """Validar e preparar configuração após inicialização."""
        # Criar diretórios necessários
        for directory in [self.data_dir, self.models_dir, self.logs_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Ajustar API em modo debug
        if self.debug:
            self.api.debug = True
            self.database.echo = True
        
        logger.info(f"Configuração carregada para ambiente: {self.environment}")


# Instância global (singleton pattern)
_config: Optional[Config] = None


def get_config() -> Config:
    """
    Obter configuração global.
    Implementa lazy loading e singleton pattern.
    """
    global _config
    if _config is None:
        _config = Config()
    return _config


def reset_config() -> None:
    """Resetar configuração (útil para testes)."""
    global _config
    _config = None


# Exportar constantes de configuração padrão
DEFAULT_CONFIG = get_config()