"""
Constantes e enumerações para Medicina de Precisão em Psiquiatria.
"""

from enum import Enum
from typing import Dict, List, Tuple
from dataclasses import dataclass


class MedicineClass(str, Enum):
    """Classes de medicamentos psiquiátricos."""
    SSRI = "SSRI"  # Inibidor Seletivo de Recaptação de Serotonina
    SNRI = "SNRI"  # Inibidor de Recaptação de Serotonina-Noradrenalina
    TRICYCLIC = "TRICYCLIC"  # Tricíclicos
    ATYPICAL = "ATYPICAL"  # Atípicos
    STIMULANT = "STIMULANT"  # Estimulantes
    MAOI = "MAOI"  # Inibidor da Monoamina Oxidase


class MetabolizerStatus(str, Enum):
    """Status de metabolizador de fármacos."""
    POOR = "poor"  # Metabolizador lento
    INTERMEDIATE = "intermediate"  # Metabolizador intermediário
    EXTENSIVE = "extensive"  # Metabolizador normal
    ULTRA = "ultra"  # Metabolizador ultra-rápido


class RiskCategory(str, Enum):
    """Categorias de risco de resposta terapêutica."""
    LOW = "low"  # Baixo risco (alta probabilidade de resposta)
    MODERATE = "moderate"  # Risco moderado
    HIGH = "high"  # Alto risco (baixa probabilidade de resposta)


class ResponseStatus(str, Enum):
    """Status de resposta ao tratamento."""
    RESPONDER = "responder"
    NON_RESPONDER = "non_responder"
    PARTIAL_RESPONDER = "partial_responder"
    REFRACTORY = "refractory"


class GeneticVariant(str, Enum):
    """Variantes genéticas relevantes."""
    # Genes do sistema CYP450
    CYP2D6 = "CYP2D6"
    CYP3A4 = "CYP3A4"
    CYP1A2 = "CYP1A2"
    CYP2C19 = "CYP2C19"
    CYP2B6 = "CYP2B6"
    
    # Outros genes relevantes
    MTHFR = "MTHFR"
    COMT = "COMT"
    BDNF = "BDNF"
    HTR1A = "5HTR1A"
    TPH1 = "TPH1"
    SLC6A4 = "SLC6A4"


class BiomarkerType(str, Enum):
    """Tipos de biomarcadores."""
    INFLAMMATORY = "inflammatory"
    NEUROBIOLOGICAL = "neurobiological"
    HORMONAL = "hormonal"
    METABOLIC = "metabolic"


class PsychometricScale(str, Enum):
    """Escalas psicométricas padronizadas."""
    PHQ9 = "phq9"  # Patient Health Questionnaire
    GAD7 = "gad7"  # Generalized Anxiety Disorder Scale
    PANSS_POS = "panss_positive"  # PANSS Positivo
    PANSS_NEG = "panss_negative"  # PANSS Negativo
    CTQ = "ctq_total"  # Childhood Trauma Questionnaire
    ACE = "ace_total"  # Adverse Childhood Experiences
    PSQI = "psqi"  # Pittsburgh Sleep Quality Index


# ==================== Valores Padrão ====================

# Genes mais relevantes para resposta antidepressiva
CORE_GENES: List[str] = [
    "CYP2D6", "CYP3A4", "CYP1A2", "CYP2C19", "CYP2B6",
    "MTHFR", "COMT", "BDNF", "5HTR1A", "TPH1", "SLC6A4"
]

# Biomarcadores neurobiológicos padrão
NEUROBIOLOGICAL_MARKERS: List[str] = [
    "il6_pg_ml", "tnf_alpha_pg_ml", "crp_mg_l",
    "cortisol_morning_nmol_l", "cortisol_evening_nmol_l", "acth_pm_mlU_l",
    "serotonin_ng_ml", "dopamine_pg_ml", "noradrenaline_pg_ml",
    "bdnf_ng_ml", "kynurenine_nmol_l", "tryptophan_nmol_l"
]

# Escalas psicométricas padrão
PSYCHOMETRIC_SCALES: List[str] = [
    "phq9_score", "gad7_score",
    "panss_positive", "panss_negative",
    "ctq_total", "ace_total", "psqi_score",
    "social_support_score", "life_stressors"
]

# Variáveis demográficas
DEMOGRAPHIC_VARS: List[str] = [
    "age", "gender", "bmi", "education_years"
]

# Medicamentos antidepressivos comuns
MEDICATIONS: Dict[str, Dict] = {
    "Sertraline": {
        "class": MedicineClass.SSRI,
        "metabolism": ["CYP2D6", "CYP3A4"],
        "usual_dose": "50-200mg",
        "half_life_hours": 26
    },
    "Fluoxetine": {
        "class": MedicineClass.SSRI,
        "metabolism": ["CYP2D6", "CYP2C9"],
        "usual_dose": "20-80mg",
        "half_life_hours": 240
    },
    "Paroxetine": {
        "class": MedicineClass.SSRI,
        "metabolism": ["CYP2D6"],
        "usual_dose": "20-60mg",
        "half_life_hours": 21
    },
    "Citalopram": {
        "class": MedicineClass.SSRI,
        "metabolism": ["CYP3A4", "CYP2C19"],
        "usual_dose": "20-40mg",
        "half_life_hours": 35
    },
    "Venlafaxine": {
        "class": MedicineClass.SNRI,
        "metabolism": ["CYP2D6"],
        "usual_dose": "75-375mg",
        "half_life_hours": 11
    },
    "Duloxetine": {
        "class": MedicineClass.SNRI,
        "metabolism": ["CYP1A2", "CYP2D6"],
        "usual_dose": "60-120mg",
        "half_life_hours": 12
    },
    "Bupropion": {
        "class": MedicineClass.ATYPICAL,
        "metabolism": ["CYP2D6"],
        "usual_dose": "300-450mg",
        "half_life_hours": 21
    },
    "Mirtazapine": {
        "class": MedicineClass.ATYPICAL,
        "metabolism": ["CYP1A2", "CYP2D6", "CYP3A4"],
        "usual_dose": "15-45mg",
        "half_life_hours": 20
    },
    "Amitriptyline": {
        "class": MedicineClass.TRICYCLIC,
        "metabolism": ["CYP2D6", "CYP1A2"],
        "usual_dose": "75-300mg",
        "half_life_hours": 25
    },
    "Nortriptyline": {
        "class": MedicineClass.TRICYCLIC,
        "metabolism": ["CYP2D6"],
        "usual_dose": "75-150mg",
        "half_life_hours": 30
    }
}

# Range de referência para biomarcadores
BIOMARKER_RANGES: Dict[str, Tuple[float, float]] = {
    "il6_pg_ml": (1.0, 5.0),
    "tnf_alpha_pg_ml": (2.0, 10.0),
    "crp_mg_l": (0.3, 3.0),
    "cortisol_morning_nmol_l": (200.0, 400.0),
    "cortisol_evening_nmol_l": (100.0, 200.0),
    "serotonin_ng_ml": (40.0, 80.0),
    "dopamine_pg_ml": (20.0, 40.0),
    "noradrenaline_pg_ml": (150.0, 250.0),
    "bdnf_ng_ml": (50.0, 150.0)
}

# Ranges de escalas psicométricas
SCALE_RANGES: Dict[str, Tuple[int, int]] = {
    "phq9_score": (0, 27),
    "gad7_score": (0, 21),
    "panss_positive": (7, 49),
    "panss_negative": (7, 49),
    "ctq_total": (28, 140),
    "ace_total": (0, 10),
    "psqi_score": (0, 21),
    "social_support_score": (0, 40)
}

# Interpretação de severidade PHQ-9
PHQ9_SEVERITY: Dict[str, Tuple[int, int]] = {
    "minimal": (0, 4),
    "mild": (5, 9),
    "moderate": (10, 14),
    "moderately_severe": (15, 19),
    "severe": (20, 27)
}

# Interpretação de severidade GAD-7
GAD7_SEVERITY: Dict[str, Tuple[int, int]] = {
    "minimal": (0, 4),
    "mild": (5, 9),
    "moderate": (10, 14),
    "severe": (15, 21)
}

# Metabolismo de medicamentos por CYP2D6 status
METABOLIZER_MEDICATION_MAP: Dict[str, Dict[str, List[str]]] = {
    "poor": {
        "avoid": ["Fluoxetine", "Paroxetine", "Venlafaxine"],
        "caution": ["Sertraline", "Citalopram"],
        "safe": ["Citalopram", "Escitalopram", "Mirtazapine"]
    },
    "intermediate": {
        "avoid": [],
        "caution": ["Fluoxetine", "Paroxetine"],
        "safe": ["Sertraline", "Citalopram", "Venlafaxine"]
    },
    "extensive": {
        "avoid": [],
        "caution": [],
        "safe": ["Sertraline", "Fluoxetine", "Paroxetine", "Citalopram"]
    },
    "ultra": {
        "avoid": [],
        "caution": ["Sertraline", "Paroxetine"],
        "safe": ["Fluoxetine", "Citalopram"]
    }
}

# Limites clínicos de validação
CLINICAL_THRESHOLDS: Dict[str, float] = {
    "min_auc_roc": 0.70,
    "min_sensitivity": 0.60,
    "min_specificity": 0.60,
    "min_npv": 0.70,
    "min_ppv": 0.70,
    "max_demographic_bias": 0.05,  # 5%
    "max_calibration_error": 0.05
}

# Limites de depressão refratária
REFRACTORY_CRITERIA: Dict[str, int] = {
    "min_previous_treatments": 2,
    "min_phq9_score": 15,
    "min_treatment_weeks": 4
}

# Fatores de risco para não-resposta
RISK_FACTORS: List[str] = [
    "phq9_score",  # Severidade baseline
    "ctq_total",  # Trauma prévio
    "num_previous_meds",  # Número de tentativas prévias
    "il6_pg_ml",  # Inflamação elevada
    "cortisol_morning_nmol_l",  # Disfunção HPA
    "bdnf_ng_ml",  # BDNF baixo
    "age",  # Idade
    "life_stressors"  # Estressores de vida
]

# Protetor de resposta
PROTECTIVE_FACTORS: List[str] = [
    "social_support_score",  # Suporte social
    "bdnf_ng_ml",  # BDNF alto
    "employment_status",  # Emprego ativo
    "education_years"  # Anos de educação
]

# Configurações padrão de modelo
MODEL_DEFAULTS: Dict[str, int] = {
    "random_state": 42,
    "test_size": 0.2,
    "validation_size": 0.1,
    "cv_folds": 5,
    "min_samples_leaf": 2,
    "min_samples_split": 5
}

# Hiperparâmetros padrão por modelo
HYPERPARAMETERS: Dict[str, Dict] = {
    "xgboost": {
        "max_depth": 7,
        "learning_rate": 0.05,
        "n_estimators": 200,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1,
        "gamma": 0,
        "reg_alpha": 0,
        "reg_lambda": 1
    },
    "random_forest": {
        "n_estimators": 100,
        "max_depth": 15,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "bootstrap": True
    },
    "svm": {
        "kernel": "rbf",
        "C": 1.0,
        "gamma": "scale",
        "probability": True
    }
}

# Mapeamento de genótipos para interpretação
GENOTYPE_INTERPRETATION: Dict[int, str] = {
    0: "Homozigoto para alelo de referência",
    1: "Heterozigoto",
    2: "Homozigoto para alelo alternativo"
}

# Mensagens padrão de erro
ERROR_MESSAGES: Dict[str, str] = {
    "invalid_patient_id": "ID de paciente inválido",
    "missing_required_field": "Campo obrigatório faltando: {field}",
    "invalid_biomarker_range": "Valor de biomarcador fora do range esperado: {marker}",
    "model_not_loaded": "Modelo não foi carregado",
    "invalid_metabolizer_status": "Status de metabolizador inválido",
    "insufficient_data": "Dados insuficientes para predição"
}

# Mensagens de sucesso
SUCCESS_MESSAGES: Dict[str, str] = {
    "model_trained": "Modelo treinado com sucesso",
    "prediction_generated": "Predição gerada com sucesso",
    "data_validated": "Dados validados com sucesso"
}


@dataclass
class FeatureConfig:
    """Configuração de features."""
    genes: List[str] = None
    neurobiological: List[str] = None
    psychometric: List[str] = None
    demographic: List[str] = None
    
    def __post_init__(self):
        if self.genes is None:
            self.genes = CORE_GENES
        if self.neurobiological is None:
            self.neurobiological = NEUROBIOLOGICAL_MARKERS
        if self.psychometric is None:
            self.psychometric = PSYCHOMETRIC_SCALES
        if self.demographic is None:
            self.demographic = DEMOGRAPHIC_VARS
    
    @property
    def all_features(self) -> List[str]:
        """Obter todas as features."""
        features = []
        features.extend([f"{g}_genotype" for g in self.genes])
        features.extend(self.neurobiological)
        features.extend(self.psychometric)
        features.extend(self.demographic)
        return features
    
    @property
    def n_features(self) -> int:
        """Número total de features."""
        return len(self.all_features)