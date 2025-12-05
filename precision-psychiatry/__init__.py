"""
src/__init__.py - Inicializador do pacote principal
"""

from src.config import get_config
from src.constants import (
    CORE_GENES,
    NEUROBIOLOGICAL_MARKERS,
    PSYCHOMETRIC_SCALES,
    MetabolizerStatus,
    RiskCategory,
    ResponseStatus
)

__version__ = "1.0.0"
__author__ = "Data Science & AI Lab"
__email__ = "info@precisionpsychiatry.com"

__all__ = [
    'get_config',
    'CORE_GENES',
    'NEUROBIOLOGICAL_MARKERS',
    'PSYCHOMETRIC_SCALES',
    'MetabolizerStatus',
    'RiskCategory',
    'ResponseStatus'
]

---

"""
src/data/__init__.py
"""

from src.data.loaders import (
    load_data,
    save_data,
    CSVDataLoader,
    ParquetDataLoader,
    DatasetCombiner
)
from src.data.validators import (
    validate_data,
    DataValidator,
    ComprehensiveValidator
)
from src.data.preprocessors import (
    preprocess_data,
    PreprocessingPipeline,
    build_default_pipeline
)

__all__ = [
    'load_data',
    'save_data',
    'CSVDataLoader',
    'ParquetDataLoader',
    'DatasetCombiner',
    'validate_data',
    'DataValidator',
    'ComprehensiveValidator',
    'preprocess_data',
    'PreprocessingPipeline',
    'build_default_pipeline'
]

---

"""
src/features/__init__.py
"""

from src.features.genomic import (
    GenomicFeatureExtractor,
    GenomicInterpreter,
    extract_genomic_features
)
from src.features.neurobiological import (
    NeurobiologicalFeatureExtractor,
    BiomarkerInterpreter,
    extract_neurobiological_features
)
from src.features.psychosocial import (
    PsychosocialFeatureExtractor,
    ClinicalSeverityAssessor,
    extract_psychosocial_features
)

__all__ = [
    'GenomicFeatureExtractor',
    'GenomicInterpreter',
    'extract_genomic_features',
    'NeurobiologicalFeatureExtractor',
    'BiomarkerInterpreter',
    'extract_neurobiological_features',
    'PsychosocialFeatureExtractor',
    'ClinicalSeverityAssessor',
    'extract_psychosocial_features'
]

---

"""
src/utils/__init__.py
"""

from src.utils.logger import StructuredLogger
from src.utils.decorators import timer, log_calls, cache_result
from src.utils.helpers import (
    ensure_dataframe,
    ensure_numpy,
    get_memory_usage,
    flatten_dict,
    chunk_list,
    normalize_range,
    detect_outliers,
    compare_distributions
)

__all__ = [
    'StructuredLogger',
    'timer',
    'log_calls',
    'cache_result',
    'ensure_dataframe',
    'ensure_numpy',
    'get_memory_usage',
    'flatten_dict',
    'chunk_list',
    'normalize_range',
    'detect_outliers',
    'compare_distributions'
]

---

"""
src/visualization/__init__.py
"""

from src.visualization.plotters import (
    plot_roc_curve,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_prediction_distribution
)

__all__ = [
    'plot_roc_curve',
    'plot_confusion_matrix',
    'plot_feature_importance',
    'plot_prediction_distribution'
]