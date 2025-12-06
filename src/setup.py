"""
setup.py - Configuração de instalação do pacote.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Ler README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text() if readme_path.exists() else ""

# Ler requirements
requirements_path = Path(__file__).parent / "requirements.txt"
install_requires = []
if requirements_path.exists():
    install_requires = [
        line.strip() 
        for line in requirements_path.read_text().split('\n')
        if line.strip() and not line.startswith('#')
    ]

setup(
    name="precision-psychiatry",
    version="1.0.0",
    description="Plataforma de IA/ML para Medicina de Precisão em Psiquiatria",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Data Science & AI Lab",
    author_email="info@precisionpsychiatry.com",
    url="https://github.com/seu-usuario/precision-psychiatry",
    license="MIT",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=install_requires,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=23.0",
            "flake8>=6.0",
            "mypy>=1.0",
        ],
        "docs": [
            "sphinx>=7.0",
            "sphinx-rtd-theme>=1.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "precision-psychiatry-train=scripts.train:main",
            "precision-psychiatry-infer=scripts.inference:main",
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Medical Science Apps",
    ],
    keywords=[
        "psychiatry",
        "precision-medicine",
        "machine-learning",
        "deep-learning",
        "genomics",
        "biomarkers"
    ]
)

---

# .env.example - Variáveis de Ambiente de Exemplo

# ========== Environment ==========
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=INFO

# ========== Database PostgreSQL ==========
DB_HOST=localhost
DB_PORT=5432
DB_USER=postgres
DB_PASSWORD=postgres
DB_NAME=precision_psychiatry

# ========== Graph Database Neo4j ==========
NEO4J_HOST=localhost
NEO4J_PORT=7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
NEO4J_DB=neo4j

# ========== Redis Cache ==========
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=
REDIS_SSL=false

# ========== API Configuration ==========
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
CORS_ORIGINS=http://localhost:3000,http://localhost:8080
API_KEY=dev-key-change-in-production
JWT_SECRET=dev-secret-change-in-production

# ========== Logging ==========
LOG_FILE=logs/app.log
STRUCTURED_LOGGING=true

# ========== Security ==========
ALLOWED_HOSTS=localhost,127.0.0.1

# ========== pgAdmin ==========
PGADMIN_EMAIL=admin@example.com
PGADMIN_PASSWORD=admin

# ========== Grafana ==========
GRAFANA_PASSWORD=admin

# ========== Data Generation ==========
SYNTHETIC_DATA_N_SAMPLES=500
SYNTHETIC_DATA_SEED=42

# ========== Model Configuration ==========
MODEL_TYPE=xgboost
MODEL_PATH=models/v1
RANDOM_STATE=42

# ========== Feature Configuration ==========
TEST_SIZE=0.2
VALIDATION_SIZE=0.1
CV_FOLDS=5

# ========== Validation Thresholds ==========
MIN_AUC=0.70
MIN_SENSITIVITY=0.60
MIN_SPECIFICITY=0.60
MAX_DEMOGRAPHIC_BIAS=0.05