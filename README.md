# Precision Psychiatry
## Sistema de Personalização de Tratamento Psiquiatrico

**Precision Psychiatry** é um sistema protótipo de pesquisa de teste de IA/ML que integra dados genéticos, neurobiológicos e psicossociais para:

- **Predição de Resposta Terapêutica**: Prever resposta a medicamentos antidepressivos
- **Análise Genômica**: Integração de dados de polimorfismos genéticos relevantes
- **Marcadores Neurobiológicos**: Processamento de dados de neuroimagem, EEG, biomarcadores
- **Fatores Psicossociais**: Avaliação de estressores ambientais e dados demográficos
- **Recomendação de Medicamentos**: Sugestões personalizadas para pacientes com depressão refratária
- **Monitoramento Longitudinal**: Acompanhamento de resposta ao tratamento ao longo do tempo

## Stack Tecnológico

### Data & ML
- **Pandas, NumPy**: Processamento de dados
- **Scikit-learn**: Modelos clássicos de ML
- **XGBoost, LightGBM**: Gradient Boosting
- **TensorFlow/Keras**: Deep Learning
- **SHAP**: Explicabilidade de modelos

### Database
- **PostgreSQL**: Dados estruturados
- **Neo4j**: Grafo de relacionamentos genômicos
- **Redis**: Cache e filas

### API & Backend
- **FastAPI**: API REST moderna
- **Pydantic**: Validação de dados
- **SQLAlchemy**: ORM

### DevOps
- **Docker/Docker-compose**: Containerização
- **pytest**: Testes automatizados
- **MLflow/Weights & Biases**: Rastreamento de experimentos

## Features Principais

### 1. **Análise Genômica Avançada**
- Processamento de SNPs (Single Nucleotide Polymorphisms)
- Análise de polimorfismos relevantes (CYP2D6, CYP3A4, MTHFR, BDNF)
- Cálculo de risco genético poligênico
- Anotação funcional de variantes

### 2. **Integração de Biomarcadores Neurobiológicos**
- Cortisol, ACTH, citocinas (IL-6, TNF-α)
- Neurotransmissores (serotonina, dopamina, noradrenalina)
- Processamento de neuroimagem (fMRI, estrutural)
- Métricas de EEG (potência espectral, coerência)

### 3. **Avaliação Psicossocial Estruturada**
- Escalas clínicas (PHQ-9, GAD-7, PANSS)
- Histórico de trauma (CTQ, ACE)
- Suporte social e estressores ambientais
- Comorbidades psiquiátricas

### 4. **Modelos Preditivos Híbridos**
- Ensemble de modelos (RF, XGBoost, SVM, redes neurais)
- Validação cruzada estratificada
- Calibração de probabilidades
- Análise de incerteza

### 5. **Explicabilidade e Interpretabilidade**
- SHAP values para feature importance
- LIME para explicações locais
- Permutation importance
- Partial dependence plots

### 6. **Sistema de Recomendação**
- Scores de eficácia por medicamento
- Previsão de efeitos adversos
- Dosagem personalizada
- ContraIndicações baseadas em genótipo

## Dados Sintéticos

O projeto inclui gerador de dados sintéticos realistas para:
- 500+ pacientes
- 50+ features genômicas
- 20+ biomarcadores
- Respostas terapêuticas baseadas em regras clínicas
- Distribuições que refletem características reais

## Instalação Rápida

```bash
# Clone o repositório
git clone https://github.com/mathadriele/precisionPsychiatry-sistema-personalizacao-tratamento-psiquiatrico.git
cd precision-psychiatry

# Crie ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou: venv\Scripts\activate  # Windows

# Instale dependências
pip install -r requirements.txt

# Configure variáveis de ambiente
cp .env.example .env

# Execute testes
pytest tests/ -v

# Inicie a API
python -m uvicorn src.api.main:app --reload
```

## Uso

### Treinamento de Modelo
```bash
python scripts/train.py --config config/default.yaml --output models/v1
```

### Inferência para Paciente Individual
```bash
python scripts/inference.py \
  --model-path models/v1/model.pkl \
  --patient-data data/sample_patient.json \
  --output results/recommendation.json
```

### API REST
```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d @data/patient.json
```

## Validação Clínica

O projeto implementa rigorosa validação clínica:
- Métricas de performance (AUC, sensitivity, specificity)
- Análise de concordância com diagnósticos clínicos
- Validação em coortes independentes
- Análise de bias por demografia
- Conformidade com GDPR/HIPAA

## Contato e Suporte

Para dúvidas ou sugestões, abra uma issue no repositório ou entre em contato.
