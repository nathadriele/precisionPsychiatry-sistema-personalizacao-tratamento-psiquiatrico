"""
Script de inferência para predições em tempo real.
Carrega modelo treinado e gera predições para novos pacientes.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import numpy as np

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_config
from src.ml_pipeline.pipeline import PrecisionPsychiatryPipeline
from src.data.validators import ComprehensiveValidator, ValidationResult
from src.features.genomic import GenomicInterpreter
from src.features.neurobiological import BiomarkerInterpreter
from src.features.psychosocial import ClinicalSeverityAssessor
from src.constants import RiskCategory

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class InferenceEngine:
    """Engine de inferência completa."""
    
    def __init__(self, model_path: Path):
        """
        Inicializar engine.
        
        Args:
            model_path: Caminho do modelo treinado
        """
        self.config = get_config()
        self.model_path = Path(model_path)
        self.pipeline = None
        self.validator = ComprehensiveValidator(strict=False)
        
        self._load_model()
    
    def _load_model(self):
        """Carregar modelo treinado."""
        logger.info(f"Carregando modelo de: {self.model_path}")
        
        try:
            self.pipeline = PrecisionPsychiatryPipeline(
                model_type="xgboost"
            )
            self.pipeline.load(self.model_path)
            logger.info("✓ Modelo carregado com sucesso")
        except Exception as e:
            logger.error(f"✗ Erro ao carregar modelo: {e}")
            raise
    
    def predict_patient(
        self,
        patient_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Fazer predição para um paciente.
        
        Args:
            patient_data: Dados completos do paciente
        
        Returns:
            Dicionário com predição e explicações
        """
        logger.info(f"Processando paciente: {patient_data.get('patient_id')}")
        
        # 1. Validar dados
        validation_result = self.validator.validate_patient_complete(patient_data)
        
        if not validation_result.is_valid:
            logger.warning(f"Validação com problemas:")
            for error in validation_result.errors:
                logger.warning(f"  - {error}")
            return {
                "error": "Validação de dados falhou",
                "details": validation_result.errors
            }
        
        # 2. Preparar features
        X = self._prepare_features(patient_data)
        
        # 3. Fazer predição
        try:
            X_processed, _ = self.pipeline.preprocess_data(X, fit=False)
            prediction_prob = self.pipeline.model.predict_proba(X_processed)[0, 1]
            
            # Classificar risco
            risk_category = self._classify_risk(prediction_prob)
            
            # Obter explicações
            explanation = self.pipeline.explain_prediction(X_processed, top_n=5)
            
            # Recomendações clínicas
            recommendations = self._get_clinical_recommendations(
                patient_data,
                prediction_prob
            )
            
            # Contra-indicações
            contraindications = self._get_contraindications(patient_data)
            
            result = {
                "patient_id": patient_data.get("patient_id"),
                "prediction": float(prediction_prob),
                "risk_category": risk_category,
                "confidence": float(max(prediction_prob, 1 - prediction_prob)),
                "recommended_medications": recommendations["medications"],
                "treatment_considerations": recommendations["considerations"],
                "contraindications": contraindications,
                "explanation": explanation,
                "validation": {
                    "is_valid": validation_result.is_valid,
                    "warnings": validation_result.warnings
                }
            }
            
            logger.info(f"✓ Predição concluída: {prediction_prob:.2%} de resposta")
            
            return result
            
        except Exception as e:
            logger.error(f"✗ Erro na predição: {e}")
            return {
                "error": str(e),
                "patient_id": patient_data.get("patient_id")
            }
    
    def _prepare_features(self, patient_data: Dict) -> pd.DataFrame:
        """Preparar features para predição."""
        # Construir DataFrame com dados do paciente
        features = {
            "age": patient_data["psychosocial_data"]["age"],
            "gender": patient_data["psychosocial_data"]["gender"],
            "bmi": patient_data["psychosocial_data"]["bmi"],
            "education_years": patient_data["psychosocial_data"]["education_years"],
        }
        
        # Adicionar genômicos
        genomic = patient_data.get("genomic_data", {})
        for key, value in genomic.items():
            features[key] = value
        
        # Adicionar biomarcadores
        neurobiological = patient_data.get("neurobiological_data", {})
        for key, value in neurobiological.items():
            features[key] = value
        
        # Adicionar psicossociais
        psychosocial = patient_data.get("psychosocial_data", {})
        for key in ["phq9_score", "gad7_score", "ctq_total", "ace_total", 
                    "psqi_score", "social_support_score"]:
            if key in psychosocial:
                features[key] = psychosocial[key]
        
        return pd.DataFrame([features])
    
    def _classify_risk(self, prediction_prob: float) -> str:
        """Classificar categoria de risco."""
        thresholds = self.config.features.risk_stratification
        
        if prediction_prob < thresholds["moderate"][0]:
            return RiskCategory.LOW.value
        elif prediction_prob < thresholds["high"][0]:
            return RiskCategory.MODERATE.value
        else:
            return RiskCategory.HIGH.value
    
    def _get_clinical_recommendations(
        self,
        patient_data: Dict,
        prediction_prob: float
    ) -> Dict:
        """Obter recomendações clínicas."""
        recommendations = {
            "medications": [],
            "considerations": []
        }
        
        genomic = patient_data.get("genomic_data", {})
        neurobiological = patient_data.get("neurobiological_data", {})
        psychosocial = patient_data.get("psychosocial_data", {})
        
        # Recomendações de medicamentos
        med_recs = GenomicInterpreter.get_medication_recommendations(
            genomic,
            genomic.get("metabolizer_status", "extensive")
        )
        
        recommendations["medications"] = med_recs["recommended"][:3]  # Top 3
        
        # Considerações clínicas
        if neurobiological.get("il6_pg_ml", 0) > 5:
            recommendations["considerations"].append(
                "Paciente com inflamação elevada: considerar adjuvantes anti-inflamatórios"
            )
        
        if psychosocial.get("ctq_total", 0) > 100:
            recommendations["considerations"].append(
                "História significativa de trauma: psicoterpia concomitante recomendada"
            )
        
        if prediction_prob > 0.7:
            recommendations["considerations"].append(
                "Boa probabilidade de resposta: monitorar por 4-6 semanas"
            )
        else:
            recommendations["considerations"].append(
                "Resposta incerta: considerar combinação de medicações ou alternativas"
            )
        
        return recommendations
    
    def _get_contraindications(self, patient_data: Dict) -> list:
        """Obter contra-indicações."""
        contraindications = []
        
        genomic = patient_data.get("genomic_data", {})
        
        # Baseado em genótipo
        contra = GenomicInterpreter.get_contraindications(genomic)
        contraindications.extend(contra)
        
        # Baseado em biomarcadores
        neurobiological = patient_data.get("neurobiological_data", {})
        
        if neurobiological.get("cortisol_morning_nmol_l", 0) > 500:
            contraindications.append(
                "HPA axis muito ativo: considerar estabilização antes de medicação"
            )
        
        return contraindications
    
    def predict_batch(
        self,
        patient_list: list
    ) -> list:
        """
        Fazer predições para múltiplos pacientes.
        
        Args:
            patient_list: Lista de dicionários com dados de pacientes
        
        Returns:
            Lista de resultados
        """
        logger.info(f"Processando {len(patient_list)} pacientes em batch...")
        
        results = []
        for patient in patient_list:
            result = self.predict_patient(patient)
            results.append(result)
        
        return results


def load_patient_from_json(filepath: Path) -> Dict:
    """Carregar dados de paciente de arquivo JSON."""
    with open(filepath, 'r') as f:
        return json.load(f)


def save_results(results: Dict, output_path: Path):
    """Salvar resultados de predição."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"✓ Resultados salvos em: {output_path}")


def main():
    """Função principal."""
    parser = argparse.ArgumentParser(
        description="Script de inferência para Medicina de Precisão em Psiquiatria"
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Caminho do modelo treinado"
    )
    
    parser.add_argument(
        "--patient-data",
        type=str,
        required=True,
        help="Caminho do arquivo JSON com dados do paciente"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="Caminho de saída para resultados (opcional)"
    )
    
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Processar como batch (arquivo JSON com lista de pacientes)"
    )
    
    args = parser.parse_args()
    
    try:
        # Carregar modelo
        engine = InferenceEngine(Path(args.model_path))
        
        # Carregar dados do paciente
        logger.info(f"Carregando dados de: {args.patient_data}")
        patient_data = load_patient_from_json(Path(args.patient_data))
        
        # Fazer predição
        if args.batch:
            results = engine.predict_batch(patient_data)
        else:
            results = engine.predict_patient(patient_data)
        
        # Salvar resultados
        if args.output:
            save_results(results, Path(args.output))
        else:
            # Imprimir na tela
            print("\n" + "=" * 60)
            print("RESULTADOS DA PREDIÇÃO")
            print("=" * 60)
            print(json.dumps(results, indent=2, default=str))
        
    except Exception as e:
        logger.error(f"Erro na execução: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()