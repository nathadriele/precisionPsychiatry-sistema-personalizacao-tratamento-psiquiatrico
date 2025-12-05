"""
Engenharia de features psicossociais para resposta a antidepressivos.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

from src.constants import PHQ9_SEVERITY, GAD7_SEVERITY

logger = logging.getLogger(__name__)


class PsychosocialFeatureExtractor:
    """Extrator de features psicossociais."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
    
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extrair features psicossociais.
        
        Args:
            df: DataFrame com dados psicossociais
        
        Returns:
            DataFrame com features derivadas
        """
        self.log("Extraindo features psicossociais...")
        
        features = pd.DataFrame(index=df.index)
        
        # 1. Features de severidade depressiva (PHQ-9)
        features = self._add_depressive_severity_features(features, df)
        
        # 2. Features de severidade de ansiedade (GAD-7)
        features = self._add_anxiety_severity_features(features, df)
        
        # 3. Features de trauma (CTQ)
        features = self._add_trauma_features(features, df)
        
        # 4. Features de adversidade na infância (ACE)
        features = self._add_adversity_features(features, df)
        
        # 5. Features de qualidade de sono (PSQI)
        features = self._add_sleep_features(features, df)
        
        # 6. Features de suporte social
        features = self._add_social_support_features(features, df)
        
        # 7. Features demográficas
        features = self._add_demographic_features(features, df)
        
        # 8. Features compostas de risco
        features = self._add_composite_risk_features(features, df)
        
        self.log(f"✓ Features psicossociais extraídas: {features.shape[1]} features")
        
        return features
    
    def _add_depressive_severity_features(
        self,
        features: pd.DataFrame,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """Adicionar features de severidade depressiva."""
        
        if "phq9_score" not in df.columns:
            return features
        
        phq9 = df["phq9_score"]
        
        # Categorias de severidade
        for severity, (min_val, max_val) in PHQ9_SEVERITY.items():
            features[f"phq9_{severity}"] = (
                (phq9 >= min_val) & (phq9 <= max_val)
            ).astype(int)
        
        # Normalizar PHQ-9
        features["phq9_normalized"] = phq9 / 27
        
        # PHQ-9 muito elevado (muito deprimido)
        features["phq9_very_severe"] = (phq9 >= 20).astype(int)
        
        # PHQ-9 leve (possível caso de remissão)
        features["phq9_remission_possible"] = (phq9 <= 4).astype(int)
        
        # Mudança esperada para remissão (50% redução)
        features["phq9_threshold_remission"] = 5  # Threshold para remissão
        
        return features
    
    def _add_anxiety_severity_features(
        self,
        features: pd.DataFrame,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """Adicionar features de severidade de ansiedade."""
        
        if "gad7_score" not in df.columns:
            return features
        
        gad7 = df["gad7_score"]
        
        # Categorias de severidade
        for severity, (min_val, max_val) in GAD7_SEVERITY.items():
            features[f"gad7_{severity}"] = (
                (gad7 >= min_val) & (gad7 <= max_val)
            ).astype(int)
        
        # Normalizar GAD-7
        features["gad7_normalized"] = gad7 / 21
        
        # GAD-7 severo
        features["gad7_severe"] = (gad7 >= 15).astype(int)
        
        # Comorbidade ansiedade-depressão
        if "phq9_score" in df.columns:
            phq9_normalized = df["phq9_score"] / 27
            gad7_normalized = gad7 / 21
            features["anxiety_depression_comorbidity"] = (phq9_normalized + gad7_normalized) / 2
        
        return features
    
    def _add_trauma_features(
        self,
        features: pd.DataFrame,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """Adicionar features de trauma (CTQ)."""
        
        if "ctq_total" not in df.columns:
            return features
        
        ctq = df["ctq_total"]
        
        # CTQ normalizado
        features["ctq_normalized"] = (ctq - 28) / (140 - 28)
        
        # Trauma severo
        features["trauma_severe"] = (ctq >= 100).astype(int)
        features["trauma_moderate"] = ((ctq >= 72) & (ctq < 100)).astype(int)
        features["trauma_mild"] = ((ctq >= 41) & (ctq < 72)).astype(int)
        features["trauma_minimal"] = (ctq <= 40).astype(int)
        
        # Fator de risco para não-resposta (trauma associado com pior prognóstico)
        features["trauma_risk_factor"] = (ctq >= 72).astype(int)
        
        return features
    
    def _add_adversity_features(
        self,
        features: pd.DataFrame,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """Adicionar features de adversidade (ACE)."""
        
        if "ace_total" not in df.columns:
            return features
        
        ace = df["ace_total"]
        
        # ACE normalizado
        features["ace_normalized"] = ace / 10
        
        # Categorias ACE
        features["ace_none"] = (ace == 0).astype(int)
        features["ace_moderate"] = ((ace >= 1) & (ace <= 3)).astype(int)
        features["ace_high"] = (ace >= 4).astype(int)
        
        # ACE como fator de risco
        features["ace_risk_factor"] = (ace >= 4).astype(int)
        
        return features
    
    def _add_sleep_features(
        self,
        features: pd.DataFrame,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """Adicionar features de qualidade de sono (PSQI)."""
        
        if "psqi_score" not in df.columns:
            return features
        
        psqi = df["psqi_score"]
        
        # PSQI normalizado
        features["psqi_normalized"] = psqi / 21
        
        # Qualidade de sono ruins (PSQI > 5 é cutoff clínico)
        features["sleep_poor"] = (psqi > 5).astype(int)
        features["sleep_very_poor"] = (psqi > 10).astype(int)
        
        # Sono como fator de risco (insônia associada com pior prognóstico)
        features["sleep_risk_factor"] = (psqi > 5).astype(int)
        
        return features
    
    def _add_social_support_features(
        self,
        features: pd.DataFrame,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """Adicionar features de suporte social."""
        
        if "social_support_score" not in df.columns:
            return features
        
        social_support = df["social_support_score"]
        
        # Normalizar suporte social
        features["social_support_normalized"] = social_support / 40
        
        # Categorias de suporte
        features["social_support_low"] = (social_support < 20).astype(int)
        features["social_support_moderate"] = ((social_support >= 20) & (social_support < 30)).astype(int)
        features["social_support_high"] = (social_support >= 30).astype(int)
        
        # Suporte social como fator protetor
        features["social_support_protective"] = (social_support >= 30).astype(int)
        
        return features
    
    def _add_demographic_features(
        self,
        features: pd.DataFrame,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """Adicionar features demográficas."""
        
        # Idade
        if "age" in df.columns:
            age = df["age"]
            features["age"] = age
            features["age_normalized"] = (age - age.min()) / (age.max() - age.min())
            
            # Categorias de idade
            features["age_young"] = (age < 35).astype(int)
            features["age_middle"] = ((age >= 35) & (age < 55)).astype(int)
            features["age_older"] = (age >= 55).astype(int)
        
        # Gênero
        if "gender" in df.columns:
            features["is_female"] = (df["gender"] == "F").astype(int)
            features["is_male"] = (df["gender"] == "M").astype(int)
        
        # IMC
        if "bmi" in df.columns:
            bmi = df["bmi"]
            features["bmi"] = bmi
            features["bmi_normalized"] = (bmi - bmi.min()) / (bmi.max() - bmi.min())
            
            # Categorias de IMC
            features["bmi_underweight"] = (bmi < 18.5).astype(int)
            features["bmi_normal"] = ((bmi >= 18.5) & (bmi < 25)).astype(int)
            features["bmi_overweight"] = ((bmi >= 25) & (bmi < 30)).astype(int)
            features["bmi_obese"] = (bmi >= 30).astype(int)
        
        # Educação
        if "education_years" in df.columns:
            education = df["education_years"]
            features["education_years"] = education
            features["education_normalized"] = (education - education.min()) / (education.max() - education.min())
        
        return features
    
    def _add_composite_risk_features(
        self,
        features: pd.DataFrame,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """Adicionar features compostas de risco."""
        
        # Complexidade clínica (comorbidades + trauma + adversidade)
        complexity_factors = []
        if "anxiety_depression_comorbidity" in features.columns:
            complexity_factors.append("anxiety_depression_comorbidity")
        if "trauma_risk_factor" in features.columns:
            complexity_factors.append("trauma_risk_factor")
        if "ace_risk_factor" in features.columns:
            complexity_factors.append("ace_risk_factor")
        if "sleep_risk_factor" in features.columns:
            complexity_factors.append("sleep_risk_factor")
        
        if complexity_factors:
            features["clinical_complexity_score"] = \
                features[complexity_factors].sum(axis=1) / len(complexity_factors)
        
        # Score de fatores de risco geral
        risk_factors = []
        if "trauma_risk_factor" in features.columns:
            risk_factors.append("trauma_risk_factor")
        if "ace_risk_factor" in features.columns:
            risk_factors.append("ace_risk_factor")
        if "sleep_risk_factor" in features.columns:
            risk_factors.append("sleep_risk_factor")
        if "phq9_very_severe" in features.columns:
            risk_factors.append("phq9_very_severe")
        if "gad7_severe" in features.columns:
            risk_factors.append("gad7_severe")
        
        if risk_factors:
            features["psychosocial_risk_score"] = \
                features[risk_factors].sum(axis=1) / len(risk_factors)
        
        # Score de fatores protetores
        protective_factors = []
        if "social_support_protective" in features.columns:
            protective_factors.append("social_support_protective")
        if "phq9_remission_possible" in features.columns:
            protective_factors.append("phq9_remission_possible")
        if "trauma_minimal" in features.columns:
            protective_factors.append("trauma_minimal")
        
        if protective_factors:
            features["psychosocial_protective_score"] = \
                features[protective_factors].sum(axis=1) / len(protective_factors)
        
        return features
    
    def log(self, message: str):
        """Log."""
        if self.verbose:
            logger.info(message)


class ClinicalSeverityAssessor:
    """Avaliador de severidade clínica."""
    
    @staticmethod
    def get_overall_severity(data: Dict) -> str:
        """
        Obter severidade clínica geral.
        
        Args:
            data: Dicionário com dados psicossociais
        
        Returns:
            Categoria de severidade
        """
        phq9 = data.get("phq9_score", 0)
        gad7 = data.get("gad7_score", 0)
        
        composite = (phq9 + gad7) / 2
        
        if composite < 5:
            return "minimal"
        elif composite < 10:
            return "mild"
        elif composite < 15:
            return "moderate"
        elif composite < 20:
            return "moderately_severe"
        else:
            return "severe"
    
    @staticmethod
    def get_risk_level(data: Dict) -> str:
        """
        Obter nível de risco de não-resposta.
        
        Args:
            data: Dicionário com dados psicossociais
        
        Returns:
            Nível de risco
        """
        risk_score = 0
        
        # Trauma
        if data.get("ctq_total", 0) >= 72:
            risk_score += 1
        
        # Adversidade
        if data.get("ace_total", 0) >= 4:
            risk_score += 1
        
        # Sono
        if data.get("psqi_score", 0) > 5:
            risk_score += 1
        
        # Severidade elevada
        if data.get("phq9_score", 0) >= 20:
            risk_score += 1
        
        # Ansiedade comórbida
        if data.get("gad7_score", 0) >= 10:
            risk_score += 1
        
        if risk_score <= 1:
            return "low"
        elif risk_score <= 2:
            return "moderate"
        else:
            return "high"


def extract_psychosocial_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extrair features psicossociais do DataFrame.
    
    Args:
        df: DataFrame com dados psicossociais
    
    Returns:
        DataFrame com features psicossociais
    """
    extractor = PsychosocialFeatureExtractor()
    return extractor.extract_features(df)