"""
Engenharia de features neurobiológicas para resposta a antidepressivos.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class NeurobiologicalFeatureExtractor:
    """Extrator de features neurobiológicas."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
    
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extrair features neurobiológicas.
        
        Args:
            df: DataFrame com biomarcadores
        
        Returns:
            DataFrame com features derivadas
        """
        self.log("Extraindo features neurobiológicas...")
        
        features = pd.DataFrame(index=df.index)
        
        # 1. Features de Inflamação
        features = self._add_inflammatory_features(features, df)
        
        # 2. Features do Eixo HPA
        features = self._add_hpa_features(features, df)
        
        # 3. Features de Neurotransmissores
        features = self._add_neurotransmitter_features(features, df)
        
        # 4. Features de Triptofano/Quinurenina
        features = self._add_tryptophan_features(features, df)
        
        # 5. Features compostas
        features = self._add_composite_features(features, df)
        
        self.log(f"✓ Features neurobiológicas extraídas: {features.shape[1]} features")
        
        return features
    
    def _add_inflammatory_features(
        self,
        features: pd.DataFrame,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """Adicionar features de inflamação."""
        
        # IL-6 elevada = pior prognóstico
        if "il6_pg_ml" in df.columns:
            features["il6_elevated"] = (df["il6_pg_ml"] > 5.0).astype(int)
            features["il6_normalized"] = df["il6_pg_ml"] / df["il6_pg_ml"].max()
        
        # TNF-α elevado = inflamação sistêmica
        if "tnf_alpha_pg_ml" in df.columns:
            features["tnf_alpha_elevated"] = (df["tnf_alpha_pg_ml"] > 10.0).astype(int)
        
        # CRP elevado = inflamação crônica
        if "crp_mg_l" in df.columns:
            features["crp_elevated"] = (df["crp_mg_l"] > 3.0).astype(int)
        
        # Score de inflamação composto
        inflammatory_cols = ["il6_pg_ml", "tnf_alpha_pg_ml", "crp_mg_l"]
        available_cols = [col for col in inflammatory_cols if col in df.columns]
        
        if available_cols:
            # Normalizar cada coluna
            normalized = (df[available_cols] - df[available_cols].min()) / \
                        (df[available_cols].max() - df[available_cols].min())
            features["inflammatory_score"] = normalized.mean(axis=1)
        
        return features
    
    def _add_hpa_features(
        self,
        features: pd.DataFrame,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """Adicionar features do Eixo HPA (Hipotalâmico-Pituitário-Adrenal)."""
        
        # Cortisol matinal
        if "cortisol_morning_nmol_l" in df.columns:
            features["cortisol_morning_elevated"] = (df["cortisol_morning_nmol_l"] > 400).astype(int)
            features["cortisol_morning_low"] = (df["cortisol_morning_nmol_l"] < 200).astype(int)
        
        # Cortisol noturno
        if "cortisol_evening_nmol_l" in df.columns:
            features["cortisol_evening_elevated"] = (df["cortisol_evening_nmol_l"] > 200).astype(int)
            features["cortisol_evening_low"] = (df["cortisol_evening_nmol_l"] < 100).astype(int)
        
        # Ritmo circadiano (diferença morning-evening)
        if "cortisol_morning_nmol_l" in df.columns and "cortisol_evening_nmol_l" in df.columns:
            features["cortisol_rhythm"] = \
                df["cortisol_morning_nmol_l"] - df["cortisol_evening_nmol_l"]
            features["cortisol_rhythm_disrupted"] = \
                (features["cortisol_rhythm"] < 50).astype(int)
        
        # ACTH
        if "acth_pm_mlU_l" in df.columns:
            features["acth_elevated"] = (df["acth_pm_mlU_l"] > 5.0).astype(int)
        
        # HPA axis dysfunction score
        hpa_features = []
        if "cortisol_morning_elevated" in features.columns:
            hpa_features.append("cortisol_morning_elevated")
        if "cortisol_evening_elevated" in features.columns:
            hpa_features.append("cortisol_evening_elevated")
        if "cortisol_rhythm_disrupted" in features.columns:
            hpa_features.append("cortisol_rhythm_disrupted")
        
        if hpa_features:
            features["hpa_dysfunction_score"] = features[hpa_features].sum(axis=1) / len(hpa_features)
        
        return features
    
    def _add_neurotransmitter_features(
        self,
        features: pd.DataFrame,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """Adicionar features de neurotransmissores."""
        
        # Serotonina
        if "serotonin_ng_ml" in df.columns:
            features["serotonin_low"] = (df["serotonin_ng_ml"] < 40).astype(int)
            features["serotonin_high"] = (df["serotonin_ng_ml"] > 80).astype(int)
        
        # Dopamina
        if "dopamine_pg_ml" in df.columns:
            features["dopamine_low"] = (df["dopamine_pg_ml"] < 20).astype(int)
        
        # Noradrenalina
        if "noradrenaline_pg_ml" in df.columns:
            features["noradrenaline_low"] = (df["noradrenaline_pg_ml"] < 150).astype(int)
        
        # Score de neurotransmissores
        nt_cols = ["serotonin_ng_ml", "dopamine_pg_ml", "noradrenaline_pg_ml"]
        available_nt = [col for col in nt_cols if col in df.columns]
        
        if available_nt:
            normalized = (df[available_nt] - df[available_nt].min()) / \
                        (df[available_nt].max() - df[available_nt].min())
            features["neurotransmitter_score"] = normalized.mean(axis=1)
        
        return features
    
    def _add_tryptophan_features(
        self,
        features: pd.DataFrame,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """Adicionar features de triptofano/quinurenina."""
        
        # Kynurenine/Tryptophan ratio (indicador de ativação imunológica)
        if "kynurenine_nmol_l" in df.columns and "tryptophan_nmol_l" in df.columns:
            # Evitar divisão por zero
            tryptophan_safe = df["tryptophan_nmol_l"].replace(0, np.nan)
            features["kyn_trp_ratio"] = df["kynurenine_nmol_l"] / tryptophan_safe
            
            # Ratio elevado = ativação imunológica
            features["kyn_trp_ratio_elevated"] = (features["kyn_trp_ratio"] > np.nanpercentile(
                features["kyn_trp_ratio"], 75
            )).astype(int)
        
        return features
    
    def _add_composite_features(
        self,
        features: pd.DataFrame,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """Adicionar features compostas."""
        
        # BDNF score (predictor forte de resposta)
        if "bdnf_ng_ml" in df.columns:
            features["bdnf_low"] = (df["bdnf_ng_ml"] < 50).astype(int)
            features["bdnf_normal"] = ((df["bdnf_ng_ml"] >= 50) & (df["bdnf_ng_ml"] <= 150)).astype(int)
            features["bdnf_high"] = (df["bdnf_ng_ml"] > 150).astype(int)
            
            # BDNF normalizado
            features["bdnf_normalized"] = (df["bdnf_ng_ml"] - df["bdnf_ng_ml"].min()) / \
                                         (df["bdnf_ng_ml"].max() - df["bdnf_ng_ml"].min())
        
        # Neuroinflammation score (combinação de biomarcadores)
        neuroinflammation_cols = []
        if "il6_elevated" in features.columns:
            neuroinflammation_cols.append("il6_elevated")
        if "tnf_alpha_elevated" in features.columns:
            neuroinflammation_cols.append("tnf_alpha_elevated")
        if "kyn_trp_ratio_elevated" in features.columns:
            neuroinflammation_cols.append("kyn_trp_ratio_elevated")
        
        if neuroinflammation_cols:
            features["neuroinflammation_score"] = \
                features[neuroinflammation_cols].sum(axis=1) / len(neuroinflammation_cols)
        
        # Overall neurobiological risk score
        risk_features = []
        if "inflammatory_score" in features.columns:
            risk_features.append("inflammatory_score")
        if "hpa_dysfunction_score" in features.columns:
            risk_features.append("hpa_dysfunction_score")
        if "neuroinflammation_score" in features.columns:
            risk_features.append("neuroinflammation_score")
        
        if risk_features:
            features["neurobiological_risk_score"] = \
                features[risk_features].mean(axis=1)
        
        return features
    
    def log(self, message: str):
        """Log."""
        if self.verbose:
            logger.info(message)


class BiomarkerInterpreter:
    """Interpretação de biomarcadores."""
    
    @staticmethod
    def interpret_biomarker_profile(
        biomarkers: Dict[str, float]
    ) -> Dict[str, str]:
        """
        Interpretar perfil completo de biomarcadores.
        
        Args:
            biomarkers: Dicionário de biomarcadores
        
        Returns:
            Interpretações
        """
        interpretations = {}
        
        # Inflamação
        if biomarkers.get("il6_pg_ml", 0) > 5.0:
            interpretations["inflammation"] = "Elevada"
        else:
            interpretations["inflammation"] = "Normal"
        
        # HPA axis
        cortisol_morning = biomarkers.get("cortisol_morning_nmol_l", 0)
        cortisol_evening = biomarkers.get("cortisol_evening_nmol_l", 0)
        
        if cortisol_morning > 400 or cortisol_evening > 200:
            interpretations["hpa_axis"] = "Hiperativo"
        elif cortisol_morning < 200 or cortisol_evening < 100:
            interpretations["hpa_axis"] = "Hipoativo"
        else:
            interpretations["hpa_axis"] = "Normal"
        
        # Serotonina
        if biomarkers.get("serotonin_ng_ml", 60) < 40:
            interpretations["serotonin"] = "Baixa"
        else:
            interpretations["serotonin"] = "Normal"
        
        # BDNF
        bdnf = biomarkers.get("bdnf_ng_ml", 100)
        if bdnf < 50:
            interpretations["bdnf"] = "Baixo (redução de neuroplasticidade)"
        elif bdnf > 150:
            interpretations["bdnf"] = "Alto (boa neuroplasticidade)"
        else:
            interpretations["bdnf"] = "Normal"
        
        return interpretations
    
    @staticmethod
    def get_treatment_implications(
        biomarkers: Dict[str, float]
    ) -> List[str]:
        """
        Obter implicações para tratamento.
        
        Args:
            biomarkers: Dicionário de biomarcadores
        
        Returns:
            Lista de implicações
        """
        implications = []
        
        # Inflamação elevada
        if biomarkers.get("il6_pg_ml", 0) > 5.0:
            implications.append("Considerar medicações com propriedades anti-inflamatórias")
            implications.append("Avaliar dieta anti-inflamatória e exercício físico")
        
        # HPA axis disfunção
        cortisol_morning = biomarkers.get("cortisol_morning_nmol_l", 0)
        if cortisol_morning > 400:
            implications.append("HPA axis hiperativo: considerar técnicas de manejo de estresse")
        elif cortisol_morning < 200:
            implications.append("HPA axis hipoativo: pode beneficiar de suporte adaptogênico")
        
        # Serotonina baixa
        if biomarkers.get("serotonin_ng_ml", 60) < 40:
            implications.append("Serotonina baixa: SSRI/SNRI como primeira linha")
        
        # BDNF baixo
        if biomarkers.get("bdnf_ng_ml", 100) < 50:
            implications.append("BDNF baixo: importante combinar medicação com psicoterapia")
            implications.append("Considerar exercício físico aeróbico")
        
        return implications


def extract_neurobiological_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extrair features neurobiológicas do DataFrame.
    
    Args:
        df: DataFrame com biomarcadores
    
    Returns:
        DataFrame com features neurobiológicas
    """
    extractor = NeurobiologicalFeatureExtractor()
    return extractor.extract_features(df)