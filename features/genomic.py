"""
Engenharia de features genômicas para resposta a antidepressivos.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

from src.constants import CORE_GENES, MetabolizerStatus, GENOTYPE_INTERPRETATION

logger = logging.getLogger(__name__)


class GenomicFeatureExtractor:
    """Extrator de features genômicas."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.gene_importance = self._get_gene_importance()
    
    def _get_gene_importance(self) -> Dict[str, float]:
        """Importância de cada gene para resposta antidepressiva."""
        return {
            "CYP2D6": 0.25,      # Metabolismo crítico
            "CYP3A4": 0.20,      # Metabolismo importante
            "CYP1A2": 0.08,      # Metabolismo secundário
            "CYP2C19": 0.10,     # Metabolismo importante
            "CYP2B6": 0.05,      # Metabolismo menor
            "MTHFR": 0.10,       # Metabolismo de folato
            "COMT": 0.08,        # Degradação de neurotransmissores
            "BDNF": 0.12,        # Neuroplasticidade
            "5HTR1A": 0.08,      # Receptor de serotonina
            "TPH1": 0.06,        # Síntese de serotonina
            "SLC6A4": 0.08       # Transportador de serotonina
        }
    
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extrair features genômicas.
        
        Args:
            df: DataFrame com dados genômicos
        
        Returns:
            DataFrame com features genômicas derivadas
        """
        self.log("Extraindo features genômicas...")
        
        features = pd.DataFrame(index=df.index)
        
        # 1. Features de genótipo individual
        for gene in CORE_GENES:
            col_name = f"{gene}_genotype"
            if col_name in df.columns:
                features[col_name] = df[col_name]
                features[f"{gene}_heterozygous"] = (df[col_name] == 1).astype(int)
                features[f"{gene}_homozygous_alt"] = (df[col_name] == 2).astype(int)
        
        # 2. Score poligênico de metabolização
        features["metabolizer_score"] = self._calculate_metabolizer_score(df)
        
        # 3. Score poligênico de serotonina
        features["serotonin_pathway_score"] = self._calculate_serotonin_score(df)
        
        # 4. Score poligênico de neuroplasticidade
        features["neuroplasticity_score"] = self._calculate_neuroplasticity_score(df)
        
        # 5. Status de metabolizador categórico
        if "metabolizer_status" in df.columns:
            status_map = {
                "poor": 0,
                "intermediate": 1,
                "extensive": 2,
                "ultra": 3
            }
            features["metabolizer_status_numeric"] = df["metabolizer_status"].map(status_map)
        
        # 6. Número de alelos de risco
        features["risk_allele_count"] = self._count_risk_alleles(df)
        
        # 7. Heterozigosidade genômica
        features["genomic_heterozygosity"] = self._calculate_heterozygosity(df)
        
        self.log(f"✓ Features genômicas extraídas: {features.shape[1]} features")
        
        return features
    
    def _calculate_metabolizer_score(self, df: pd.DataFrame) -> pd.Series:
        """Calcular score de metabolização."""
        score = pd.Series(0.0, index=df.index)
        
        metabolizer_genes = ["CYP2D6", "CYP3A4", "CYP2C19", "CYP1A2", "CYP2B6"]
        
        for gene in metabolizer_genes:
            col_name = f"{gene}_genotype"
            if col_name in df.columns:
                weight = self.gene_importance[gene]
                score += weight * (df[col_name] / 2.0)
        
        # Normalizar entre 0 e 1
        return score / sum([self.gene_importance[g] for g in metabolizer_genes])
    
    def _calculate_serotonin_score(self, df: pd.DataFrame) -> pd.Series:
        """Calcular score da via de serotonina."""
        score = pd.Series(0.0, index=df.index)
        
        serotonin_genes = ["SLC6A4", "5HTR1A", "TPH1"]
        
        for gene in serotonin_genes:
            col_name = f"{gene}_genotype"
            if col_name in df.columns:
                weight = self.gene_importance[gene]
                score += weight * (df[col_name] / 2.0)
        
        return score / sum([self.gene_importance[g] for g in serotonin_genes])
    
    def _calculate_neuroplasticity_score(self, df: pd.DataFrame) -> pd.Series:
        """Calcular score de neuroplasticidade."""
        score = pd.Series(0.0, index=df.index)
        
        # BDNF é o principal marcador de neuroplasticidade
        if "BDNF_genotype" in df.columns:
            score = (df["BDNF_genotype"] / 2.0) * self.gene_importance["BDNF"]
        
        return score
    
    def _count_risk_alleles(self, df: pd.DataFrame) -> pd.Series:
        """Contar número de alelos de risco."""
        count = pd.Series(0, index=df.index)
        
        for gene in CORE_GENES:
            col_name = f"{gene}_genotype"
            if col_name in df.columns:
                # Genótipo 0 = risco maior (homozigoto ref)
                count += (2 - df[col_name])
        
        return count
    
    def _calculate_heterozygosity(self, df: pd.DataFrame) -> pd.Series:
        """Calcular heterozigosidade genômica."""
        n_hetero = pd.Series(0, index=df.index)
        n_genes = 0
        
        for gene in CORE_GENES:
            col_name = f"{gene}_genotype"
            if col_name in df.columns:
                n_hetero += (df[col_name] == 1).astype(int)
                n_genes += 1
        
        return n_hetero / n_genes if n_genes > 0 else pd.Series(0, index=df.index)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Obter importância de features."""
        return self.gene_importance.copy()
    
    def log(self, message: str):
        """Log."""
        if self.verbose:
            logger.info(message)


class GenomicInterpreter:
    """Interpretação de achados genômicos."""
    
    @staticmethod
    def interpret_cyp2d6_activity(genotype: int) -> Tuple[str, str]:
        """
        Interpretar atividade CYP2D6.
        
        Args:
            genotype: Genótipo (0, 1, 2)
        
        Returns:
            Tupla (activity_level, interpretation)
        """
        interpretations = {
            0: ("low", "Homozigoto para alelo de baixa atividade"),
            1: ("intermediate", "Heterozigoto"),
            2: ("high", "Homozigoto para alelo de alta atividade")
        }
        return interpretations.get(genotype, ("unknown", "Genótipo desconhecido"))
    
    @staticmethod
    def get_medication_recommendations(
        genotypes: Dict[str, int],
        metabolizer_status: str
    ) -> Dict[str, List[str]]:
        """
        Obter recomendações de medicamentos baseado em genótipo.
        
        Args:
            genotypes: Dicionário de genótipos
            metabolizer_status: Status de metabolizador
        
        Returns:
            Dicionário com recomendações
        """
        recommendations = {
            "recommended": [],
            "caution": [],
            "avoid": []
        }
        
        # Baseado em CYP2D6 status
        if metabolizer_status == "poor":
            recommendations["avoid"].extend(["Fluoxetine", "Paroxetine"])
            recommendations["caution"].extend(["Sertraline", "Citalopram"])
            recommendations["recommended"].extend(["Escitalopram", "Mirtazapine"])
        
        elif metabolizer_status == "intermediate":
            recommendations["caution"].extend(["Fluoxetine", "Paroxetine"])
            recommendations["recommended"].extend(["Sertraline", "Citalopram"])
        
        elif metabolizer_status == "extensive":
            recommendations["recommended"].extend([
                "Sertraline", "Fluoxetine", "Paroxetine", "Citalopram"
            ])
        
        elif metabolizer_status == "ultra":
            recommendations["caution"].extend(["Sertraline", "Paroxetine"])
            recommendations["recommended"].extend(["Fluoxetine", "Citalopram"])
        
        return recommendations
    
    @staticmethod
    def get_contraindications(genotypes: Dict[str, int]) -> List[str]:
        """
        Obter contra-indicações baseado em genótipo.
        
        Args:
            genotypes: Dicionário de genótipos
        
        Returns:
            Lista de contra-indicações
        """
        contraindications = []
        
        # CYP2D6 homozigoto mutante pode causar toxicidade
        if genotypes.get("CYP2D6_genotype") == 0:
            contraindications.append(
                "Metabolizador lento: risco aumentado de efeitos adversos com "
                "substâncias CYP2D6"
            )
        
        # MTHFR pode afetar metabolismo de folato
        if genotypes.get("MTHFR_genotype") == 0:
            contraindications.append(
                "MTHFR C677T: Considerar suplementação de metilfolato"
            )
        
        return contraindications
    
    @staticmethod
    def calculate_genomic_risk_score(
        genotypes: Dict[str, int]
    ) -> float:
        """
        Calcular score genômico de risco.
        
        Args:
            genotypes: Dicionário de genótipos
        
        Returns:
            Score entre 0 e 1 (0=baixo risco, 1=alto risco)
        """
        risk_score = 0.0
        n_genes = 0
        
        for gene, genotype in genotypes.items():
            if "_genotype" in gene:
                # Genótipo 0 = alelo de risco
                risk_score += (2 - genotype) / 2.0
                n_genes += 1
        
        if n_genes == 0:
            return 0.0
        
        return risk_score / n_genes


def extract_genomic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extrair features genômicas do DataFrame.
    
    Args:
        df: DataFrame com dados genômicos
    
    Returns:
        DataFrame com features genômicas
    """
    extractor = GenomicFeatureExtractor()
    return extractor.extract_features(df)