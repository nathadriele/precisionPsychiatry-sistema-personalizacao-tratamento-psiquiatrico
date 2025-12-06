"""
Módulo de Engenharia de Features
Extração e transformação de features genômicas, neurobiológicas e psicossociais
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import logging
import numpy as np
from enum import Enum

logger = logging.getLogger(__name__)


class GenotypicValue(Enum):
    """Codificação de genótipos"""
    HOMOZYGOUS_REF = 0
    HETEROZYGOUS = 1
    HOMOZYGOUS_ALT = 2


@dataclass
class FeatureSet:
    """Conjunto de features processadas"""
    patient_id: str
    feature_vector: Dict[str, float]
    feature_names: List[str]
    feature_descriptions: Dict[str, str]
    missing_values: List[str]
    scaling_info: Dict[str, Tuple[float, float]]  # {feature_name: (mean, std)}


class GenomicFeatureExtractor:
    """Extrai features genômicas"""
    
    # Mapeamento de genótipos para scores de risco
    GENOTYPE_RISK_SCORES = {
        # 5-HTTLPR (Serotonin Transporter)
        "5-HTTLPR_L_L": {"risk": 0.3, "interpretation": "Low risk - High serotonin reuptake"},
        "5-HTTLPR_L_S": {"risk": 0.5, "interpretation": "Intermediate risk"},
        "5-HTTLPR_S_S": {"risk": 0.8, "interpretation": "High risk - Low serotonin reuptake"},
        
        # BDNF Val66Met (Brain-Derived Neurotrophic Factor)
        "BDNF_Val_Val": {"risk": 0.4, "interpretation": "Normal BDNF activity"},
        "BDNF_Val_Met": {"risk": 0.5, "interpretation": "Reduced BDNF activity"},
        "BDNF_Met_Met": {"risk": 0.7, "interpretation": "Significantly reduced BDNF activity"},
        
        # COMT Val158Met (Catechol-O-Methyltransferase)
        "COMT_Val_Val": {"risk": 0.3, "interpretation": "Efficient dopamine metabolism"},
        "COMT_Val_Met": {"risk": 0.5, "interpretation": "Intermediate dopamine metabolism"},
        "COMT_Met_Met": {"risk": 0.7, "interpretation": "Slower dopamine metabolism - may increase anxiety"},
        
        # MTHFR C677T (Methylenetetrahydrofolate Reductase)
        "MTHFR_C_C": {"risk": 0.2, "interpretation": "Normal enzyme activity"},
        "MTHFR_C_T": {"risk": 0.4, "interpretation": "Heterozygous - 35% reduced activity"},
        "MTHFR_T_T": {"risk": 0.6, "interpretation": "Homozygous - 65% reduced activity"},
        
        # FKBP5 (FK506-binding protein 5) - Gene x Environment
        "FKBP5_T_T": {"risk": 0.2, "interpretation": "Low risk for gene-environment interaction"},
        "FKBP5_T_C": {"risk": 0.5, "interpretation": "Intermediate risk"},
        "FKBP5_C_C": {"risk": 0.8, "interpretation": "High risk for gene-environment interaction with trauma"},
    }
    
    # Fenótipos CYP para ajuste de dose
    CYP_METABOLIZER_STATUS = {
        "Ultra-rapid": {"dose_multiplier": 1.5, "efficacy_risk": "lower"},
        "Rapid": {"dose_multiplier": 1.2, "efficacy_risk": "slightly_lower"},
        "Normal": {"dose_multiplier": 1.0, "efficacy_risk": "normal"},
        "Intermediate": {"dose_multiplier": 0.8, "efficacy_risk": "slightly_higher"},
        "Poor": {"dose_multiplier": 0.6, "efficacy_risk": "higher"}
    }
    
    def __init__(self):
        self.logger = logger
    
    def extract_genomic_features(self, genetic_profile: Dict) -> Dict[str, float]:
        """Extrai features numéricas do perfil genético"""
        
        features = {}
        
        # 5-HTTLPR score
        genotype = genetic_profile.get("serotonin_transporter_genotype", "Unknown")
        features["5httlpr_risk_score"] = self.GENOTYPE_RISK_SCORES.get(
            f"5-HTTLPR_{genotype}", {"risk": 0.5}
        )["risk"]
        
        # BDNF score
        bdnf = genetic_profile.get("bdnf_genotype", "Unknown")
        features["bdnf_risk_score"] = self.GENOTYPE_RISK_SCORES.get(
            f"BDNF_{bdnf}", {"risk": 0.5}
        )["risk"]
        
        # COMT score
        comt = genetic_profile.get("comt_genotype", "Unknown")
        features["comt_risk_score"] = self.GENOTYPE_RISK_SCORES.get(
            f"COMT_{comt}", {"risk": 0.5}
        )["risk"]
        
        # MTHFR score
        mthfr = genetic_profile.get("mthfr_genotype", "Unknown")
        features["mthfr_risk_score"] = self.GENOTYPE_RISK_SCORES.get(
            f"MTHFR_{mthfr}", {"risk": 0.5}
        )["risk"]
        
        # FKBP5 score
        fkbp5 = genetic_profile.get("fkbp5_genotype", "Unknown")
        features["fkbp5_risk_score"] = self.GENOTYPE_RISK_SCORES.get(
            f"FKBP5_{fkbp5}", {"risk": 0.5}
        )["risk"]
        
        # Score agregado genômico
        genetic_scores = [features.get(f"{key}_risk_score", 0.5) for key in 
                         ["5httlpr", "bdnf", "comt", "mthfr", "fkbp5"]]
        features["genomic_risk_aggregate"] = np.mean(genetic_scores)
        
        # CYP2D6 metabolizer status
        cyp2d6 = genetic_profile.get("CYP2D6_phenotype", "Normal")
        features["cyp2d6_dose_multiplier"] = self.CYP_METABOLIZER_STATUS.get(
            cyp2d6, {"dose_multiplier": 1.0}
        )["dose_multiplier"]
        
        # CYP3A4 metabolizer status
        cyp3a4 = genetic_profile.get("cyp3a4_activity", "Normal")
        features["cyp3a4_dose_multiplier"] = {
            "Low": 0.7,
            "Normal": 1.0,
            "High": 1.3
        }.get(cyp3a4, 1.0)
        
        return features
    
    def extract_gene_environment_interaction(
        self,
        genetic_profile: Dict,
        psychosocial_factors: Dict
    ) -> Dict[str, float]:
        """Extrai features de interação gene-ambiente"""
        
        features = {}
        
        # FKBP5 x Trauma (Gene x Environment)
        fkbp5_risk = self.GENOTYPE_RISK_SCORES.get(
            f"FKBP5_{genetic_profile.get('fkbp5_genotype', 'Unknown')}",
            {"risk": 0.5}
        )["risk"]
        
        childhood_trauma = 1.0 if psychosocial_factors.get("childhood_trauma", False) else 0.0
        ace_score = psychosocial_factors.get("ace_score", 0) / 10.0  # Normalizar 0-10
        
        # Interação: alto FKBP5 risk + trauma = maior efeito
        features["fkbp5_trauma_interaction"] = fkbp5_risk * (0.5 + ace_score)
        
        # Serotonin x Stress
        serotonin_risk = self.GENOTYPE_RISK_SCORES.get(
            f"5-HTTLPR_{genetic_profile.get('serotonin_transporter_genotype', 'Unknown')}",
            {"risk": 0.5}
        )["risk"]
        
        work_stress = 1.0 if psychosocial_factors.get("work_stress_level") == "high" else 0.5
        major_stressors = len(psychosocial_factors.get("major_life_stressors", [])) / 5.0
        
        features["serotonin_stress_interaction"] = serotonin_risk * (0.5 + major_stressors)
        
        return features


class NeurobiologicalFeatureExtractor:
    """Extrai features de marcadores neurobiológicos"""
    
    def __init__(self):
        self.logger = logger
    
    def extract_neurobiological_features(
        self,
        neurobiological_markers: Dict
    ) -> Dict[str, float]:
        """Extrai features de marcadores neurobiológicos"""
        
        features = {}
        
        # Normalizar cortisol (valores típicos: 10-20 mcg/dL pela manhã, 3-10 mcg/dL à noite)
        morning_cortisol = neurobiological_markers.get("cortisol_morning_mcg_dl", 12)
        evening_cortisol = neurobiological_markers.get("cortisol_evening_mcg_dl", 5)
        
        features["cortisol_morning_normalized"] = min(morning_cortisol / 20.0, 1.0)
        features["cortisol_evening_normalized"] = min(evening_cortisol / 10.0, 1.0)
        
        # Cortisol rhythm (diferença manhã-noite deveria ser alta)
        cortisol_rhythm = (morning_cortisol - evening_cortisol) / 15.0
        features["cortisol_rhythm_score"] = min(max(cortisol_rhythm, 0), 1.0)
        
        # Marcadores inflamatórios
        inflammatory_markers = neurobiological_markers.get("inflammatory_markers", {})
        if inflammatory_markers:
            il6 = inflammatory_markers.get("IL6", 2.0) / 10.0  # Normal ~1-3 pg/mL
            tnf_alpha = inflammatory_markers.get("TNF_alpha", 5.0) / 20.0  # Normal ~5 pg/mL
            crp = inflammatory_markers.get("CRP", 3.0) / 10.0  # Normal <3 mg/L
            
            features["il6_normalized"] = min(il6, 1.0)
            features["tnf_alpha_normalized"] = min(tnf_alpha, 1.0)
            features["crp_normalized"] = min(crp, 1.0)
            features["inflammation_aggregate"] = np.mean([il6, tnf_alpha, crp])
        
        # EEG features
        if neurobiological_markers.get("eeg_theta_power"):
            features["eeg_theta_power_normalized"] = min(
                neurobiological_markers["eeg_theta_power"] / 100.0, 1.0
            )
        
        if neurobiological_markers.get("eeg_alpha_power"):
            features["eeg_alpha_power_normalized"] = min(
                neurobiological_markers["eeg_alpha_power"] / 50.0, 1.0
            )
        
        # Assimetria frontal (theta anterior direito vs esquerdo)
        # Score positivo indica depressão
        features["frontal_asymmetry_index"] = neurobiological_markers.get(
            "frontal_asymmetry", 0.0
        )
        
        # Neuroimagem
        if neurobiological_markers.get("hippocampal_volume_mm3"):
            # Volume normal ~3500-4000 mm³
            hippo_vol = neurobiological_markers["hippocampal_volume_mm3"]
            features["hippocampal_volume_normalized"] = hippo_vol / 4000.0
        
        if neurobiological_markers.get("prefrontal_cortex_activation"):
            features["prefrontal_activation"] = min(
                neurobiological_markers["prefrontal_cortex_activation"], 1.0
            )
        
        if neurobiological_markers.get("amygdala_reactivity"):
            features["amygdala_hyperreactivity"] = min(
                neurobiological_markers["amygdala_reactivity"], 1.0
            )
        
        # Sono
        sleep_quality = neurobiological_markers.get("sleep_quality_score", 5)
        features["sleep_quality_normalized"] = min(sleep_quality / 10.0, 1.0)
        
        # Cognição
        cognitive = neurobiological_markers.get("cognitive_performance_score", 0.5)
        features["cognitive_impairment"] = 1.0 - min(cognitive, 1.0)
        
        return features
    
    def extract_hpa_axis_dysfunction(
        self,
        neurobiological_markers: Dict
    ) -> Dict[str, float]:
        """Extrai features de disfunção do eixo HPA"""
        
        features = {}
        
        morning_cortisol = neurobiological_markers.get("cortisol_morning_mcg_dl", 12)
        evening_cortisol = neurobiological_markers.get("cortisol_evening_mcg_dl", 5)
        
        # Padrão de cortisol elevado persistente (marker de depressão)
        features["elevated_cortisol"] = 1.0 if morning_cortisol > 20 else min(morning_cortisol / 20, 0.8)
        
        # Rhythmicity loss (depressão)
        features["cortisol_rhythmicity_loss"] = 1.0 - features.get("cortisol_rhythm_score", 0.5)
        
        # HPA axis overactivation score
        features["hpa_overactivation"] = min(
            (morning_cortisol / 20.0 + (1 - features.get("cortisol_rhythm_score", 0.5))) / 2,
            1.0
        )
        
        return features


class PsychosocialFeatureExtractor:
    """Extrai features de fatores psicossociais"""
    
    def __init__(self):
        self.logger = logger
    
    def extract_psychosocial_features(
        self,
        psychosocial_factors: Dict
    ) -> Dict[str, float]:
        """Extrai features de fatores psicossociais"""
        
        features = {}
        
        # Suporte social
        features["social_support_score"] = min(
            psychosocial_factors.get("perceived_support_score", 5) / 10.0, 1.0
        )
        features["social_isolation"] = 1.0 if psychosocial_factors.get("social_isolation", False) else 0.0
        features["close_relationships"] = min(
            psychosocial_factors.get("close_relationships", 0) / 5.0, 1.0
        )
        
        # Estresse
        num_stressors = len(psychosocial_factors.get("major_life_stressors", []))
        features["major_life_stressors_count"] = min(num_stressors / 5.0, 1.0)
        features["recent_trauma"] = 1.0 if psychosocial_factors.get("recent_trauma", False) else 0.0
        
        # Trauma cumulativo
        ace = psychosocial_factors.get("ace_score", 0)
        features["adverse_childhood_experiences"] = min(ace / 10.0, 1.0)
        
        # Funcionamento ocupacional
        job_satisfaction = psychosocial_factors.get("job_satisfaction_score", 5)
        features["job_satisfaction"] = min(job_satisfaction / 10.0, 1.0)
        
        employment = psychosocial_factors.get("employment_status", "employed")
        features["unemployed"] = 1.0 if employment == "unemployed" else 0.0
        
        # Qualidade de vida
        qol = psychosocial_factors.get("quality_of_life_score", 50)
        features["quality_of_life_normalized"] = min(qol / 100.0, 1.0)
        
        # Propósito/significado
        purpose = psychosocial_factors.get("purpose_meaning_score", 5)
        features["sense_of_purpose"] = min(purpose / 10.0, 1.0)
        
        # Relacionamentos
        relationship_quality = psychosocial_factors.get("relationship_quality_score", 5)
        features["relationship_quality"] = min(relationship_quality / 10.0, 1.0)
        
        # Engajamento em atividades
        activities = psychosocial_factors.get("activities_engagement", 5)
        features["activities_engagement"] = min(activities / 10.0, 1.0)
        
        # Vulnerabilidade psicossocial agregada
        vulnerability_factors = [
            features.get("social_isolation", 0),
            features.get("major_life_stressors_count", 0),
            features.get("adverse_childhood_experiences", 0),
            features.get("unemployed", 0),
            1.0 - features.get("quality_of_life_normalized", 0.5)
        ]
        features["psychosocial_vulnerability_index"] = np.mean(vulnerability_factors)
        
        return features


class FeatureEngineer:
    """Orquestra extração de features de múltiplas fontes"""
    
    def __init__(self):
        self.logger = logger
        self.genomic_extractor = GenomicFeatureExtractor()
        self.neurobiological_extractor = NeurobiologicalFeatureExtractor()
        self.psychosocial_extractor = PsychosocialFeatureExtractor()
    
    def engineer_features(
        self,
        patient_id: str,
        genetic_profile: Optional[Dict] = None,
        neurobiological_markers: Optional[Dict] = None,
        psychosocial_factors: Optional[Dict] = None,
        clinical_assessment: Optional[Dict] = None
    ) -> FeatureSet:
        """
        Engenharia de features completa de múltiplas fontes
        """
        
        self.logger.info(f"Iniciando engenharia de features para {patient_id}")
        
        all_features = {}
        missing_features = []
        
        # Extrair features genômicas
        if genetic_profile:
            genomic_features = self.genomic_extractor.extract_genomic_features(genetic_profile)
            all_features.update(genomic_features)
            
            # Features de interação gene-ambiente
            if psychosocial_factors:
                ge_interaction = self.genomic_extractor.extract_gene_environment_interaction(
                    genetic_profile, psychosocial_factors
                )
                all_features.update(ge_interaction)
        else:
            missing_features.append("genetic_profile")
        
        # Extrair features neurobiológicas
        if neurobiological_markers:
            neuro_features = self.neurobiological_extractor.extract_neurobiological_features(
                neurobiological_markers
            )
            all_features.update(neuro_features)
            
            # Features de disfunção HPA
            hpa_features = self.neurobiological_extractor.extract_hpa_axis_dysfunction(
                neurobiological_markers
            )
            all_features.update(hpa_features)
        else:
            missing_features.append("neurobiological_markers")
        
        # Extrair features psicossociais
        if psychosocial_factors:
            psychosocial_features = self.psychosocial_extractor.extract_psychosocial_features(
                psychosocial_factors
            )
            all_features.update(psychosocial_features)
        else:
            missing_features.append("psychosocial_factors")
        
        # Extrair features clínicas
        if clinical_assessment:
            clinical_features = self._extract_clinical_features(clinical_assessment)
            all_features.update(clinical_features)
        
        # Normalizar features
        scaled_features, scaling_info = self._normalize_features(all_features)
        
        # Criar descriptions
        descriptions = self._create_feature_descriptions(scaled_features.keys())
        
        feature_set = FeatureSet(
            patient_id=patient_id,
            feature_vector=scaled_features,
            feature_names=list(scaled_features.keys()),
            feature_descriptions=descriptions,
            missing_values=missing_features,
            scaling_info=scaling_info
        )
        
        self.logger.info(f"Features engenheiradas: {len(scaled_features)} features, "
                        f"{len(missing_features)} fontes faltando")
        
        return feature_set
    
    def _extract_clinical_features(self, clinical_assessment: Dict) -> Dict[str, float]:
        """Extrai features do assessment clínico"""
        
        features = {}
        
        # Escalas de sintomas
        phq9 = clinical_assessment.get("phq9_score", 15)
        features["phq9_normalized"] = min(phq9 / 27.0, 1.0)
        
        # Severidade
        severity = clinical_assessment.get("severity", "moderate")
        severity_map = {
            "minimal": 0.1,
            "mild": 0.25,
            "moderate": 0.5,
            "moderately_severe": 0.75,
            "severe": 1.0
        }
        features["severity_score"] = severity_map.get(severity, 0.5)
        
        # Histórico
        num_episodes = clinical_assessment.get("num_episodes", 1)
        features["episode_recurrence"] = min(num_episodes / 5.0, 1.0)
        
        return features
    
    def _normalize_features(
        self,
        features: Dict[str, float]
    ) -> Tuple[Dict[str, float], Dict[str, Tuple[float, float]]]:
        """Normaliza features com z-score"""
        
        # Para este exemplo, vamos usar min-max normalization
        scaling_info = {}
        normalized = {}
        
        for name, value in features.items():
            # Já estão em 0-1 na maioria dos casos
            if isinstance(value, (int, float)):
                normalized[name] = min(max(float(value), 0.0), 1.0)
                scaling_info[name] = (0.0, 1.0)
        
        return normalized, scaling_info
    
    def _create_feature_descriptions(self, feature_names: List[str]) -> Dict[str, str]:
        """Cria descrições de features"""
        
        descriptions = {
            "5httlpr_risk_score": "Risk score for serotonin transporter variant",
            "bdnf_risk_score": "Risk score for BDNF Val66Met variant",
            "comt_risk_score": "Risk score for COMT Val158Met variant",
            "genomic_risk_aggregate": "Aggregated genomic risk score",
            "fkbp5_trauma_interaction": "Gene-environment interaction for trauma",
            "serotonin_stress_interaction": "Serotonin system x stress interaction",
            "cortisol_rhythm_score": "Quality of diurnal cortisol rhythm",
            "inflammation_aggregate": "Aggregated inflammatory markers",
            "social_support_score": "Perceived social support",
            "psychosocial_vulnerability_index": "Overall psychosocial vulnerability",
            "quality_of_life_normalized": "Overall quality of life score",
            "phq9_normalized": "PHQ-9 depression symptom severity"
        }
        
        return {name: descriptions.get(name, f"Feature: {name}") for name in feature_names}