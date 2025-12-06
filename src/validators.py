"""
Módulo de Validação de Dados
Valida dados clínicos, genéticos e neurobiológicos
"""

from typing import Dict, List, Tuple, Optional
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Exceção de validação"""
    pass


class DataQualityScore:
    """Calcula score de qualidade dos dados"""
    
    def __init__(self):
        self.checks_passed = 0
        self.checks_total = 0
        self.issues = []
    
    def add_check(self, passed: bool, message: str = ""):
        """Adicionar resultado de check"""
        self.checks_total += 1
        if passed:
            self.checks_passed += 1
        else:
            self.issues.append(message)
    
    def get_score(self) -> float:
        """Obter score de qualidade (0-1)"""
        if self.checks_total == 0:
            return 0.0
        return self.checks_passed / self.checks_total
    
    def get_report(self) -> Dict:
        """Gerar relatório de qualidade"""
        return {
            "quality_score": self.get_score(),
            "checks_passed": self.checks_passed,
            "checks_total": self.checks_total,
            "issues": self.issues,
            "status": "PASSED" if self.get_score() >= 0.8 else "WARNING"
        }


# ============================================================================
# VALIDADORES CLÍNICOS
# ============================================================================

class ClinicalValidator:
    """Valida dados clínicos"""
    
    @staticmethod
    def validate_clinical_assessment(data: Dict) -> Tuple[bool, List[str]]:
        """
        Valida avaliação clínica completa
        Retorna (is_valid, list_of_errors)
        """
        errors = []
        quality = DataQualityScore()
        
        # Validar PHQ-9
        phq9 = data.get("phq9_score")
        if phq9 is None:
            errors.append("PHQ-9 score é obrigatório")
            quality.add_check(False, "PHQ-9 faltando")
        elif not (0 <= phq9 <= 27):
            errors.append(f"PHQ-9 score {phq9} fora do range 0-27")
            quality.add_check(False, f"PHQ-9 inválido: {phq9}")
        else:
            quality.add_check(True)
        
        # Validar Hamilton
        hamilton = data.get("hamilton_score")
        if hamilton is None:
            errors.append("Hamilton score é obrigatório")
            quality.add_check(False, "Hamilton faltando")
        elif not (0 <= hamilton <= 52):
            errors.append(f"Hamilton score {hamilton} fora do range 0-52")
            quality.add_check(False, f"Hamilton inválido: {hamilton}")
        else:
            quality.add_check(True)
        
        # Validar MADRS
        madrs = data.get("madrs_score")
        if madrs is None:
            errors.append("MADRS score é obrigatório")
            quality.add_check(False, "MADRS faltando")
        elif not (0 <= madrs <= 60):
            errors.append(f"MADRS score {madrs} fora do range 0-60")
            quality.add_check(False, f"MADRS inválido: {madrs}")
        else:
            quality.add_check(True)
        
        # Validar episódios anteriores
        num_episodes = data.get("num_previous_episodes", 0)
        if not isinstance(num_episodes, int) or num_episodes < 0:
            errors.append("Número de episódios anteriores deve ser número não-negativo")
            quality.add_check(False, "Episódios inválido")
        else:
            quality.add_check(True)
        
        # Validar duração do episódio
        duration = data.get("episode_duration_months")
        if duration and (not isinstance(duration, int) or duration < 0):
            errors.append("Duração do episódio deve ser número não-negativo")
            quality.add_check(False, "Duração inválida")
        else:
            quality.add_check(True)
        
        # Validar idade de onset
        age_onset = data.get("age_of_onset")
        if age_onset and (not isinstance(age_onset, int) or age_onset < 0 or age_onset > 120):
            errors.append(f"Idade de onset {age_onset} inválida")
            quality.add_check(False, "Idade de onset inválida")
        else:
            quality.add_check(True)
        
        # Validar medicações anteriores (se presentes)
        medications = data.get("medication_trials", [])
        if not isinstance(medications, list):
            errors.append("medication_trials deve ser uma lista")
            quality.add_check(False, "Medicações inválidas")
        else:
            quality.add_check(True)
        
        # Validar booleanos
        boolean_fields = ["anxiety_disorder", "substance_abuse", "personality_disorder", 
                         "suicidal_ideation", "family_history_depression"]
        for field in boolean_fields:
            value = data.get(field, False)
            if not isinstance(value, bool):
                errors.append(f"{field} deve ser booleano")
                quality.add_check(False, f"{field} inválido")
            else:
                quality.add_check(True)
        
        logger.info(f"Validação clínica: {len(errors)} erros, Score: {quality.get_score():.1%}")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_phq9_scores(scores: List[float]) -> bool:
        """Valida série temporal de PHQ-9"""
        if not scores or len(scores) < 2:
            return False
        
        for score in scores:
            if not (0 <= score <= 27):
                logger.warning(f"PHQ-9 score fora de range: {score}")
                return False
        
        return True
    
    @staticmethod
    def validate_suicidal_ideation_severity(severity: int) -> bool:
        """Valida severidade de ideação suicida (0-10)"""
        return 0 <= severity <= 10


# ============================================================================
# VALIDADORES GENÉTICOS
# ============================================================================

class GeneticValidator:
    """Valida dados genéticos"""
    
    # Genótipos válidos
    VALID_5HTTLPR = ["L/L", "L/S", "S/S", "Unknown"]
    VALID_BDNF = ["Val/Val", "Val/Met", "Met/Met", "Unknown"]
    VALID_COMT = ["Val/Val", "Val/Met", "Met/Met", "Unknown"]
    VALID_MTHFR = ["C/C", "C/T", "T/T", "Unknown"]
    VALID_FKBP5 = ["F/F", "F/T", "T/T", "Unknown"]
    VALID_CYP2D6 = ["Ultra-rapid", "Rapid", "Normal", "Intermediate", "Poor", "Unknown"]
    VALID_CYP3A4 = ["Low", "Normal", "High", "Unknown"]
    
    @staticmethod
    def validate_genetic_profile(data: Dict) -> Tuple[bool, List[str]]:
        """Valida perfil genético completo"""
        errors = []
        quality = DataQualityScore()
        
        # Validar 5-HTTLPR
        genotype_5httlpr = data.get("serotonin_transporter_genotype", "Unknown")
        if genotype_5httlpr not in GeneticValidator.VALID_5HTTLPR:
            errors.append(f"5-HTTLPR genótipo inválido: {genotype_5httlpr}")
            quality.add_check(False, "5-HTTLPR inválido")
        else:
            quality.add_check(True)
        
        # Validar BDNF
        bdnf = data.get("bdnf_genotype", "Unknown")
        if bdnf not in GeneticValidator.VALID_BDNF:
            errors.append(f"BDNF genótipo inválido: {bdnf}")
            quality.add_check(False, "BDNF inválido")
        else:
            quality.add_check(True)
        
        # Validar COMT
        comt = data.get("comt_genotype", "Unknown")
        if comt not in GeneticValidator.VALID_COMT:
            errors.append(f"COMT genótipo inválido: {comt}")
            quality.add_check(False, "COMT inválido")
        else:
            quality.add_check(True)
        
        # Validar MTHFR
        mthfr = data.get("mthfr_genotype", "Unknown")
        if mthfr not in GeneticValidator.VALID_MTHFR:
            errors.append(f"MTHFR genótipo inválido: {mthfr}")
            quality.add_check(False, "MTHFR inválido")
        else:
            quality.add_check(True)
        
        # Validar FKBP5
        fkbp5 = data.get("fkbp5_genotype", "Unknown")
        if fkbp5 not in GeneticValidator.VALID_FKBP5:
            errors.append(f"FKBP5 genótipo inválido: {fkbp5}")
            quality.add_check(False, "FKBP5 inválido")
        else:
            quality.add_check(True)
        
        # Validar CYP2D6
        cyp2d6 = data.get("cyp2d6_phenotype", "Unknown")
        if cyp2d6 not in GeneticValidator.VALID_CYP2D6:
            errors.append(f"CYP2D6 fenótipo inválido: {cyp2d6}")
            quality.add_check(False, "CYP2D6 inválido")
        else:
            quality.add_check(True)
        
        # Validar CYP3A4
        cyp3a4 = data.get("cyp3a4_activity", "Unknown")
        if cyp3a4 not in GeneticValidator.VALID_CYP3A4:
            errors.append(f"CYP3A4 atividade inválida: {cyp3a4}")
            quality.add_check(False, "CYP3A4 inválido")
        else:
            quality.add_check(True)
        
        logger.info(f"Validação genética: {len(errors)} erros, Score: {quality.get_score():.1%}")
        
        return len(errors) == 0, errors


# ============================================================================
# VALIDADORES NEUROBIOLÓGICOS
# ============================================================================

class NeurobiologicalValidator:
    """Valida marcadores neurobiológicos"""
    
    # Ranges de valores normais
    CORTISOL_MORNING_NORMAL = (10.0, 20.0)  # mcg/dL
    CORTISOL_EVENING_NORMAL = (3.0, 10.0)   # mcg/dL
    IL6_NORMAL = (0.5, 3.0)                 # pg/mL
    TNF_ALPHA_NORMAL = (1.0, 10.0)          # pg/mL
    CRP_NORMAL = (0.0, 3.0)                 # mg/L
    HIPPOCAMPAL_VOLUME_NORMAL = (3500, 4000)  # mm³
    
    @staticmethod
    def validate_neurobiological_markers(data: Dict) -> Tuple[bool, List[str]]:
        """Valida marcadores neurobiológicos"""
        errors = []
        quality = DataQualityScore()
        
        # Cortisol
        cortisol_morning = data.get("cortisol_morning_mcg_dl")
        if cortisol_morning is not None:
            if cortisol_morning < 0:
                errors.append(f"Cortisol matinal negativo: {cortisol_morning}")
                quality.add_check(False, "Cortisol matinal inválido")
            elif cortisol_morning > 50:
                logger.warning(f"Cortisol matinal muito elevado: {cortisol_morning}")
                quality.add_check(False, "Cortisol matinal elevado")
            else:
                quality.add_check(True)
        else:
            quality.add_check(False, "Cortisol matinal faltando")
        
        # Volume hipocampal
        hippo_volume = data.get("hippocampal_volume_mm3")
        if hippo_volume is not None:
            if hippo_volume < 2000 or hippo_volume > 5000:
                errors.append(f"Volume hipocampal inválido: {hippo_volume} mm³")
                quality.add_check(False, "Volume hipocampal inválido")
            else:
                quality.add_check(True)
        else:
            quality.add_check(False, "Volume hipocampal faltando")
        
        # Sleep quality
        sleep_quality = data.get("sleep_quality_score")
        if sleep_quality is not None:
            if not (0 <= sleep_quality <= 100):
                errors.append(f"Sleep quality score {sleep_quality} fora de range 0-100")
                quality.add_check(False, "Sleep quality inválido")
            else:
                quality.add_check(True)
        else:
            quality.add_check(False, "Sleep quality faltando")
        
        logger.info(f"Validação neurobiológica: {len(errors)} erros, Score: {quality.get_score():.1%}")
        
        return len(errors) == 0, errors


# ============================================================================
# VALIDADORES PSICOSSOCIAIS
# ============================================================================

class PsychosocialValidator:
    """Valida fatores psicossociais"""
    
    VALID_MARITAL_STATUS = ["Single", "Married", "Divorced", "Widowed", "Unknown"]
    VALID_EMPLOYMENT = ["Employed", "Unemployed", "Student", "Retired", "Disabled"]
    VALID_STRESS_LEVELS = ["Low", "Moderate", "High"]
    
    @staticmethod
    def validate_psychosocial_factors(data: Dict) -> Tuple[bool, List[str]]:
        """Valida fatores psicossociais"""
        errors = []
        quality = DataQualityScore()
        
        # Status marital
        marital = data.get("marital_status", "Unknown")
        if marital not in PsychosocialValidator.VALID_MARITAL_STATUS:
            errors.append(f"Status marital inválido: {marital}")
            quality.add_check(False, "Status marital inválido")
        else:
            quality.add_check(True)
        
        # Employment
        employment = data.get("employment_status", "Unknown")
        if employment and employment not in PsychosocialValidator.VALID_EMPLOYMENT:
            errors.append(f"Status de emprego inválido: {employment}")
            quality.add_check(False, "Employment status inválido")
        else:
            quality.add_check(True)
        
        # Social support scores
        social_support = data.get("perceived_support_score")
        if social_support is not None:
            if not (0 <= social_support <= 10):
                errors.append(f"Suporte social {social_support} fora de range 0-10")
                quality.add_check(False, "Social support score inválido")
            else:
                quality.add_check(True)
        else:
            quality.add_check(False, "Social support score faltando")
        
        # Quality of life
        qol = data.get("quality_of_life_score")
        if qol is not None:
            if not (0 <= qol <= 100):
                errors.append(f"Quality of life {qol} fora de range 0-100")
                quality.add_check(False, "QoL score inválido")
            else:
                quality.add_check(True)
        else:
            quality.add_check(False, "QoL score faltando")
        
        # ACE score
        ace = data.get("ace_score")
        if ace is not None:
            if not (0 <= ace <= 10):
                errors.append(f"ACE score {ace} fora de range 0-10")
                quality.add_check(False, "ACE score inválido")
            else:
                quality.add_check(True)
        else:
            quality.add_check(False, "ACE score faltando")
        
        logger.info(f"Validação psicossocial: {len(errors)} erros, Score: {quality.get_score():.1%}")
        
        return len(errors) == 0, errors


# ============================================================================
# VALIDADOR COMPLETO
# ============================================================================

class ComprehensiveValidator:
    """Valida dados completos (clínicos + genéticos + neurobiológicos + psicossociais)"""
    
    @staticmethod
    def validate_complete_patient_data(
        clinical: Dict,
        genetic: Dict,
        neurobiological: Dict,
        psychosocial: Dict
    ) -> Dict:
        """
        Valida todos os dados do paciente
        Retorna relatório completo de validação
        """
        
        clinical_valid, clinical_errors = ClinicalValidator.validate_clinical_assessment(clinical)
        genetic_valid, genetic_errors = GeneticValidator.validate_genetic_profile(genetic)
        neuro_valid, neuro_errors = NeurobiologicalValidator.validate_neurobiological_markers(neurobiological)
        psycho_valid, psycho_errors = PsychosocialValidator.validate_psychosocial_factors(psychosocial)
        
        all_valid = clinical_valid and genetic_valid and neuro_valid and psycho_valid
        all_errors = clinical_errors + genetic_errors + neuro_errors + psycho_errors
        
        return {
            "is_valid": all_valid,
            "total_errors": len(all_errors),
            "errors": all_errors,
            "sections": {
                "clinical": {"valid": clinical_valid, "errors": clinical_errors},
                "genetic": {"valid": genetic_valid, "errors": genetic_errors},
                "neurobiological": {"valid": neuro_valid, "errors": neuro_errors},
                "psychosocial": {"valid": psycho_valid, "errors": psycho_errors}
            },
            "ready_for_prediction": all_valid
        }